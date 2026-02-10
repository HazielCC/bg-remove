"""
PyTorch Dataset for matting tasks.

Handles:
  - HuggingFace datasets (aisegmentcn-matting-human and similar)
  - Local image+alpha pairs
  - Automatic trimap generation via dilation/erosion
  - Synthetic composition (foreground + random backgrounds)
  - Augmentations: blur, noise, JPEG compression, illumination, resize
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import binary_dilation, binary_erosion


class MattingDataset(Dataset):
    """
    Dataset that yields (image, trimap, alpha) tuples for MODNet training.

    Expects a directory structure:
        root/
            images/    # RGB images  (.jpg / .png)
            alphas/    # Alpha mattes (.png, single-channel 0-255)

    Or a manifest file (JSON lines):
        {"image": "path/to/img.jpg", "alpha": "path/to/alpha.png"}
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 512,
        augment: bool = True,
        trimap_dilation: int = 15,
        backgrounds_dir: str | None = None,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment and split == "train"
        self.trimap_dilation = trimap_dilation
        self.backgrounds_dir = Path(backgrounds_dir) if backgrounds_dir else None

        # Discover samples
        self.samples = self._find_samples(split)

        # Load background images list for synthetic composition
        self.bg_images: list[Path] = []
        if self.backgrounds_dir and self.backgrounds_dir.exists():
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            self.bg_images = [
                p for p in self.backgrounds_dir.rglob("*") if p.suffix.lower() in exts
            ]

    def _find_samples(self, split: str) -> list[tuple[Path, Path]]:
        """Find (image_path, alpha_path) pairs."""
        images_dir = self.root / "images"
        alphas_dir = self.root / "alphas"

        # Try split-specific subdirectories first
        if (images_dir / split).exists():
            images_dir = images_dir / split
            alphas_dir = alphas_dir / split

        if not images_dir.exists() or not alphas_dir.exists():
            # Fallback: flat directory
            images_dir = self.root / "images"
            alphas_dir = self.root / "alphas"
            if not images_dir.exists():
                return []

        exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = sorted(
            [f for f in images_dir.iterdir() if f.suffix.lower() in exts]
        )
        samples = []
        for img_path in image_files:
            # Try matching alpha file with same stem
            for ext in [".png", ".jpg", ".jpeg"]:
                alpha_path = alphas_dir / (img_path.stem + ext)
                if alpha_path.exists():
                    samples.append((img_path, alpha_path))
                    break
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, alpha_path = self.samples[idx]

        # Load image and alpha
        img = Image.open(img_path).convert("RGB")
        alpha = Image.open(alpha_path)

        # Extract alpha channel if RGBA
        if alpha.mode == "RGBA":
            alpha = alpha.split()[-1]  # A channel
        elif alpha.mode == "RGB":
            alpha = alpha.convert("L")
        elif alpha.mode != "L":
            alpha = alpha.convert("L")

        # Synthetic composition with random background
        if self.bg_images and random.random() < 0.5:
            img, alpha = self._composite_on_bg(img, alpha)

        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        alpha = alpha.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Augmentation
        if self.augment:
            img, alpha = self._augment(img, alpha)

        # Convert to tensors
        img_t = self._img_to_tensor(img)  # [3, H, W] normalized to [-1, 1]
        alpha_t = self._alpha_to_tensor(alpha)  # [1, H, W] in [0, 1]

        # Generate trimap from alpha
        trimap_t = self._generate_trimap(alpha_t)  # [1, H, W] values: 0, 0.5, 1

        return img_t, trimap_t, alpha_t

    def _composite_on_bg(self, fg: Image.Image, alpha: Image.Image):
        """Composite foreground onto a random background."""
        bg_path = random.choice(self.bg_images)
        bg = Image.open(bg_path).convert("RGB").resize(fg.size, Image.BILINEAR)
        alpha_f = np.array(alpha).astype(np.float32) / 255.0
        fg_np = np.array(fg).astype(np.float32)
        bg_np = np.array(bg).astype(np.float32)
        comp = fg_np * alpha_f[..., None] + bg_np * (1 - alpha_f[..., None])
        return Image.fromarray(comp.astype(np.uint8)), alpha

    def _augment(self, img: Image.Image, alpha: Image.Image):
        """Apply random augmentations."""
        # Random horizontal flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            alpha = alpha.transpose(Image.FLIP_LEFT_RIGHT)

        # Random brightness/contrast
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))

        # Random blur
        if random.random() < 0.2:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius))

        # Random color jitter
        if random.random() < 0.2:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))

        return img, alpha

    @staticmethod
    def _img_to_tensor(img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor normalized to [-1, 1]."""
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

    @staticmethod
    def _alpha_to_tensor(alpha: Image.Image) -> torch.Tensor:
        """Convert alpha PIL to tensor in [0, 1]."""
        arr = np.array(alpha).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

    def _generate_trimap(self, alpha: torch.Tensor) -> torch.Tensor:
        """Generate trimap from alpha matte using dilation/erosion."""
        a = alpha.squeeze(0).numpy()
        fg = (a > 0.9).astype(np.uint8)
        bg = (a < 0.1).astype(np.uint8)

        # Dilate foreground and erode to find transition zone
        struct = np.ones((self.trimap_dilation, self.trimap_dilation))
        fg_dilated = binary_dilation(fg, structure=struct).astype(np.uint8)
        bg_eroded = 1 - binary_dilation(1 - bg, structure=struct).astype(np.uint8)

        trimap = np.zeros_like(a)
        trimap[bg_eroded == 1] = 0.0  # background
        trimap[fg == 1] = 1.0  # foreground
        # Everything else is transition (0.5)
        transition = (fg_dilated == 1) & (fg == 0)
        trimap[transition] = 0.5

        return torch.from_numpy(trimap).unsqueeze(0).float()


class HFMattingDataset(MattingDataset):
    """
    Wrapper to load a HuggingFace dataset and organize it into the
    expected local directory format (images/ + alphas/).

    Supports:
      1. Standard HF `datasets` format (parquet / arrow / image-folder with metadata)
      2. Raw image repos downloaded via `snapshot_download`
      3. RGBA images where the alpha channel IS the matte
    """

    # Common directory names for images / alphas in raw repos
    _IMAGE_DIR_NAMES = {
        "images",
        "image",
        "clip_img",
        "input",
        "rgb",
        "fg",
        "foreground",
        "img",
    }
    _ALPHA_DIR_NAMES = {
        "alpha",
        "alphas",
        "matte",
        "mattes",
        "matting",
        "mask",
        "masks",
        "trimap",
        "trimaps",
        "gt",
    }
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    # ── Public helpers ───────────────────────────────────

    @staticmethod
    def validate_hf_repo(dataset_name: str, hf_token: str | None = None) -> dict:
        """
        Check that a HuggingFace repo actually contains downloadable data.
        Returns a dict with keys: valid, file_count, data_format, message.
        """
        from huggingface_hub import HfApi, hf_hub_url
        import requests as _requests

        api = HfApi(token=hf_token)
        try:
            info = api.dataset_info(dataset_name, token=hf_token)
        except Exception as exc:
            return {
                "valid": False,
                "file_count": 0,
                "data_format": None,
                "message": f"Dataset not found on HuggingFace: {exc}",
            }

        siblings = info.siblings or []
        data_files = [
            s
            for s in siblings
            if not s.rfilename.startswith(".")
            and s.rfilename.lower() not in {"readme.md", "license", "license.md"}
        ]

        if not data_files:
            return {
                "valid": False,
                "file_count": 0,
                "data_format": None,
                "message": (
                    f"El dataset «{dataset_name}» no contiene archivos de datos. "
                    "Puede que haya sido eliminado o que sea solo un README."
                ),
            }

        # Detect format
        exts = {Path(f.rfilename).suffix.lower() for f in data_files}
        image_exts = exts & HFMattingDataset._IMAGE_EXTS
        parquet = ".parquet" in exts
        arrow = ".arrow" in exts

        if parquet or arrow:
            fmt = "datasets"
        elif image_exts:
            fmt = "image_repo"
        else:
            fmt = "unknown"

        return {
            "valid": True,
            "file_count": len(data_files),
            "data_format": fmt,
            "message": f"{len(data_files)} archivos encontrados (formato: {fmt})",
        }

    @staticmethod
    def prepare_from_hf(
        dataset_name: str,
        output_dir: str,
        split: str = "train",
        max_samples: int | None = None,
        hf_token: str | None = None,
    ) -> Path:
        """
        Download a HuggingFace matting dataset and organize as images/ + alphas/.

        Strategy:
          1) Validate the repo has actual files
          2) Try `datasets.load_dataset()` (standard parquet/arrow/imagefolder)
          3) Fallback: `snapshot_download` + auto-detect directory structure
          4) Fallback: treat RGBA images as image+alpha

        Returns the output directory path.
        """
        # ── Pre-validate ─────────────────────────────────
        check = HFMattingDataset.validate_hf_repo(dataset_name, hf_token)
        if not check["valid"]:
            raise ValueError(check["message"])

        out = Path(output_dir)
        images_dir = out / "images" / split
        alphas_dir = out / "alphas" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        alphas_dir.mkdir(parents=True, exist_ok=True)

        saved = 0

        # ── Strategy 1: datasets library ──────────────────
        if check["data_format"] in ("datasets", "unknown"):
            try:
                saved = HFMattingDataset._download_via_datasets(
                    dataset_name,
                    split,
                    max_samples,
                    hf_token,
                    images_dir,
                    alphas_dir,
                )
            except Exception as e:
                print(f"[dataset] datasets.load_dataset failed: {e}")
                # Will try snapshot_download below
                saved = 0

        # ── Strategy 2: snapshot_download for image repos ─
        if saved == 0:
            try:
                saved = HFMattingDataset._download_via_snapshot(
                    dataset_name,
                    max_samples,
                    hf_token,
                    images_dir,
                    alphas_dir,
                )
            except Exception as e:
                print(f"[dataset] snapshot_download failed: {e}")
                saved = 0

        if saved == 0:
            # Clean up empty dirs
            import shutil

            if out.exists():
                shutil.rmtree(out, ignore_errors=True)
            raise ValueError(
                f"No se pudieron extraer imágenes del dataset «{dataset_name}». "
                "El formato del repo no es compatible o no contiene pares imagen/alpha. "
                "Prueba con otro dataset como «Voxel51/DUTS» o sube un dataset local."
            )

        return out

    # ── Private download strategies ──────────────────────

    @staticmethod
    def _download_via_datasets(
        dataset_name: str,
        split: str,
        max_samples: int | None,
        hf_token: str | None,
        images_dir: Path,
        alphas_dir: Path,
    ) -> int:
        """Standard HuggingFace `datasets` download. Returns count of saved images."""
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split, token=hf_token)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        saved = 0
        for i, sample in enumerate(ds):
            img = None
            alpha = None

            for key in ("image", "clip_img", "input_image", "rgb"):
                if key in sample:
                    img = sample[key]
                    break

            for key in ("alpha", "matte", "matting", "mask", "trimap", "label"):
                if key in sample:
                    alpha = sample[key]
                    break

            if img is None:
                continue

            if isinstance(img, str):
                img = Image.open(img)
            if not isinstance(img, Image.Image):
                continue

            # If the image is RGBA and no alpha column, extract from alpha channel
            if alpha is None and img.mode == "RGBA":
                alpha = img.split()[-1]
                img = img.convert("RGB")

            img_path = images_dir / f"{i:06d}.jpg"
            img.convert("RGB").save(img_path, quality=95)
            saved += 1

            if alpha is not None:
                if isinstance(alpha, str):
                    alpha = Image.open(alpha)
                if isinstance(alpha, Image.Image):
                    if alpha.mode == "RGBA":
                        alpha = alpha.split()[-1]
                    elif alpha.mode != "L":
                        alpha = alpha.convert("L")
                    alpha_path = alphas_dir / f"{i:06d}.png"
                    alpha.save(alpha_path)

        return saved

    @staticmethod
    def _download_via_snapshot(
        dataset_name: str,
        max_samples: int | None,
        hf_token: str | None,
        images_dir: Path,
        alphas_dir: Path,
    ) -> int:
        """
        Download individual files from `dataset_name` via hf_hub API.
        Auto-detects image/alpha directory structure from the file listing
        without downloading the entire repo.

        Returns count of saved images.
        """
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=hf_token)
        img_exts = HFMattingDataset._IMAGE_EXTS
        img_dir_names = HFMattingDataset._IMAGE_DIR_NAMES
        alpha_dir_names = HFMattingDataset._ALPHA_DIR_NAMES

        # ── List all files in the repo ───────────────────
        all_files = [
            item.rfilename
            for item in api.list_repo_tree(
                dataset_name,
                repo_type="dataset",
                recursive=True,
                token=hf_token,
            )
            if hasattr(item, "rfilename")
            and Path(item.rfilename).suffix.lower() in img_exts
        ]

        if not all_files:
            return 0

        # ── Detect directory structure ───────────────────
        # Group files by their top-level directory
        dir_groups: dict[str, list[str]] = {}
        flat_files: list[str] = []
        for f in all_files:
            parts = Path(f).parts
            if len(parts) > 1:
                top_dir = parts[0].lower()
                dir_groups.setdefault(top_dir, []).append(f)
            else:
                flat_files.append(f)

        src_image_files: list[str] = []
        src_alpha_files: dict[str, str] = {}  # stem -> rfilename

        # Try to match known dir names
        found_img_dir = None
        found_alpha_dir = None
        for d, files in dir_groups.items():
            if d in img_dir_names and found_img_dir is None:
                found_img_dir = d
                src_image_files = files
            elif d in alpha_dir_names and found_alpha_dir is None:
                found_alpha_dir = d
                src_alpha_files = {Path(f).stem: f for f in files}

        # If no known dirs, use files from the largest directory or flat files
        if not src_image_files:
            if dir_groups:
                # Use the directory with the most files as "images"
                sorted_dirs = sorted(dir_groups.items(), key=lambda x: -len(x[1]))
                main_dir, main_files = sorted_dirs[0]
                src_image_files = main_files
                # If there's a second large dir, use it as alphas
                if len(sorted_dirs) > 1:
                    alpha_dir, alpha_files = sorted_dirs[1]
                    src_alpha_files = {Path(f).stem: f for f in alpha_files}
            else:
                src_image_files = flat_files

        # ── Download and process ─────────────────────────
        limit = max_samples or len(src_image_files)
        saved = 0

        for i, rfilename in enumerate(src_image_files[:limit]):
            try:
                local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=rfilename,
                    repo_type="dataset",
                    token=hf_token,
                )
                img = Image.open(local_path)

                # Handle RGBA as image + alpha
                if img.mode == "RGBA" and not src_alpha_files:
                    alpha = img.split()[-1]
                    img.convert("RGB").save(images_dir / f"{i:06d}.jpg", quality=95)
                    alpha.save(alphas_dir / f"{i:06d}.png")
                    saved += 1
                    continue

                img.convert("RGB").save(images_dir / f"{i:06d}.jpg", quality=95)
                saved += 1

                # Try to download matching alpha
                stem = Path(rfilename).stem
                if stem in src_alpha_files:
                    alpha_rfilename = src_alpha_files[stem]
                    alpha_local = hf_hub_download(
                        repo_id=dataset_name,
                        filename=alpha_rfilename,
                        repo_type="dataset",
                        token=hf_token,
                    )
                    a = Image.open(alpha_local)
                    if a.mode == "RGBA":
                        a = a.split()[-1]
                    elif a.mode != "L":
                        a = a.convert("L")
                    a.save(alphas_dir / f"{i:06d}.png")

            except Exception as exc:
                print(f"[dataset] Skipping {rfilename}: {exc}")
                continue

        return saved

    @staticmethod
    def _copy_flat_images(
        all_images: list[Path],
        max_samples: int | None,
        images_dir: Path,
        alphas_dir: Path,
    ) -> int:
        """Copy a flat list of images, extracting alpha from RGBA if present."""
        limit = max_samples or len(all_images)
        saved = 0
        for i, f in enumerate(all_images[:limit]):
            try:
                img = Image.open(f)
                if img.mode == "RGBA":
                    alpha = img.split()[-1]
                    img.convert("RGB").save(images_dir / f"{i:06d}.jpg", quality=95)
                    alpha.save(alphas_dir / f"{i:06d}.png")
                else:
                    img.convert("RGB").save(images_dir / f"{i:06d}.jpg", quality=95)
                saved += 1
            except Exception:
                continue
        return saved
