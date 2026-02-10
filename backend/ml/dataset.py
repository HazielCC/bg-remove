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
    """

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

        Returns the output directory path.
        """
        from datasets import load_dataset

        out = Path(output_dir)
        images_dir = out / "images" / split
        alphas_dir = out / "alphas" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        alphas_dir.mkdir(parents=True, exist_ok=True)

        ds = load_dataset(dataset_name, split=split, token=hf_token)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        for i, sample in enumerate(ds):
            # Try common column names
            img = None
            alpha = None

            for key in ["image", "clip_img", "input_image", "rgb"]:
                if key in sample:
                    img = sample[key]
                    break

            for key in ["alpha", "matte", "matting", "mask", "trimap"]:
                if key in sample:
                    alpha = sample[key]
                    break

            if img is None:
                continue

            # Convert to PIL if needed
            if isinstance(img, str):
                img = Image.open(img)
            if not isinstance(img, Image.Image):
                continue

            img_path = images_dir / f"{i:06d}.jpg"
            img.convert("RGB").save(img_path, quality=95)

            if alpha is not None:
                if isinstance(alpha, str):
                    alpha = Image.open(alpha)
                if isinstance(alpha, Image.Image):
                    # Extract alpha channel if RGBA
                    if alpha.mode == "RGBA":
                        alpha = alpha.split()[-1]
                    elif alpha.mode != "L":
                        alpha = alpha.convert("L")
                    alpha_path = alphas_dir / f"{i:06d}.png"
                    alpha.save(alpha_path)

        return out
