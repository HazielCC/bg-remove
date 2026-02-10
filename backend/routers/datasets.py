"""
Dataset management API.

Endpoints:
  GET  /search        — search HuggingFace Hub
  GET  /suggested     — list of known-good matting datasets
  POST /validate      — check a dataset has actual data before downloading
  POST /download      — download dataset from HF
  GET  /local         — list locally downloaded datasets
  GET  /{id}/preview  — preview samples
  POST /{id}/curate   — validate & curate dataset
  GET  /{id}/stats    — dataset statistics
"""

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────
class DownloadRequest(BaseModel):
    dataset_name: str  # e.g. "Voxel51/DUTS"
    split: str = "train"
    max_samples: int | None = None


class ValidateRequest(BaseModel):
    dataset_name: str


class CurateRequest(BaseModel):
    min_resolution: int = 256
    check_alpha_range: bool = True
    remove_broken: bool = False


# ── Suggested datasets (known to work) ───────────────────
SUGGESTED_DATASETS = [
    {
        "id": "Voxel51/DUTS",
        "description": "Salient object detection. ~15K images con máscaras binarias.",
        "downloads": 2600,
        "format": "image_repo",
        "recommended": True,
    },
    {
        "id": "chitradrishti/duts",
        "description": "DUTS dataset alternativo, formato datasets estándar.",
        "downloads": 8,
        "format": "datasets",
        "recommended": False,
    },
]


@router.get("/suggested")
async def get_suggested_datasets():
    """Return a list of datasets known to work well for matting/segmentation."""
    return SUGGESTED_DATASETS


# ── Validate dataset before download ─────────────────────
@router.post("/validate")
async def validate_dataset(req: ValidateRequest):
    """Check that a HuggingFace dataset has actual downloadable data."""
    import asyncio
    from ml.dataset import HFMattingDataset

    def _validate():
        return HFMattingDataset.validate_hf_repo(req.dataset_name, settings.hf_token)

    return await asyncio.to_thread(_validate)


# ── Search HuggingFace Hub ───────────────────────────────
@router.get("/search")
async def search_datasets(q: str = Query(..., description="Search query")):
    """Search HuggingFace Hub for matting/segmentation datasets."""
    from huggingface_hub import HfApi

    api = HfApi(token=settings.hf_token)
    results = list(
        api.list_datasets(search=q, limit=20, sort="downloads", direction=-1)
    )

    return [
        {
            "id": ds.id,
            "author": ds.author,
            "downloads": ds.downloads,
            "likes": ds.likes,
            "tags": ds.tags[:10] if ds.tags else [],
            "last_modified": str(ds.last_modified) if ds.last_modified else None,
        }
        for ds in results
    ]


# ── Download dataset ─────────────────────────────────────
@router.post("/download")
async def download_dataset(req: DownloadRequest):
    """
    Download a HuggingFace dataset and organize into images/ + alphas/.
    Returns immediately with a task id; progress can be tracked.
    """
    import asyncio
    from ml.dataset import HFMattingDataset

    output_dir = settings.dataset_path / req.dataset_name.replace("/", "__")

    if output_dir.exists():
        return {
            "status": "already_exists",
            "path": str(output_dir),
            "message": "Dataset already downloaded. Delete it first to re-download.",
        }

    # Run download in thread to avoid blocking
    def _download():
        return HFMattingDataset.prepare_from_hf(
            dataset_name=req.dataset_name,
            output_dir=str(output_dir),
            split=req.split,
            max_samples=req.max_samples,
            hf_token=settings.hf_token,
        )

    try:
        result_path = await asyncio.to_thread(_download)
        # Count files
        images = list((result_path / "images").rglob("*.*"))
        alphas = list((result_path / "alphas").rglob("*.*"))

        return {
            "status": "completed",
            "path": str(result_path),
            "images_count": len(images),
            "alphas_count": len(alphas),
        }
    except ValueError as e:
        # Known validation errors — return 422 with descriptive message
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Clean up partial downloads on unexpected errors
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error descargando «{req.dataset_name}»: {e}",
        )


# ── List local datasets ─────────────────────────────────
@router.get("/local")
async def list_local_datasets():
    """List datasets downloaded locally."""
    base = settings.dataset_path
    if not base.exists():
        return []

    datasets = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue

        images_dir = d / "images"
        alphas_dir = d / "alphas"

        img_count = len(list(images_dir.rglob("*.*"))) if images_dir.exists() else 0
        alpha_count = len(list(alphas_dir.rglob("*.*"))) if alphas_dir.exists() else 0

        # Check for curated metadata
        meta_path = d / "metadata.json"
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())

        datasets.append(
            {
                "id": d.name,
                "name": d.name.replace("__", "/"),
                "path": str(d),
                "images_count": img_count,
                "alphas_count": alpha_count,
                "curated": metadata.get("curated", False),
                "metadata": metadata,
            }
        )

    return datasets


# ── Preview samples ──────────────────────────────────────
@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, n: int = Query(8, ge=1, le=50)):
    """Get preview thumbnails (base64) of dataset samples."""
    import base64
    from io import BytesIO
    from PIL import Image

    ds_path = settings.dataset_path / dataset_id
    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    images_dir = ds_path / "images"
    alphas_dir = ds_path / "alphas"

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_images = sorted([f for f in images_dir.rglob("*") if f.suffix.lower() in exts])[
        :n
    ]

    samples = []
    for img_path in all_images:
        # Create thumbnail
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((256, 256))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Try to find corresponding alpha
        alpha_b64 = None
        for ext in [".png", ".jpg", ".jpeg"]:
            alpha_path = alphas_dir / img_path.parent.name / (img_path.stem + ext)
            if not alpha_path.exists():
                # Try flat structure
                alpha_path = alphas_dir / (img_path.stem + ext)
            if alpha_path.exists():
                a = Image.open(alpha_path)
                if a.mode == "RGBA":
                    a = a.split()[-1]
                elif a.mode != "L":
                    a = a.convert("L")
                a.thumbnail((256, 256))
                buf2 = BytesIO()
                a.save(buf2, format="PNG")
                alpha_b64 = base64.b64encode(buf2.getvalue()).decode()
                break

        samples.append(
            {
                "filename": img_path.name,
                "image": f"data:image/jpeg;base64,{img_b64}",
                "alpha": f"data:image/png;base64,{alpha_b64}" if alpha_b64 else None,
                "width": img.width,
                "height": img.height,
            }
        )

    return samples


# ── Curate / validate dataset ────────────────────────────
@router.post("/{dataset_id}/curate")
async def curate_dataset(dataset_id: str, req: CurateRequest):
    """Validate and curate dataset: check alpha range, resolution, broken files."""
    import asyncio
    from PIL import Image
    import numpy as np

    ds_path = settings.dataset_path / dataset_id
    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    def _curate():
        images_dir = ds_path / "images"
        alphas_dir = ds_path / "alphas"
        exts = {".jpg", ".jpeg", ".png", ".webp"}

        all_images = sorted(
            [f for f in images_dir.rglob("*") if f.suffix.lower() in exts]
        )

        stats = {
            "total": len(all_images),
            "valid": 0,
            "too_small": 0,
            "broken_alpha": 0,
            "no_alpha": 0,
            "all_white_alpha": 0,
            "all_black_alpha": 0,
            "issues": [],
        }

        for img_path in all_images:
            try:
                img = Image.open(img_path)
                w, h = img.size

                if min(w, h) < req.min_resolution:
                    stats["too_small"] += 1
                    stats["issues"].append(
                        {
                            "file": img_path.name,
                            "issue": "too_small",
                            "detail": f"{w}x{h}",
                        }
                    )
                    continue

                # Find alpha
                alpha_found = False
                for ext in [".png", ".jpg", ".jpeg"]:
                    alpha_path = (
                        alphas_dir / img_path.parent.name / (img_path.stem + ext)
                    )
                    if not alpha_path.exists():
                        alpha_path = alphas_dir / (img_path.stem + ext)
                    if alpha_path.exists():
                        alpha_found = True
                        a = Image.open(alpha_path)
                        if a.mode == "RGBA":
                            a = a.split()[-1]
                        elif a.mode != "L":
                            a = a.convert("L")

                        arr = np.array(a)

                        if req.check_alpha_range:
                            if arr.max() == 0:
                                stats["all_black_alpha"] += 1
                                stats["issues"].append(
                                    {"file": img_path.name, "issue": "all_black_alpha"}
                                )
                                continue
                            if arr.min() == 255:
                                stats["all_white_alpha"] += 1
                                stats["issues"].append(
                                    {"file": img_path.name, "issue": "all_white_alpha"}
                                )
                                continue
                        break

                if not alpha_found:
                    stats["no_alpha"] += 1
                    stats["issues"].append({"file": img_path.name, "issue": "no_alpha"})
                    continue

                stats["valid"] += 1

            except Exception as e:
                stats["broken_alpha"] += 1
                stats["issues"].append(
                    {"file": img_path.name, "issue": "broken", "detail": str(e)}
                )

        # Limit issues list to 100 for response size
        stats["issues"] = stats["issues"][:100]
        return stats

    result = await asyncio.to_thread(_curate)

    # Save metadata
    meta = {
        "curated": True,
        "valid_count": result["valid"],
        "total_count": result["total"],
        "min_resolution": req.min_resolution,
    }
    (ds_path / "metadata.json").write_text(json.dumps(meta, indent=2))

    return result


# ── Dataset statistics ───────────────────────────────────
@router.get("/{dataset_id}/stats")
async def dataset_stats(dataset_id: str):
    """Get statistics about a dataset."""
    import asyncio
    from PIL import Image
    import numpy as np

    ds_path = settings.dataset_path / dataset_id
    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    def _stats():
        images_dir = ds_path / "images"
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        all_images = sorted(
            [f for f in images_dir.rglob("*") if f.suffix.lower() in exts]
        )

        widths, heights, sizes = [], [], []
        for p in all_images[:500]:  # sample first 500
            try:
                img = Image.open(p)
                w, h = img.size
                widths.append(w)
                heights.append(h)
                sizes.append(p.stat().st_size)
            except Exception:
                pass

        total_size = sum(f.stat().st_size for f in images_dir.rglob("*") if f.is_file())

        return {
            "total_images": len(all_images),
            "avg_width": int(np.mean(widths)) if widths else 0,
            "avg_height": int(np.mean(heights)) if heights else 0,
            "min_width": int(np.min(widths)) if widths else 0,
            "min_height": int(np.min(heights)) if heights else 0,
            "max_width": int(np.max(widths)) if widths else 0,
            "max_height": int(np.max(heights)) if heights else 0,
            "avg_file_size_kb": int(np.mean(sizes) / 1024) if sizes else 0,
            "total_size_mb": round(total_size / (1024 * 1024), 1),
        }

    return await asyncio.to_thread(_stats)


# ── Delete dataset ───────────────────────────────────────
@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a locally downloaded dataset."""
    ds_path = settings.dataset_path / dataset_id
    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    shutil.rmtree(ds_path)
    return {"status": "deleted", "id": dataset_id}
