"""
Model management API: export, quantize, deploy.

Endpoints:
  GET  /list             — list available models (base + fine-tuned)
  POST /{id}/export-onnx — export checkpoint to ONNX
  POST /{id}/quantize    — quantize ONNX (fp16 / uint8)
  POST /{id}/deploy      — copy ONNX to public/models/ for frontend use
  POST /compare          — run inference with 2 models for comparison
"""

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from config import settings

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────
class ExportRequest(BaseModel):
    img_size: int = 512


class QuantizeRequest(BaseModel):
    dtype: str = "uint8"  # "fp16" or "uint8"


class DeployRequest(BaseModel):
    target_dir: str = "../public/models/modnet/onnx"


class CompareRequest(BaseModel):
    model_a: str  # checkpoint path or ONNX path
    model_b: str


# ── List Models ──────────────────────────────────────────
@router.get("/list")
async def list_models():
    """List all available models: checkpoints + exported ONNX."""
    models = []

    # Checkpoints
    ckpt_dir = settings.checkpoint_path
    if ckpt_dir.exists():
        for run_dir in sorted(ckpt_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for ckpt in sorted(run_dir.glob("*.ckpt")):
                models.append(
                    {
                        "type": "checkpoint",
                        "run": run_dir.name,
                        "name": ckpt.name,
                        "path": str(ckpt),
                        "size_mb": round(ckpt.stat().st_size / (1024 * 1024), 2),
                    }
                )

    # Exported ONNX
    export_dir = settings.export_path
    if export_dir.exists():
        for onnx_file in sorted(export_dir.rglob("*.onnx")):
            models.append(
                {
                    "type": "onnx",
                    "name": onnx_file.name,
                    "path": str(onnx_file),
                    "size_mb": round(onnx_file.stat().st_size / (1024 * 1024), 2),
                }
            )

    # Pre-trained (if present)
    pretrained_dir = settings.model_path
    if pretrained_dir.exists():
        for ckpt in pretrained_dir.glob("*.ckpt"):
            models.append(
                {
                    "type": "pretrained",
                    "name": ckpt.name,
                    "path": str(ckpt),
                    "size_mb": round(ckpt.stat().st_size / (1024 * 1024), 2),
                }
            )

    return models


# ── Export to ONNX ───────────────────────────────────────
@router.post("/{model_id}/export-onnx")
async def export_onnx(model_id: str, req: ExportRequest):
    """Export a checkpoint to ONNX format."""
    from ml.export import export_to_onnx

    # Find checkpoint
    ckpt_path = _find_checkpoint(model_id)
    if not ckpt_path:
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {model_id}")

    output_name = Path(ckpt_path).stem + ".onnx"
    output_path = settings.export_path / output_name

    try:
        result = await asyncio.to_thread(
            export_to_onnx,
            ckpt_path=str(ckpt_path),
            output_path=str(output_path),
            img_size=req.img_size,
            device="cpu",  # ONNX export works best on CPU
        )
        return {
            "status": "exported",
            "path": result,
            "size_mb": round(output_path.stat().st_size / (1024 * 1024), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Quantize ─────────────────────────────────────────────
@router.post("/{model_id}/quantize")
async def quantize_model(model_id: str, req: QuantizeRequest):
    """Quantize an ONNX model to fp16 or uint8."""
    from ml.export import quantize_fp16, quantize_uint8

    # Find ONNX file
    onnx_path = _find_onnx(model_id)
    if not onnx_path:
        raise HTTPException(status_code=404, detail=f"ONNX model not found: {model_id}")

    stem = Path(onnx_path).stem
    if req.dtype == "fp16":
        output = settings.export_path / f"{stem}_fp16.onnx"
        result = await asyncio.to_thread(quantize_fp16, str(onnx_path), str(output))
    elif req.dtype == "uint8":
        output = settings.export_path / f"{stem}_uint8.onnx"
        result = await asyncio.to_thread(quantize_uint8, str(onnx_path), str(output))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported dtype: {req.dtype}")

    return {
        "status": "quantized",
        "dtype": req.dtype,
        "path": result,
        "size_mb": round(output.stat().st_size / (1024 * 1024), 2),
    }


# ── Deploy to frontend ──────────────────────────────────
@router.post("/{model_id}/deploy")
async def deploy_model(model_id: str, req: DeployRequest):
    """Copy an ONNX model to the frontend public directory."""
    onnx_path = _find_onnx(model_id)
    if not onnx_path:
        raise HTTPException(status_code=404, detail=f"ONNX model not found: {model_id}")

    target = Path(req.target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Determine target filename based on dtype
    src_name = Path(onnx_path).name
    if "fp16" in src_name:
        dest_name = "model_fp16.onnx"
    elif "uint8" in src_name or "quantized" in src_name:
        dest_name = "model_quantized.onnx"
    else:
        dest_name = "model.onnx"

    dest = target / dest_name
    shutil.copy2(onnx_path, dest)

    return {
        "status": "deployed",
        "source": str(onnx_path),
        "destination": str(dest),
        "filename": dest_name,
    }


# ── Compare Models ───────────────────────────────────────
@router.post("/compare")
async def compare_models(req: CompareRequest):
    """
    Compare two models by running inference on the same test images.
    Returns paths to result images for side-by-side comparison.
    """
    import torch
    import numpy as np
    from PIL import Image
    from ml.modnet import MODNet
    import base64
    from io import BytesIO

    device = settings.get_torch_device()

    async def _compare():
        # Load both models
        model_a = MODNet.from_pretrained(
            req.model_a, device=str(device), backbone_pretrained=False
        )
        model_b = MODNet.from_pretrained(
            req.model_b, device=str(device), backbone_pretrained=False
        )
        model_a.eval()
        model_b.eval()

        # Find test images (from any dataset)
        test_images = []
        for ds_dir in settings.dataset_path.iterdir():
            if not ds_dir.is_dir():
                continue
            img_dir = ds_dir / "images"
            if img_dir.exists():
                exts = {".jpg", ".jpeg", ".png"}
                imgs = sorted(
                    [f for f in img_dir.rglob("*") if f.suffix.lower() in exts]
                )
                test_images.extend(imgs[:5])
            if len(test_images) >= 10:
                break

        if not test_images:
            return {"results": [], "message": "No test images found"}

        results = []
        with torch.no_grad():
            for img_path in test_images[:8]:
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((512, 512))
                arr = np.array(img_resized).astype(np.float32) / 255.0
                arr = (arr - 0.5) / 0.5
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

                # Model A
                mask_a = model_a(tensor, inference=True)
                mask_a_np = mask_a.squeeze().cpu().numpy()
                mask_a_img = Image.fromarray((mask_a_np * 255).astype(np.uint8))
                buf_a = BytesIO()
                mask_a_img.save(buf_a, format="PNG")
                mask_a_b64 = base64.b64encode(buf_a.getvalue()).decode()

                # Model B
                mask_b = model_b(tensor, inference=True)
                mask_b_np = mask_b.squeeze().cpu().numpy()
                mask_b_img = Image.fromarray((mask_b_np * 255).astype(np.uint8))
                buf_b = BytesIO()
                mask_b_img.save(buf_b, format="PNG")
                mask_b_b64 = base64.b64encode(buf_b.getvalue()).decode()

                # Original thumbnail
                img.thumbnail((256, 256))
                buf_orig = BytesIO()
                img.save(buf_orig, format="JPEG", quality=80)
                orig_b64 = base64.b64encode(buf_orig.getvalue()).decode()

                results.append(
                    {
                        "filename": img_path.name,
                        "original": f"data:image/jpeg;base64,{orig_b64}",
                        "mask_a": f"data:image/png;base64,{mask_a_b64}",
                        "mask_b": f"data:image/png;base64,{mask_b_b64}",
                    }
                )

        return {"results": results}

    return await asyncio.to_thread(_compare)


# ── Helpers ──────────────────────────────────────────────
def _find_checkpoint(model_id: str) -> Path | None:
    """Find a checkpoint by name or path."""
    # Direct path
    p = Path(model_id)
    if p.exists() and p.suffix == ".ckpt":
        return p

    # Search in checkpoints dir
    for ckpt in settings.checkpoint_path.rglob("*.ckpt"):
        if ckpt.stem == model_id or ckpt.name == model_id:
            return ckpt

    # Search in models dir
    for ckpt in settings.model_path.rglob("*.ckpt"):
        if ckpt.stem == model_id or ckpt.name == model_id:
            return ckpt

    return None


def _find_onnx(model_id: str) -> Path | None:
    """Find an ONNX model by name or path."""
    p = Path(model_id)
    if p.exists() and p.suffix == ".onnx":
        return p

    for onnx_file in settings.export_path.rglob("*.onnx"):
        if onnx_file.stem == model_id or onnx_file.name == model_id:
            return onnx_file

    return None
