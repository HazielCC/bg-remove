"""
Server-side inference API.

Endpoints:
  POST /run    — run inference on a single image
  POST /batch  — run inference on multiple images (for evaluation)
"""

import asyncio
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from PIL import Image

from config import settings
from ml.modnet import MODNet

router = APIRouter()


@router.post("/run")
async def run_inference(
    image: UploadFile = File(...),
    checkpoint: str = Form(None),
    img_size: int = Form(512),
):
    """
    Run MODNet inference on a single image.

    Args:
        image: uploaded image file
        checkpoint: path to checkpoint (or None for default)
        img_size: resize to this size (square)
    """

    async def _infer():
        device = settings.get_torch_device()

        # Load model
        if checkpoint and Path(checkpoint).exists():
            model = MODNet.from_pretrained(
                checkpoint, device=str(device), backbone_pretrained=False
            )
        else:
            # Try to find a best.ckpt in any run
            best = None
            for ckpt in settings.checkpoint_path.rglob("best.ckpt"):
                best = ckpt
                break
            if best:
                model = MODNet.from_pretrained(
                    str(best), device=str(device), backbone_pretrained=False
                )
            else:
                # Use freshly initialized model (for testing)
                model = MODNet(backbone_pretrained=True).to(device)

        model.eval()

        # Load image
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        original_size = img.size

        # Preprocess
        img_resized = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img_resized).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # normalize to [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            pred_matte = model(tensor, inference=True)

        # Post-process
        matte = pred_matte.squeeze().cpu().numpy()
        matte = np.clip(matte, 0, 1)

        # Resize back to original
        matte_img = Image.fromarray((matte * 255).astype(np.uint8))
        matte_img = matte_img.resize(original_size, Image.BILINEAR)

        # Create result with transparency
        img_rgba = img.copy().convert("RGBA")
        img_rgba.putalpha(matte_img)

        # Encode results
        # Matte as PNG
        buf_matte = BytesIO()
        matte_img.save(buf_matte, format="PNG")
        matte_b64 = base64.b64encode(buf_matte.getvalue()).decode()

        # Result with transparency
        buf_result = BytesIO()
        img_rgba.save(buf_result, format="PNG")
        result_b64 = base64.b64encode(buf_result.getvalue()).decode()

        return {
            "matte": f"data:image/png;base64,{matte_b64}",
            "result": f"data:image/png;base64,{result_b64}",
            "width": original_size[0],
            "height": original_size[1],
        }

    return await _infer()


@router.post("/batch")
async def batch_inference(
    checkpoint: str = Form(None),
    dataset_id: str = Form(...),
    max_images: int = Form(20),
    img_size: int = Form(512),
):
    """
    Run inference on a batch of images from a dataset.
    Returns metrics if alpha ground truth is available.
    """
    from ml.metrics import evaluate_matting

    ds_path = settings.dataset_path / dataset_id
    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    async def _batch():
        device = settings.get_torch_device()

        # Load model
        if checkpoint and Path(checkpoint).exists():
            model = MODNet.from_pretrained(
                checkpoint, device=str(device), backbone_pretrained=False
            )
        else:
            model = MODNet(backbone_pretrained=True).to(device)
        model.eval()

        images_dir = ds_path / "images"
        alphas_dir = ds_path / "alphas"
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        all_images = sorted(
            [f for f in images_dir.rglob("*") if f.suffix.lower() in exts]
        )[:max_images]

        results = []
        all_metrics = {"sad": [], "mse": [], "gradient_error": []}

        with torch.no_grad():
            for img_path in all_images:
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((img_size, img_size), Image.BILINEAR)
                arr = np.array(img_resized).astype(np.float32) / 255.0
                arr = (arr - 0.5) / 0.5
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

                pred_matte = model(tensor, inference=True)
                matte = pred_matte.squeeze().cpu().numpy()
                matte = np.clip(matte, 0, 1)

                entry = {"filename": img_path.name}

                # Check for GT alpha
                for ext in [".png", ".jpg"]:
                    alpha_path = (
                        alphas_dir / img_path.parent.name / (img_path.stem + ext)
                    )
                    if not alpha_path.exists():
                        alpha_path = alphas_dir / (img_path.stem + ext)
                    if alpha_path.exists():
                        gt = Image.open(alpha_path)
                        if gt.mode == "RGBA":
                            gt = gt.split()[-1]
                        elif gt.mode != "L":
                            gt = gt.convert("L")
                        gt = gt.resize((img_size, img_size), Image.BILINEAR)
                        gt_arr = np.array(gt).astype(np.float32) / 255.0

                        metrics = evaluate_matting(matte, gt_arr)
                        entry["metrics"] = metrics
                        for k in all_metrics:
                            if k in metrics:
                                all_metrics[k].append(metrics[k])
                        break

                results.append(entry)

        # Compute averages
        avg_metrics = {}
        for k, vals in all_metrics.items():
            if vals:
                avg_metrics[k] = float(np.mean(vals))

        return {
            "n_images": len(results),
            "avg_metrics": avg_metrics,
            "results": results,
        }

    return await asyncio.to_thread(_batch)
