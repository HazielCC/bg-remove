"""
Model management API: export, quantize, deploy.

Endpoints:
  GET  /list             — list available models (base + fine-tuned)
  POST /{id}/export-onnx — export checkpoint to ONNX
  POST /{id}/quantize    — quantize ONNX (fp16 / uint8)
  POST /{id}/deploy      — copy ONNX to public/models/ for frontend use
  POST /compare          — run inference with 2 models for comparison
  POST /benchmark        — benchmark checkpoints on a labeled local dataset
"""

import asyncio
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings

router = APIRouter()


# ── Schemas ──────────────────────────────────────────────
class ExportRequest(BaseModel):
    img_size: int = 512
    path: str | None = None


class QuantizeRequest(BaseModel):
    dtype: str = "uint8"  # "fp16" or "uint8"
    path: str | None = None


class DeployRequest(BaseModel):
    target_dir: str = "../public/models/modnet/onnx"
    path: str | None = None


class CompareRequest(BaseModel):
    model_a: str  # checkpoint path or ONNX path
    model_b: str


class BenchmarkRequest(BaseModel):
    dataset_id: str
    checkpoints: list[str] | None = None
    max_images: int = 40
    img_size: int = 512


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

    # Prefer explicit path from request (avoids ambiguous stem matches)
    ckpt_path = _resolve_checkpoint_path(req.path) if req.path else _find_checkpoint(model_id)
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

    # Prefer explicit path from request (avoids ambiguous stem matches)
    onnx_path = _resolve_onnx_path(req.path) if req.path else _find_onnx(model_id)
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
    # Prefer explicit path from request (avoids ambiguous stem matches)
    onnx_path = _resolve_onnx_path(req.path) if req.path else _find_onnx(model_id)
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

    def _compare():
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


@router.post("/benchmark")
async def benchmark_models(req: BenchmarkRequest):
    """
    Benchmark checkpoints on a local dataset with alpha labels.

    Returns per-model metrics averages and a ranking.
    """
    import numpy as np
    import torch
    from PIL import Image
    from ml.metrics import evaluate_matting
    from ml.modnet import MODNet

    if req.max_images < 1:
        raise HTTPException(status_code=400, detail="max_images must be >= 1")
    if req.img_size < 32:
        raise HTTPException(status_code=400, detail="img_size must be >= 32")

    ds_path = _resolve_dataset_path(req.dataset_id)
    if not ds_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    def _collect_checkpoints() -> list[Path]:
        if req.checkpoints:
            resolved: list[Path] = []
            seen: set[str] = set()
            for ckpt_str in req.checkpoints:
                ckpt = _resolve_checkpoint_path(ckpt_str)
                if ckpt is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid checkpoint path: {ckpt_str}",
                    )
                key = str(ckpt.resolve())
                if key not in seen:
                    seen.add(key)
                    resolved.append(ckpt)
            return resolved

        discovered: list[Path] = []
        discovered.extend(sorted(settings.checkpoint_path.rglob("*.ckpt")))
        discovered.extend(sorted(settings.model_path.rglob("*.ckpt")))

        uniq: dict[str, Path] = {}
        for ckpt in discovered:
            uniq[str(ckpt.resolve())] = ckpt
        return list(uniq.values())

    def _find_alpha_for_image(
        img_path: Path, images_dir: Path, alphas_dir: Path
    ) -> Path | None:
        rel = img_path.relative_to(images_dir)
        exts = [".png", ".jpg", ".jpeg"]
        candidates: list[Path] = []

        # Mirror the image directory structure under alphas/.
        for ext in exts:
            candidates.append(alphas_dir / rel.with_suffix(ext))

        # Fallbacks for flat or legacy layouts.
        for ext in exts:
            candidates.append(alphas_dir / (img_path.stem + ext))
            candidates.append(alphas_dir / rel.parent.name / (img_path.stem + ext))

        for p in candidates:
            if p.exists():
                return p
        return None

    def _prepare_pairs(images_dir: Path, alphas_dir: Path) -> list[tuple[Path, Path]]:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        all_images = sorted(
            [f for f in images_dir.rglob("*") if f.suffix.lower() in exts]
        )

        pairs: list[tuple[Path, Path]] = []
        for img_path in all_images:
            alpha_path = _find_alpha_for_image(img_path, images_dir, alphas_dir)
            if alpha_path is not None:
                pairs.append((img_path, alpha_path))
            if len(pairs) >= req.max_images:
                break
        return pairs

    def _benchmark():
        images_dir = ds_path / "images"
        alphas_dir = ds_path / "alphas"
        if not images_dir.exists() or not alphas_dir.exists():
            raise HTTPException(
                status_code=422,
                detail="Dataset must contain images/ and alphas/ directories",
            )

        pairs = _prepare_pairs(images_dir, alphas_dir)
        if not pairs:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No labeled pairs found in dataset. "
                    "Expected matching files under images/ and alphas/."
                ),
            )

        checkpoints = _collect_checkpoints()
        if not checkpoints:
            raise HTTPException(status_code=404, detail="No checkpoints found to benchmark")

        device = settings.get_torch_device()
        model_rows: list[dict] = []

        for ckpt_path in checkpoints:
            try:
                model = MODNet.from_pretrained(
                    str(ckpt_path),
                    device=str(device),
                    backbone_pretrained=False,
                )
                model.eval()
            except Exception as exc:
                model_rows.append(
                    {
                        "checkpoint": str(ckpt_path),
                        "run": ckpt_path.parent.name if ckpt_path.parent else None,
                        "name": ckpt_path.name,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                continue

            metric_lists = {
                "sad": [],
                "mse": [],
                "gradient_error": [],
                "connectivity_error": [],
            }
            infer_times_ms: list[float] = []

            with torch.no_grad():
                for img_path, alpha_path in pairs:
                    img = Image.open(img_path).convert("RGB")
                    alpha = Image.open(alpha_path)

                    if alpha.mode == "RGBA":
                        alpha = alpha.split()[-1]
                    elif alpha.mode != "L":
                        alpha = alpha.convert("L")

                    img_resized = img.resize((req.img_size, req.img_size), Image.BILINEAR)
                    alpha_resized = alpha.resize(
                        (req.img_size, req.img_size), Image.BILINEAR
                    )

                    arr = np.array(img_resized).astype(np.float32) / 255.0
                    arr = (arr - 0.5) / 0.5
                    tensor = (
                        torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                    )

                    t0 = time.perf_counter()
                    pred = model(tensor, inference=True)
                    infer_times_ms.append((time.perf_counter() - t0) * 1000.0)

                    matte = pred.squeeze().cpu().numpy()
                    matte = np.clip(matte, 0.0, 1.0)
                    gt = np.array(alpha_resized).astype(np.float32) / 255.0

                    m = evaluate_matting(matte, gt)
                    for key in metric_lists:
                        metric_lists[key].append(float(m[key]))

            avg_metrics = {
                key: float(np.mean(vals)) for key, vals in metric_lists.items() if vals
            }
            model_rows.append(
                {
                    "checkpoint": str(ckpt_path),
                    "run": ckpt_path.parent.name if ckpt_path.parent else None,
                    "name": ckpt_path.name,
                    "status": "ok",
                    "n_images": len(pairs),
                    "avg_metrics": avg_metrics,
                    "avg_inference_ms": float(np.mean(infer_times_ms))
                    if infer_times_ms
                    else None,
                }
            )

        valid_rows = [r for r in model_rows if r.get("status") == "ok"]
        if not valid_rows:
            return {
                "dataset_id": req.dataset_id,
                "n_images": len(pairs),
                "ranked_models": [],
                "failed_models": model_rows,
                "score_formula": "score = mean(metric / best_metric_for_that_metric)",
            }

        metric_keys = ["sad", "mse", "gradient_error", "connectivity_error"]
        metric_best: dict[str, float] = {}
        for key in metric_keys:
            vals = [
                float(r["avg_metrics"][key])
                for r in valid_rows
                if key in r.get("avg_metrics", {})
            ]
            metric_best[key] = max(min(vals), 1e-12)

        for row in valid_rows:
            ratios = [
                float(row["avg_metrics"][k]) / metric_best[k]
                for k in metric_keys
                if k in row.get("avg_metrics", {})
            ]
            row["overall_score"] = float(np.mean(ratios)) if ratios else float("inf")

        ranked = sorted(valid_rows, key=lambda r: float(r.get("overall_score", 1e9)))
        for i, row in enumerate(ranked, start=1):
            row["rank"] = i

        failed = [r for r in model_rows if r.get("status") != "ok"]
        return {
            "dataset_id": req.dataset_id,
            "n_images": len(pairs),
            "ranked_models": ranked,
            "failed_models": failed,
            "metric_baselines": metric_best,
            "score_formula": "score = mean(metric / best_metric_for_that_metric)",
        }

    return await asyncio.to_thread(_benchmark)


# ── Helpers ──────────────────────────────────────────────
def _resolve_dataset_path(dataset_id: str) -> Path:
    if not dataset_id or not dataset_id.strip():
        raise HTTPException(status_code=400, detail="Invalid dataset id")

    base = settings.dataset_path.resolve()
    dataset_path = (base / dataset_id).resolve()
    if dataset_path == base:
        raise HTTPException(status_code=400, detail="Invalid dataset id")

    try:
        dataset_path.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid dataset id") from exc

    return dataset_path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_checkpoint_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists() or p.suffix != ".ckpt":
        return None
    resolved = p.resolve()
    allowed_roots = [settings.checkpoint_path.resolve(), settings.model_path.resolve()]
    if any(_is_under(resolved, root) for root in allowed_roots):
        return resolved
    return None


def _resolve_onnx_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists() or p.suffix != ".onnx":
        return None
    resolved = p.resolve()
    if _is_under(resolved, settings.export_path.resolve()):
        return resolved
    return None


def _find_checkpoint(model_id: str) -> Path | None:
    """Find a checkpoint by name or path."""
    # Direct path
    direct = _resolve_checkpoint_path(model_id)
    if direct:
        return direct

    # Search in checkpoints dir
    matches: list[Path] = []
    for ckpt in settings.checkpoint_path.rglob("*.ckpt"):
        if ckpt.stem == model_id or ckpt.name == model_id:
            matches.append(ckpt)

    # Search in models dir
    for ckpt in settings.model_path.rglob("*.ckpt"):
        if ckpt.stem == model_id or ckpt.name == model_id:
            matches.append(ckpt)

    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Ambiguous checkpoint id: {model_id}. "
                "Provide the exact checkpoint path in request body."
            ),
        )
    if len(matches) == 1:
        return matches[0]

    return None


def _find_onnx(model_id: str) -> Path | None:
    """Find an ONNX model by name or path."""
    direct = _resolve_onnx_path(model_id)
    if direct:
        return direct

    matches: list[Path] = []
    for onnx_file in settings.export_path.rglob("*.onnx"):
        if onnx_file.stem == model_id or onnx_file.name == model_id:
            matches.append(onnx_file)

    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Ambiguous ONNX model id: {model_id}. "
                "Provide the exact ONNX path in request body."
            ),
        )
    if len(matches) == 1:
        return matches[0]

    return None
