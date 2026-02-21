"""
Training API.

Endpoints:
  POST /start    — launch fine-tuning
  GET  /stream   — SSE stream of training metrics
  POST /stop     — stop current training
  GET  /status   — current training status
  GET  /checkpoints — list saved checkpoints
  GET  /history  — training run history
"""

import asyncio
import json
from typing import Literal
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from config import settings
from ml.trainer import (
    TrainingConfig,
    get_training_state,
    run_supervised_training,
    run_soc_adaptation,
    stop_training,
)

router = APIRouter()


def _resolve_dataset_path(dataset_id: str) -> Path:
    """Resolve dataset path and ensure it stays under settings.dataset_path."""
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


# ── Schemas ──────────────────────────────────────────────
class TrainRequest(BaseModel):
    dataset_id: str
    stage: Literal["supervised", "soc"] = "supervised"
    epochs: int = Field(40, ge=1)
    lr: float = Field(0.01, gt=0)
    batch_size: int = Field(4, ge=1)
    img_size: int = Field(512, ge=32)
    pretrained_ckpt: str | None = None
    run_name: str = Field(
        "run_001",
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9._-]+$",
    )

    # Loss weights (supervised)
    semantic_loss_weight: float = Field(10.0, ge=0)
    detail_loss_weight: float = Field(10.0, ge=0)
    matte_loss_weight: float = Field(1.0, ge=0)

    # SOC params
    soc_lr: float = Field(0.00001, gt=0)
    soc_epochs: int = Field(10, ge=1)

    # Data
    train_split: float = Field(0.8, gt=0, lt=1)
    val_split: float = Field(0.1, gt=0, lt=1)
    backgrounds_dir: str | None = None

    # Checkpointing
    save_every: int = Field(5, ge=1)

    @model_validator(mode="after")
    def validate_split_sum(self):
        if self.train_split + self.val_split > 1.0:
            raise ValueError("train_split + val_split must be <= 1.0")
        return self


class RecommendConfigRequest(BaseModel):
    dataset_id: str
    stage: Literal["supervised", "soc"] = "supervised"
    has_pretrained: bool = False


def _analyze_dataset(dataset_path: Path) -> dict:
    from PIL import Image

    images_dir = dataset_path / "images"
    alphas_dir = dataset_path / "alphas"
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    images = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in exts])
    alphas = sorted([p for p in alphas_dir.rglob("*") if p.suffix.lower() in exts])

    widths: list[int] = []
    heights: list[int] = []
    for p in images[:300]:
        try:
            img = Image.open(p)
            w, h = img.size
            widths.append(w)
            heights.append(h)
        except Exception:
            continue

    avg_w = int(sum(widths) / len(widths)) if widths else 0
    avg_h = int(sum(heights) / len(heights)) if heights else 0
    min_side = int(min(min(widths), min(heights))) if widths and heights else 0

    gemini_summary = None
    summary_path = dataset_path / "gemini_assessment_summary.json"
    if summary_path.exists():
        try:
            gemini_summary = json.loads(summary_path.read_text())
        except Exception:
            gemini_summary = None

    return {
        "images_count": len(images),
        "alphas_count": len(alphas),
        "avg_width": avg_w,
        "avg_height": avg_h,
        "min_side": min_side,
        "gemini_summary": gemini_summary,
    }


def _recommend_supervised_config(stats: dict) -> tuple[dict, list[str]]:
    n = int(stats["images_count"])
    min_side = int(stats["min_side"])
    summary = stats.get("gemini_summary") or {}

    reasons: list[str] = []
    cfg = {
        "stage": "supervised",
        "epochs": 40,
        "lr": 0.01,
        "batch_size": 4,
        "img_size": 512,
        "semantic_loss_weight": 10.0,
        "detail_loss_weight": 10.0,
        "matte_loss_weight": 1.0,
        "train_split": 0.8,
        "val_split": 0.1,
        "save_every": 5,
    }

    if n < 300:
        cfg.update({"epochs": 45, "lr": 0.004, "batch_size": 2, "train_split": 0.75, "val_split": 0.2})
        reasons.append("Dataset pequeño: más epochs, LR más bajo y validación más grande.")
    elif n < 2000:
        cfg.update({"epochs": 35, "lr": 0.005, "batch_size": 4, "train_split": 0.8, "val_split": 0.15})
        reasons.append("Dataset mediano: balance entre estabilidad y tiempo de entrenamiento.")
    else:
        cfg.update({"epochs": 24, "lr": 0.008, "batch_size": 6, "train_split": 0.85, "val_split": 0.1})
        reasons.append("Dataset grande: menos epochs y mayor throughput.")

    if min_side >= 1200:
        cfg["img_size"] = 768
        reasons.append("Resolución alta detectada: se sube img_size a 768.")
    elif min_side and min_side < 480:
        cfg["img_size"] = 384
        reasons.append("Resolución baja detectada: img_size 384 para evitar artefactos.")

    avg_quality = float(summary.get("avg_quality_score", 0.0) or 0.0)
    split_counts = summary.get("split_counts", {}) if isinstance(summary, dict) else {}
    difficulty_counts = summary.get("difficulty_counts", {}) if isinstance(summary, dict) else {}

    if avg_quality > 0:
        reasons.append(f"Gemini avg_quality_score={avg_quality:.1f} aplicado al ajuste fino.")
        if avg_quality < 55:
            cfg["lr"] = round(float(cfg["lr"]) * 0.7, 6)
            cfg["detail_loss_weight"] = 12.0
            cfg["matte_loss_weight"] = 1.2
            reasons.append("Calidad baja: LR reducido y mayor peso a detalle/matte.")
        elif avg_quality >= 75:
            cfg["lr"] = round(float(cfg["lr"]) * 1.1, 6)
            reasons.append("Calidad alta: LR ligeramente mayor para convergencia más rápida.")

    assessed = int(summary.get("assessed_images", 0) or 0)
    if assessed > 0 and isinstance(split_counts, dict):
        exclude_ratio = float(split_counts.get("exclude", 0) or 0) / max(1, assessed)
        if exclude_ratio > 0.2:
            cfg["train_split"] = min(float(cfg["train_split"]), 0.75)
            cfg["val_split"] = max(float(cfg["val_split"]), 0.2)
            reasons.append("Muchos samples excluibles: se incrementa validación para controlar overfitting.")

    if assessed > 0 and isinstance(difficulty_counts, dict):
        hard_ratio = float(difficulty_counts.get("hard", 0) or 0) / max(1, assessed)
        if hard_ratio > 0.35:
            cfg["detail_loss_weight"] = max(float(cfg["detail_loss_weight"]), 12.0)
            reasons.append("Alta proporción de casos difíciles: mayor peso en pérdida de detalle.")

    return cfg, reasons


def _recommend_soc_config(has_pretrained: bool, stats: dict) -> tuple[dict, list[str]]:
    n = int(stats["images_count"])
    reasons: list[str] = []
    cfg = {
        "stage": "soc",
        "soc_epochs": 8,
        "soc_lr": 0.00001,
        "batch_size": 1,
        "img_size": 512,
        "save_every": 2,
    }

    if n >= 2000:
        cfg["soc_epochs"] = 6
        reasons.append("Dataset grande: menos epochs SOC para evitar deriva.")
    elif n < 500:
        cfg["soc_epochs"] = 10
        reasons.append("Dataset pequeño: más epochs SOC para adaptación progresiva.")

    if not has_pretrained:
        reasons.append("SOC sin checkpoint previo no es ideal; se recomienda pasar por supervisado primero.")

    return cfg, reasons


# Global SSE queue — created per training run
_event_queue: asyncio.Queue | None = None
_training_task: asyncio.Task | None = None


@router.post("/recommend-config")
async def recommend_training_config(req: RecommendConfigRequest):
    """Recommend training hyperparameters dynamically from dataset characteristics."""
    ds_path = _resolve_dataset_path(req.dataset_id)
    if not ds_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {req.dataset_id}")

    stats = await asyncio.to_thread(_analyze_dataset, ds_path)
    if stats["images_count"] < 1:
        raise HTTPException(status_code=422, detail="Dataset has no images")

    if req.stage == "soc":
        recommendation, reasons = _recommend_soc_config(req.has_pretrained, stats)
    else:
        recommendation, reasons = _recommend_supervised_config(stats)

    return {
        "dataset_id": req.dataset_id,
        "stage": req.stage,
        "recommendation": recommendation,
        "reasons": reasons,
        "dataset_stats": {
            "images_count": stats["images_count"],
            "alphas_count": stats["alphas_count"],
            "avg_width": stats["avg_width"],
            "avg_height": stats["avg_height"],
            "min_side": stats["min_side"],
            "has_gemini_summary": bool(stats.get("gemini_summary")),
        },
    }


# ── Start Training ───────────────────────────────────────
@router.post("/start")
async def start_training(req: TrainRequest):
    """Launch a fine-tuning run."""
    global _event_queue, _training_task

    state = get_training_state()
    if state.status == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Resolve dataset path
    ds_path = _resolve_dataset_path(req.dataset_id)
    if not ds_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: {req.dataset_id}"
        )

    config = TrainingConfig(
        dataset_dir=str(ds_path),
        checkpoint_dir=str(settings.checkpoint_path),
        pretrained_ckpt=req.pretrained_ckpt,
        device=settings.device,
        stage=req.stage,
        epochs=req.epochs if req.stage == "supervised" else req.soc_epochs,
        lr=req.lr,
        batch_size=req.batch_size,
        img_size=req.img_size,
        num_workers=0,
        semantic_loss_weight=req.semantic_loss_weight,
        detail_loss_weight=req.detail_loss_weight,
        matte_loss_weight=req.matte_loss_weight,
        soc_lr=req.soc_lr,
        soc_epochs=req.soc_epochs,
        save_every=req.save_every,
        train_split=req.train_split,
        val_split_ratio=req.val_split,
        backgrounds_dir=req.backgrounds_dir,
        run_name=req.run_name,
    )

    _event_queue = asyncio.Queue(maxsize=200)

    if req.stage == "supervised":
        train_fn = run_supervised_training
    elif req.stage == "soc":
        train_fn = run_soc_adaptation
    else:
        raise HTTPException(status_code=400, detail=f"Unknown stage: {req.stage}")

    async def _run():
        try:
            await asyncio.to_thread(train_fn, config, _event_queue)
        except Exception as e:
            print(f"[training error] {e}")

    _training_task = asyncio.create_task(_run())

    return {
        "status": "started",
        "run_name": req.run_name,
        "stage": req.stage,
        "epochs": config.epochs,
        "device": settings.device,
    }


# ── SSE Stream ───────────────────────────────────────────
@router.get("/stream")
async def stream_metrics():
    """Server-Sent Events stream of training metrics."""

    async def event_generator():
        # Send initial status
        state = get_training_state()
        yield f"data: {json.dumps({'type': 'status', 'status': state.status})}\n\n"

        while True:
            if _event_queue is None:
                await asyncio.sleep(1)
                continue

            try:
                event = await asyncio.wait_for(_event_queue.get(), timeout=2.0)
                yield f"data: {json.dumps(event)}\n\n"

                # Stop streaming on terminal events
                if event.get("type") in ("finished", "error", "stopped"):
                    break
            except asyncio.TimeoutError:
                # Send heartbeat
                state = get_training_state()
                yield f"data: {json.dumps({'type': 'heartbeat', 'status': state.status, 'epoch': state.current_epoch})}\n\n"

                if state.status in ("finished", "error", "stopped", "idle"):
                    break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Stop Training ────────────────────────────────────────
@router.post("/stop")
async def stop_current_training():
    """Stop the current training run."""
    state = get_training_state()
    if state.status != "running":
        raise HTTPException(status_code=400, detail="No training in progress")

    stop_training()
    return {"status": "stopping"}


# ── Training Status ──────────────────────────────────────
@router.get("/status")
async def training_status():
    """Get current training status."""
    import math

    state = get_training_state()
    best_val = state.best_val_loss
    return {
        "status": state.status,
        "epoch": state.current_epoch,
        "total_epochs": state.total_epochs,
        "total_loss": state.current_loss,
        "semantic_loss": state.semantic_loss,
        "detail_loss": state.detail_loss,
        "matte_loss": state.matte_loss,
        "val_loss": state.val_loss,
        "lr": state.lr,
        "samples_processed": state.samples_processed,
        "elapsed_seconds": state.elapsed_seconds,
        "eta_seconds": state.eta_seconds,
        "best_val_loss": None if math.isinf(best_val) else best_val,
        "error_message": state.error_message,
    }


# ── List Checkpoints ────────────────────────────────────
@router.get("/checkpoints")
async def list_checkpoints():
    """List all saved checkpoints (sorted by newest)."""
    ckpt_dir = settings.checkpoint_path
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for run_dir in ckpt_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for ckpt in run_dir.glob("*.ckpt"):
            stat = ckpt.stat()
            checkpoints.append(
                {
                    "run": run_dir.name,
                    "name": ckpt.name,
                    "path": str(ckpt),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                }
            )

    checkpoints.sort(key=lambda x: x["modified"], reverse=True)
    return checkpoints


# ── Training History ─────────────────────────────────────
@router.get("/history")
async def training_history():
    """List past training runs (based on checkpoint directories)."""
    ckpt_dir = settings.checkpoint_path
    if not ckpt_dir.exists():
        return []

    runs = []
    for run_dir in sorted(ckpt_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ckpts = sorted(run_dir.glob("*.ckpt"))
        best = run_dir / "best.ckpt"
        runs.append(
            {
                "run_name": run_dir.name,
                "n_checkpoints": len(ckpts),
                "has_best": best.exists(),
                "best_size_mb": (
                    round(best.stat().st_size / (1024 * 1024), 2)
                    if best.exists()
                    else None
                ),
            }
        )

    return runs
