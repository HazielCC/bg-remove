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
from pydantic import BaseModel, Field

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


# Global SSE queue — created per training run
_event_queue: asyncio.Queue | None = None
_training_task: asyncio.Task | None = None


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
    """List all saved checkpoints."""
    ckpt_dir = settings.checkpoint_path
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for run_dir in sorted(ckpt_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for ckpt in sorted(run_dir.glob("*.ckpt")):
            checkpoints.append(
                {
                    "run": run_dir.name,
                    "name": ckpt.name,
                    "path": str(ckpt),
                    "size_mb": round(ckpt.stat().st_size / (1024 * 1024), 2),
                    "modified": ckpt.stat().st_mtime,
                }
            )

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
