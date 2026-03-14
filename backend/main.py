"""FastAPI application for MODNet fine-tuning backend."""

import torch

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Ensure directories exist
    settings.model_path
    settings.dataset_path
    settings.checkpoint_path
    settings.export_path
    print(f"[startup] device={settings.device}  dirs ready")
    yield
    print("[shutdown] bye")


app = FastAPI(
    title="MODNet Fine-Tuning API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",
        "http://localhost:3002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── routers ──────────────────────────────────────────────
import os
from routers import datasets, training, models, inference, layered, video  # noqa: E402
from fastapi.staticfiles import StaticFiles

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(layered.router, prefix="/api/layered", tags=["layered"])
app.include_router(video.router, prefix="/api/video", tags=["video"])

# Servir videos generados
os.makedirs("exports/videos", exist_ok=True)
app.mount("/videos", StaticFiles(directory="exports/videos"), name="videos")

@app.get("/api/health")
async def health():
    return {"status": "ok", "device": settings.device}
