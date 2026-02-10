"""FastAPI application for MODNet fine-tuning backend."""

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
from routers import datasets, training, models, inference  # noqa: E402

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "device": settings.device}
