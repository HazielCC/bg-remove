"""Centralized configuration for the fine-tuning backend."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure MPS fallback is enabled before torch import
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Device
    device: str = "mps"

    # Directories (relative to backend/)
    model_dir: str = "./models"
    dataset_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    export_dir: str = "./exports"

    # Training defaults
    default_epochs: int = 40
    default_lr: float = 0.01
    default_batch_size: int = 4

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    frontend_url: str = "http://localhost:3002"

    # HuggingFace
    hf_token: str | None = None

    # Gemini (dataset assessment)
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3-flash-preview"
    gemini_timeout_seconds: int = 45
    gemini_default_max_images: int = 200

    def _resolve_backend_path(self, value: str) -> Path:
        """Resolve relative backend paths independently from the launch cwd."""
        path = Path(value)
        if not path.is_absolute():
            path = (BACKEND_ROOT / path).resolve()
        return path

    @property
    def model_path(self) -> Path:
        p = self._resolve_backend_path(self.model_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def dataset_path(self) -> Path:
        p = self._resolve_backend_path(self.dataset_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def checkpoint_path(self) -> Path:
        p = self._resolve_backend_path(self.checkpoint_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def export_path(self) -> Path:
        p = self._resolve_backend_path(self.export_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def public_models_path(self) -> Path:
        p = (PROJECT_ROOT / "public" / "models").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def default_public_model_path(self) -> Path:
        p = self.public_models_path / "modnet"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_torch_device(self):
        import torch

        if self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


settings = Settings()
