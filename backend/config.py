"""Centralized configuration for the fine-tuning backend."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Ensure MPS fallback is enabled before torch import
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class Settings(BaseSettings):
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def model_path(self) -> Path:
        p = Path(self.model_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def dataset_path(self) -> Path:
        p = Path(self.dataset_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def checkpoint_path(self) -> Path:
        p = Path(self.checkpoint_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def export_path(self) -> Path:
        p = Path(self.export_dir)
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
