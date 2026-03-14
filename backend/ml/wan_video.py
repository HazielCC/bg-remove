import os
import uuid
import torch
from pathlib import Path

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from ml.hf_downloader import HFModelDownloader
from config import settings

# Directorio de exportación
OUTPUT_DIR = settings.export_path / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"

# Instanciar el helper unificado de descargas
downloader = HFModelDownloader(model_id=MODEL_ID)

# Variable global para mantener el modelo cargado en memoria (singleton)
pipe = None

def get_status() -> dict:
    """Devuelve el estado usando el downloader unificado."""
    status = downloader.get_status()
    return {
        "is_downloaded": status["is_downloaded"],
        "is_downloading": status["is_downloading"],
        "downloaded_gb": status["downloaded_bytes"] / (1024**3) if status["downloaded_bytes"] else 0.0,
        "total_gb": status["total_bytes"] / (1024**3) if status["total_bytes"] else 0.0,
        "progress": status["progress"],
        "detail": status["message"]
    }

def start_download():
    """Inicia la descarga delegando al downloader en background."""
    downloader.start_download_bg()

def get_model():
    global pipe

    if pipe is None:
        if not downloader.check_exists():
            raise RuntimeError(f"El modelo {MODEL_ID} no está descargado todavía.")

        print(f"[Wan2.2] Cargando modelo {MODEL_ID} en VRAM...")
        downloader.set_message("Cargando modelo en VRAM...")
        
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
        print("[Wan2.2] Modelo cargado con éxito en CUDA.")
        
        downloader.set_message("Modelo listo en memoria.")

    return pipe

def generate(prompt: str, duration: int = 4) -> str:
    print(f"[Wan2.2] Iniciando generación de video: '{prompt}' (Duración: {duration}s)")
    downloader.set_message(f"Generando video de {duration}s...")
    
    pipe = get_model()

    output = pipe(
        prompt=prompt,
        num_frames=duration * 16
    )

    filename = f"{uuid.uuid4()}.mp4"
    path = OUTPUT_DIR / filename

    if hasattr(output, "frames"):
        export_to_video(output.frames[0], str(path), fps=16)
    else:
        output.save(str(path))

    print(f"[Wan2.2] Video guardado en: {path}")
    downloader.set_message("Modelo listo en memoria.")

    return f"/videos/{filename}"
