import uuid

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from config import settings
from ml.hf_downloader import HFModelDownloader

# Directorio de exportación
OUTPUT_DIR = settings.export_path / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Usamos el repo oficial de THUDM para CogVideoX-2b
MODEL_ID = "THUDM/CogVideoX-2b"
_download_dir = settings.model_path / "cogvideox-2b"
_download_dir.mkdir(parents=True, exist_ok=True)

# Instanciar el helper unificado de descargas
downloader = HFModelDownloader(model_id=MODEL_ID, local_dir=str(_download_dir))

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
        "speed_mbps": status.get("speed_mbps", 0.0),
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

        print(f"[CogVideoX] Cargando modelo {MODEL_ID} en memoria...")
        downloader.set_message("Cargando modelo en VRAM...")
        
        # Usamos CogVideoXPipeline para Text-to-Video por ahora
        pipe = CogVideoXPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        )
        
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        
        # Optimizaciones opcionales para MPS/CUDA
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
            
        print(f"[CogVideoX] Modelo cargado con éxito en {device}.")
        downloader.set_message("Modelo listo en memoria.")

    return pipe

def generate(prompt: str, duration: int = 4) -> str:
    print(f"[CogVideoX] Iniciando generación de video: '{prompt}'")
    downloader.set_message("Generando video...")
    
    pipe = get_model()

    # CogVideoX-2b típicamente usa frames=49 y 8 fps.
    # Mapeamos la duración a los frames (aprox 8-12 fps)
    fps = 12
    frames_to_generate = min(32, duration * fps) 
    
    result = pipe(
        prompt=prompt,
        num_frames=frames_to_generate,
        guidance_scale=6.0,
        num_inference_steps=50,
    )

    filename = f"{uuid.uuid4()}.mp4"
    path = OUTPUT_DIR / filename

    # Exportar usando diffusers export_to_video (que usa imageio por debajo)
    export_to_video(result.frames[0], str(path), fps=fps)

    print(f"[CogVideoX] Video guardado en: {path}")
    downloader.set_message("Modelo listo en memoria.")

    return f"/videos/{filename}"