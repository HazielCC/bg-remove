import os
import uuid
import torch
import threading
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

# Directorio de exportación
OUTPUT_DIR = "exports/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"

# Variable global para mantener el modelo cargado en memoria (singleton)
pipe = None

# Variables para controlar la descarga en segundo plano
_download_thread = None
_download_error = None
_download_complete = False

def get_model_dir():
    """Retorna la ruta del caché de HuggingFace para este modelo."""
    return os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}")

def get_downloaded_gb() -> float:
    """Calcula el tamaño descargado en Gigabytes del directorio del modelo."""
    path = get_model_dir()
    if not os.path.exists(path):
        return 0.0
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)

def check_model_exists() -> bool:
    """Verifica si el modelo ya está completamente descargado en la caché local."""
    global _download_complete
    if _download_complete:
        return True
    try:
        # local_files_only=True arroja error si falta algún archivo
        snapshot_download(repo_id=MODEL_ID, local_files_only=True)
        _download_complete = True
        return True
    except Exception:
        return False

def _download_task():
    global _download_error, _download_complete
    try:
        print(f"[Wan2.2] Iniciando descarga del modelo {MODEL_ID}...")
        snapshot_download(repo_id=MODEL_ID)
        _download_complete = True
        print(f"[Wan2.2] Descarga de {MODEL_ID} completada exitosamente.")
    except Exception as e:
        _download_error = str(e)
        print(f"[Wan2.2] Error descargando modelo: {e}")

def start_download():
    """Inicia la descarga del modelo en un hilo separado (no bloqueante)."""
    global _download_thread, _download_error, _download_complete
    if _download_thread is not None and _download_thread.is_alive():
        return # Ya está descargando
    if check_model_exists():
        return # Ya está descargado

    _download_error = None
    _download_complete = False
    _download_thread = threading.Thread(target=_download_task)
    _download_thread.start()

def get_status() -> dict:
    is_downloading = _download_thread is not None and _download_thread.is_alive()
    is_downloaded = check_model_exists() and not is_downloading
    
    detail = ""
    if _download_error:
        detail = f"Error: {_download_error}"
    elif is_downloading:
        detail = "Descargando modelo desde Hugging Face..."
    elif is_downloaded:
        detail = "Modelo listo y cacheado."
    else:
        detail = "Modelo no descargado."

    return {
        "is_downloaded": is_downloaded,
        "is_downloading": is_downloading,
        "downloaded_gb": get_downloaded_gb(),
        "detail": detail
    }

def get_model():
    global pipe

    if pipe is None:
        if not check_model_exists():
            raise RuntimeError(f"El modelo {MODEL_ID} no está descargado todavía.")

        print(f"[Wan2.2] Cargando modelo {MODEL_ID}...")
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        )
        # Movemos a CUDA para aprovechar GPU
        pipe = pipe.to("cuda")
        print("[Wan2.2] Modelo cargado con éxito en CUDA.")

    return pipe

def generate(prompt: str, duration: int = 4) -> str:
    print(f"[Wan2.2] Iniciando generación de video: '{prompt}' (Duración: {duration}s)")
    pipe = get_model()

    # Inferencia (dependiendo de la versión de diffusers y del modelo exacto)
    # Por defecto, 16 fps, num_frames define la longitud
    output = pipe(
        prompt=prompt,
        num_frames=duration * 16
    )

    filename = f"{uuid.uuid4()}.mp4"
    path = os.path.join(OUTPUT_DIR, filename)

    # Si el resultado trae 'frames', exportamos a video
    if hasattr(output, "frames"):
        # `output.frames` suele ser una lista de frames (imágenes PIL o arrays)
        export_to_video(output.frames[0], path, fps=16)
    else:
        # En caso de que el pipeline exponga un método save directo (depende del PR específico)
        output.save(path)

    print(f"[Wan2.2] Video guardado en: {path}")

    return f"/videos/{filename}"
