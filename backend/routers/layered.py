"""
Router for Qwen-Image-Layered decomposition.
Provides endpoints to decompose an image into multiple RGBA layers.
"""

import asyncio
import base64
import threading
from io import BytesIO
from typing import Optional

import torch
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image, ImageOps

from config import settings

router = APIRouter()

# Global variables
_pipeline = None
_load_lock = threading.Lock()
_inference_lock = None # Initialized lazily

# Note: _download_status is a global variable. In a production environment with multiple
# Gunicorn/Uvicorn workers, this status will not be shared across processes, leading to
# inconsistent UI feedback. For scaling, consider using Redis or a shared file lock.
_download_status = {
    "progress": 0,  # 0 to 100
    "is_downloading": False,
    "message": "",
    "total_bytes": 0,
    "downloaded_bytes": 0
}

def get_inference_lock():
    global _inference_lock
    if _inference_lock is None:
        _inference_lock = asyncio.Lock()
    return _inference_lock

class ProgressTracker:
    """Class-based tqdm proxy for snapshot_download."""
    _total_acc = 0
    _current_acc = 0
    _lock = threading.Lock()

    def __init__(self, iterable=None, total=None, desc=None, **kwargs):
        self.iterable = iterable
        # Accumulate total size for all files being downloaded
        if total:
            global _download_status
            with ProgressTracker._lock:
                ProgressTracker._total_acc += total
                _download_status["total_bytes"] = ProgressTracker._total_acc

    def __iter__(self):
        if self.iterable:
            for item in self.iterable:
                yield item
                self.update(1)

    def update(self, n=1):
        global _download_status
        with ProgressTracker._lock:
            ProgressTracker._current_acc += n
            if ProgressTracker._total_acc > 0:
                _download_status["downloaded_bytes"] = ProgressTracker._current_acc
                # Cap at 99 until fully loaded
                _download_status["progress"] = min(99, int((ProgressTracker._current_acc / ProgressTracker._total_acc) * 100))
                _download_status["is_downloading"] = True

    def close(self):
        pass

    def refresh(self):
        pass

    def set_description(self, desc, refresh=True):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

@router.get("/status")
async def get_status():
    """Get the current download/loading status."""
    return _download_status

def get_pipeline():
    global _pipeline, _download_status
    with _load_lock:
        if _pipeline is None:
            from diffusers import DiffusionPipeline
            from huggingface_hub import snapshot_download
            
            repo_id = "Qwen/Qwen-Image-Layered"
            device = settings.get_torch_device()
            
            _download_status["is_downloading"] = True
            _download_status["message"] = "Descargando modelo de Hugging Face..."
            _download_status["progress"] = 0
            
            try:
                # Reset accumulators before download
                with ProgressTracker._lock:
                    ProgressTracker._total_acc = 0
                    ProgressTracker._current_acc = 0

                snapshot_download(
                    repo_id=repo_id,
                    tqdm_class=ProgressTracker
                )
                
                # Update total bytes in status once known
                with ProgressTracker._lock:
                    _download_status["total_bytes"] = ProgressTracker._total_acc

                _download_status["message"] = f"Cargando modelo en {device}..."
                _download_status["progress"] = 100
                
                dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
                
                # trust_remote_code=True is required for Qwen's custom pipeline architecture.
                # Safe here as "Qwen" is a verified/trusted organization on Hugging Face.
                _pipeline = DiffusionPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                _pipeline.to(device)
                
                if device.type in ("mps", "cuda"):
                    _pipeline.enable_attention_slicing()
                    
                _download_status["is_downloading"] = False
                _download_status["message"] = "Listo"
                print("[layered] Model loaded successfully")
            except Exception as e:
                _download_status["is_downloading"] = False
                _download_status["message"] = f"Error: {str(e)}"
                print(f"[layered] Error loading model: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load Qwen model: {str(e)}")
            
    return _pipeline


@router.post("/decompose")
async def decompose_image(
    image: UploadFile = File(...),
    prompt: str = Form("A detailed image decomposed into layers"),
    layer_num: int = Form(4, ge=2, le=12),
    num_inference_steps: int = Form(50, ge=1, le=100),
    seed: Optional[int] = Form(None),
):
    """
    Decompose an image into multiple RGBA layers using Qwen-Image-Layered.
    """
    
    # Validation: Limit file size to 15MB to prevent OOM
    MAX_FILE_SIZE = 15 * 1024 * 1024 # 15MB
    if image.size and image.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 15MB.")

    # Read bytes early to avoid temporary file closure during long model loading
    image_bytes = await image.read()
    
    # Secondary check for cases where image.size was None
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 15MB.")

    async with get_inference_lock():
        def _process():
            pipeline = get_pipeline()
            device = settings.get_torch_device()
            
            # Load and prepare image
            try:
                img = Image.open(BytesIO(image_bytes))
                img = ImageOps.exif_transpose(img).convert("RGBA")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
                
            # Limits to avoid OOM
            MAX_DIM = 1024 
            width, height = img.size
            
            curr_width, curr_height = width, height
            if max(curr_width, curr_height) > MAX_DIM:
                scale = MAX_DIM / max(curr_width, curr_height)
                curr_width = max(1, int(curr_width * scale))
                curr_height = max(1, int(curr_height * scale))

            # Round to multiple of 16
            new_width = max(16, round(curr_width / 16) * 16)
            new_height = max(16, round(curr_height / 16) * 16)
            
            # Single resize operation to preserve quality
            if new_width != width or new_height != height:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Set seed
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)

            # Run inference
            print(f"[layered] Starting decomposition into {layer_num} layers...")
            with torch.no_grad():
                output = pipeline(
                    prompt=prompt,
                    image=img,
                    layer_num=layer_num,
                    height=new_height,
                    width=new_width,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                )
            
            layers_b64 = []
            for i, layer_img in enumerate(output.images):
                buf = BytesIO()
                layer_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                layers_b64.append(f"data:image/png;base64,{b64}")
                
            # Free VRAM after inference
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

            return {
                "layers": layers_b64,
                "count": len(layers_b64),
                "width": new_width,
                "height": new_height,
                "has_reference": True
            }

        return await asyncio.to_thread(_process)
