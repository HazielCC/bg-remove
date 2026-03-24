"""
Router for Qwen-Image-Layered decomposition.
Provides endpoints to decompose an image into multiple RGBA layers.
"""

import asyncio
import base64
import threading
import uuid as _uuid_module
from io import BytesIO

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps

from config import settings
from ml.hf_downloader import HFModelDownloader

router = APIRouter()

# Global variables
_pipeline = None
_load_lock = threading.Lock()
_inference_lock = None  # Initialized lazily

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_MAX_JOBS = 20  # Keep at most 20 jobs in memory
_background_tasks: set = set()  # Prevent premature GC of fire-and-forget tasks

MODEL_ID = "Qwen/Qwen-Image-Layered"
_download_dir = settings.model_path / "qwen-image-layered"
_download_dir.mkdir(parents=True, exist_ok=True)

downloader = HFModelDownloader(model_id=MODEL_ID, local_dir=str(_download_dir))

def get_inference_lock():
    global _inference_lock
    if _inference_lock is None:
        _inference_lock = asyncio.Lock()
    return _inference_lock

@router.get("/status")
async def get_status():
    """Get the current download/loading status."""
    st = downloader.get_status()
    return {
        "progress": st["progress"],
        "is_downloading": st["is_downloading"],
        "message": st["message"],
        "total_bytes": st["total_bytes"],
        "downloaded_bytes": st["downloaded_bytes"],
        "speed_mbps": st.get("speed_mbps", 0.0)
    }

def get_pipeline():
    global _pipeline
    with _load_lock:
        if _pipeline is None:
            from diffusers import DiffusionPipeline

            device = settings.get_torch_device()
            
            try:
                # Nos aseguramos de descargarlo de manera síncrona si no está
                if not downloader.check_exists():
                    downloader.download_sync()

                downloader.set_message(f"Cargando modelo en {device}...")
                
                dtype = (
                    torch.float16 if device.type in ("mps", "cuda") else torch.float32
                )

                _pipeline = DiffusionPipeline.from_pretrained(
                    str(_download_dir), torch_dtype=dtype, trust_remote_code=True
                )
                _pipeline.to(device)

                if device.type in ("mps", "cuda"):
                    _pipeline.enable_attention_slicing()

                downloader.set_message("Listo")
                print("[layered] Model loaded successfully")
            except Exception as e:
                downloader.set_message(f"Error: {str(e)}")
                print(f"[layered] Error loading model: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to load Qwen model: {str(e)}"
                )

    return _pipeline

async def _run_decompose(
    job_id: str,
    image_bytes: bytes,
    prompt: str,
    layer_num: int,
    num_inference_steps: int,
    seed: int | None,
) -> None:
    """Background task: load model if needed, run inference, store result in _jobs."""
    _jobs[job_id]["status"] = "waiting"
    async with get_inference_lock():
        _jobs[job_id]["status"] = "running"

        def _process() -> dict:
            pipeline = get_pipeline()
            device = settings.get_torch_device()

            try:
                img = Image.open(BytesIO(image_bytes))
                img = ImageOps.exif_transpose(img).convert("RGBA")
            except Exception as e:
                raise ValueError(f"Invalid image: {e}")

            MAX_DIM = 1024
            width, height = img.size
            curr_w, curr_h = width, height
            if max(curr_w, curr_h) > MAX_DIM:
                scale = MAX_DIM / max(curr_w, curr_h)
                curr_w = max(1, int(curr_w * scale))
                curr_h = max(1, int(curr_h * scale))

            new_width = max(16, round(curr_w / 16) * 16)
            new_height = max(16, round(curr_h / 16) * 16)

            if new_width != width or new_height != height:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)

            print(
                f"[layered] Job {job_id}: starting decomposition into {layer_num} layers..."
            )
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
            for layer_img in output.images:
                buf = BytesIO()
                layer_img.save(buf, format="PNG")
                layers_b64.append(
                    f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

            return {
                "layers": layers_b64,
                "count": len(layers_b64),
                "width": new_width,
                "height": new_height,
                "has_reference": True,
            }

        try:
            result = await asyncio.to_thread(_process)
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = result
            print(f"[layered] Job {job_id}: done ({result['count']} layers)")
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
            print(f"[layered] Job {job_id} failed: {e}")


@router.get("/job/{job_id}")
async def get_job(job_id: str):
    """Get the status (pending/running/done/error) and result of a decomposition job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.post("/decompose")
async def decompose_image(
    image: UploadFile = File(...),
    prompt: str = Form("A detailed image decomposed into layers"),
    layer_num: int = Form(4, ge=2, le=12),
    num_inference_steps: int = Form(50, ge=1, le=100),
    seed: int | None = Form(None),
):
    """
    Submit a decomposition job. Returns {job_id} immediately.
    Poll GET /job/{job_id} for status and result.
    """
    MAX_FILE_SIZE = 15 * 1024 * 1024
    if image.size and image.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum size is 15MB."
        )

    image_bytes = await image.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum size is 15MB."
        )

    # Clear stale progress from previous failed/interrupted attempts.
    if _pipeline is None and not downloader.get_status()["is_downloading"]:
        downloader.reset_status("Esperando descarga del modelo...")

    job_id = str(_uuid_module.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
        }
        # Evict oldest jobs beyond the cap
        if len(_jobs) > _MAX_JOBS:
            oldest = next(iter(_jobs))
            del _jobs[oldest]

    task = asyncio.create_task(
        _run_decompose(
            job_id, image_bytes, prompt, layer_num, num_inference_steps, seed
        )
    )
    # Prevent garbage collection before the task completes
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id}
