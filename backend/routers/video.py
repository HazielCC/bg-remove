from fastapi import APIRouter, HTTPException
import asyncio

from ml import wan_video
from models.schemas_video import VideoGenerateRequest, VideoGenerateResponse, ModelStatusResponse

router = APIRouter()

# Lock global para evitar quedarse sin VRAM (Out of Memory)
generation_lock = asyncio.Lock()

@router.get("/model-status", response_model=ModelStatusResponse)
async def check_model_status():
    """Retorna el estado de descarga del modelo Wan2.2."""
    status = wan_video.get_status()
    return ModelStatusResponse(**status)

@router.post("/download-model")
async def trigger_download():
    """Inicia la descarga del modelo en segundo plano."""
    wan_video.start_download()
    return {"message": "Descarga iniciada"}

@router.post("/generate", response_model=VideoGenerateResponse)
async def generate_video(req: VideoGenerateRequest):
    """Genera un video usando el modelo Wan2.2-TI2V-5B de forma síncrona/bloqueante."""
    print(f"[API] Nueva solicitud de video. Prompt: '{req.prompt}', duration: {req.duration}")

    # Bloqueamos el lock para que solo se procese un video a la vez en este worker
    async with generation_lock:
        try:
            # Dado que el modelo es muy pesado y bloqueante, lo ejecutamos en un threadpool
            # (si quisiéramos) o directamente. Aquí lo ejecutamos en un thread auxiliar
            # para no bloquear otras consultas rápidas a la API de FastAPI.
            video_url = await asyncio.to_thread(
                wan_video.generate,
                prompt=req.prompt,
                duration=req.duration
            )

            return VideoGenerateResponse(video_url=video_url)

        except Exception as e:
            print(f"[Error] Fallo en la generación del video: {e}")
            raise HTTPException(status_code=500, detail=str(e))
