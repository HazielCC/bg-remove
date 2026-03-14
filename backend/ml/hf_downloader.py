import os
import time
import threading
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from tqdm.auto import tqdm


def _safe_dir_size_bytes(root: Path) -> int:
    """Calcula recursivamente el tamaño en bytes del directorio local.
    Omite los enlaces simbólicos y duros duplicados (común en Windows/HF cache)."""
    try:
        total = 0
        seen_inodes = set()
        for p in root.rglob("*"):
            if p.is_file() and not p.is_symlink():
                try:
                    stat = p.stat()
                    # Identificador único del archivo físico en el disco
                    file_id = (stat.st_dev, stat.st_ino)
                    if file_id in seen_inodes:
                        continue
                    seen_inodes.add(file_id)
                    total += stat.st_size
                except OSError:
                    continue
        return total
    except OSError:
        return 0


class HFModelDownloader:
    """Helper unificado para gestionar y monitorear la descarga de modelos pesados de Hugging Face."""

    def __init__(self, model_id: str, local_dir: Optional[str] = None):
        self.model_id = model_id
        self.local_dir = local_dir
        
        # Si no se especifica directorio local, usa la caché estándar de HuggingFace
        if self.local_dir is None:
            self.cache_dir = Path(HF_HUB_CACHE) / f"models--{self.model_id.replace('/', '--')}"
        else:
            self.cache_dir = Path(self.local_dir)
            
        self.status = {
            "is_downloaded": False,
            "is_downloading": False,
            "progress": 0,
            "total_bytes": 0,
            "downloaded_bytes": 0,
            "speed_mbps": 0.0,
            "message": "Modelo no inicializado."
        }
        self._last_downloaded_bytes = 0
        self._last_speed_time = time.time()
        self._status_lock = threading.RLock()
        self._download_thread = None

    def check_exists(self) -> bool:
        """Verifica rápida y silenciosamente si los archivos base ya están cacheados."""
        with self._status_lock:
            if self.status["is_downloaded"]:
                return True
        try:
            kwargs = {"repo_id": self.model_id, "local_files_only": True}
            if self.local_dir:
                kwargs["local_dir"] = self.local_dir
                kwargs["local_dir_use_symlinks"] = False
                
            snapshot_download(**kwargs)
            with self._status_lock:
                self.status["is_downloaded"] = True
                self.status["progress"] = 100
                self.status["speed_mbps"] = 0.0
                self.status["message"] = "Modelo listo y cacheado."
            return True
        except Exception:
            return False

    def get_status(self) -> dict:
        """Retorna el estado de descarga actual."""
        # Intenta un check rápido si el estado indica que no hay nada activo
        if not self.status["is_downloaded"] and not self.status["is_downloading"]:
            self.check_exists()
            
        with self._status_lock:
            return self.status.copy()

    def set_message(self, message: str):
        """Permite inyectar mensajes de estado personalizados (ej: 'Cargando en VRAM...')."""
        with self._status_lock:
            self.status["message"] = message

    def reset_status(self, message: str = ""):
        """Reinicia la barra de progreso (útil si la carga falla y hay que reintentar)."""
        with self._status_lock:
            self.status["progress"] = 0
            self.status["is_downloading"] = False
            self.status["message"] = message
            self.status["total_bytes"] = 0
            self.status["downloaded_bytes"] = 0
            self.status["speed_mbps"] = 0.0
            self._last_downloaded_bytes = 0
            self._last_speed_time = time.time()

    def _run_download(self):
        """Lógica central de descarga y rastreo del progreso."""
        with self._status_lock:
            self.status["is_downloading"] = True
            self.status["message"] = "Calculando tamaño esperado..."
            self.status["progress"] = 0
            self.status["total_bytes"] = 0
            self.status["downloaded_bytes"] = 0
            self.status["speed_mbps"] = 0.0
            self._last_downloaded_bytes = _safe_dir_size_bytes(self.cache_dir)
            self._last_speed_time = time.time()

        try:
            print(f"[{self.model_id}] Obteniendo metadatos...")
            info = HfApi().model_info(self.model_id, files_metadata=True)
            total_bytes = sum((getattr(s, "size", 0) or 0) for s in (info.siblings or []))
            with self._status_lock:
                if total_bytes > 0:
                    self.status["total_bytes"] = int(total_bytes)

            downloader = self

            # Custom TQDM tracker to bridge huggingface_hub downloads to our API
            class ProgressTracker(tqdm):
                _last_scan_ts = 0.0
                
                def __init__(self, *args, **kwargs):
                    kwargs.pop("name", None)
                    kwargs.pop("lock_name", None)
                    kwargs.setdefault("disable", True)
                    super().__init__(*args, **kwargs)

                def update(self, n=1):
                    displayed = super().update(n)
                    now = time.time()
                    dt = now - self._last_scan_ts
                    
                    if dt >= 0.5: # Escaneo ligero cada medio segundo
                        self._last_scan_ts = now
                        downloaded = _safe_dir_size_bytes(downloader.cache_dir)
                        
                        with downloader._status_lock:
                            if downloader.status["total_bytes"] > 0:
                                # Calcular velocidad en MB/s cada 1 segundo aprox
                                speed_dt = now - downloader._last_speed_time
                                if speed_dt >= 1.0:
                                    diff = downloaded - downloader._last_downloaded_bytes
                                    if diff > 0:
                                        speed_bps = diff / speed_dt
                                        downloader.status["speed_mbps"] = speed_bps / (1024 * 1024)
                                    downloader._last_downloaded_bytes = downloaded
                                    downloader._last_speed_time = now

                                downloader.status["downloaded_bytes"] = min(downloaded, downloader.status["total_bytes"])
                                downloader.status["progress"] = min(99, int((downloader.status["downloaded_bytes"] / downloader.status["total_bytes"]) * 100))
                                downloader.status["is_downloading"] = True
                                
                                if downloader.status["speed_mbps"] > 0:
                                    downloader.status["message"] = f"Descargando... {downloader.status['speed_mbps']:.1f} MB/s"
                                else:
                                    downloader.status["message"] = "Descargando modelo desde Hugging Face..."
                    return displayed

            print(f"[{self.model_id}] Iniciando descarga de {total_bytes / (1024**3):.2f} GB...")
            
            download_kwargs = {
                "repo_id": self.model_id,
                "tqdm_class": ProgressTracker
            }
            if self.local_dir:
                download_kwargs["local_dir"] = self.local_dir
                download_kwargs["local_dir_use_symlinks"] = False

            snapshot_download(**download_kwargs)

            with self._status_lock:
                if self.status["total_bytes"] > 0:
                    self.status["downloaded_bytes"] = self.status["total_bytes"]
                self.status["is_downloading"] = False
                self.status["is_downloaded"] = True
                self.status["progress"] = 100
                self.status["speed_mbps"] = 0.0
                self.status["message"] = "Modelo listo y cacheado."
                
            print(f"[{self.model_id}] Descarga completada exitosamente.")

        except Exception as e:
            with self._status_lock:
                self.status["is_downloading"] = False
                self.status["speed_mbps"] = 0.0
                self.status["message"] = f"Error: {e}"
            print(f"[{self.model_id}] Error durante la descarga: {e}")
            raise e

    def download_sync(self):
        """Inicia la descarga y bloquea el hilo hasta que termina."""
        if self.check_exists():
            return
        self._run_download()

    def start_download_bg(self):
        """Inicia la descarga como un proceso de fondo ('fire and forget')."""
        with self._status_lock:
            if self.status["is_downloading"]:
                return
            if self._download_thread is not None and self._download_thread.is_alive():
                return
            if self.check_exists():
                return
            
            # Avisamos a la UI inmediatamente antes de que levante el hilo
            self.status["is_downloading"] = True
            self.status["message"] = "Iniciando proceso de descarga..."
            
        def _task():
            try:
                self._run_download()
            except Exception as e:
                pass # El error ya fue capturado en _run_download()

        self._download_thread = threading.Thread(target=_task)
        self._download_thread.start()
