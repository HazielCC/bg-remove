import os
from pathlib import Path
from huggingface_hub import snapshot_download

# Descargar CogVideoX-2b (Video Generation Model)
# Origen: https://huggingface.co/THUDM/CogVideoX-2b

# Usaremos la carpeta esperada por el backend en el disco.
# Asumiendo que el script se corre desde la raíz del proyecto.
model_id = "THUDM/CogVideoX-2b"
model_dir = Path("backend/models/cogvideox-2b")

# Crear el directorio si no existe
model_dir.mkdir(parents=True, exist_ok=True)

print(f"Descargando modelo {model_id}...")
print(f"Destino: {model_dir.absolute()}")

try:
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False
    )
    print(f"\nDescarga completada en {model_dir.absolute()}")
except Exception as e:
    print(f"\nError durante la descarga: {e}")
