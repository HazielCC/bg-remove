import os
import urllib.request
from pathlib import Path

# Descargar Qwen3.5-0.8B-LiteRT (int8) optimizado para Web/Mobile
# Origen: https://huggingface.co/litert-community/Qwen3.5-0.8B-LiteRT

model_dir = Path("public/models/litert")
model_file = "qwen3.5-0.8b-int8.tflite"
model_path = model_dir / model_file

# Crear el directorio si no existe
model_dir.mkdir(parents=True, exist_ok=True)

url = "https://huggingface.co/litert-community/Qwen3.5-0.8B-LiteRT/resolve/main/model_quantized.tflite?download=true"

print("Descargando modelo LiteRT Qwen3.5 0.8B (Aprox. 800MB)...")
print(f"Destino: {model_path.absolute()}")

try:
    urllib.request.urlretrieve(url, str(model_path))
    print(f"\nDescarga completada en {model_path.absolute()}")
except Exception as e:
    print(f"\nError durante la descarga: {e}")
