#!/bin/bash

# Descargar Qwen3.5-0.8B-LiteRT (int8) optimizado para Web/Mobile
# Origen: https://huggingface.co/litert-community/Qwen3.5-0.8B-LiteRT

MODEL_DIR="public/models/litert"
MODEL_FILE="qwen3.5-0.8b-int8.tflite"

mkdir -p "$MODEL_DIR"

echo "Descargando modelo LiteRT Qwen3.5 0.8B (Aprox. 800MB)..."
# The exact filename usually follows standard LiteRT naming conventions in HF.
# Often named model.tflite or qwen3.5-0.8b-instruct-int8.tflite. Let's use the standard resolve URL.
curl -L "https://huggingface.co/litert-community/Qwen3.5-0.8B-LiteRT/resolve/main/model.tflite?download=true" -o "$MODEL_DIR/$MODEL_FILE"

echo "Descarga completada en $MODEL_DIR/$MODEL_FILE"
