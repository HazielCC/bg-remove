#!/bin/bash

# Descargar Qwen2.5-1.5B-Instruct (dynamic_int8) optimizado con LiteRT
# Origen: https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct

MODEL_DIR="public/models/litert"
MODEL_FILE="qwen2.5-1.5b-int8.tflite"

mkdir -p "$MODEL_DIR"

echo "Descargando modelo LiteRT (Aprox. 1.5GB)..."
curl -L "https://huggingface.co/litert-community/Qwen2.5-1.5B-Instruct/resolve/main/model_dynamic_int8.tflite?download=true" -o "$MODEL_DIR/$MODEL_FILE"

echo "Descarga completada en $MODEL_DIR/$MODEL_FILE"
