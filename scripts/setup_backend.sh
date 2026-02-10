#!/bin/bash
# Setup script for backend Python environment (using uv)
# Run from the bg-remove/ root directory

set -e

BACKEND_DIR="backend"
MODELS_DIR="$BACKEND_DIR/models"

echo "══════════════════════════════════════════════"
echo "  MODNet Fine-Tuning Backend Setup (uv)"
echo "══════════════════════════════════════════════"

# 1. Check uv is available
if ! command -v uv &>/dev/null; then
    echo "  ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo ""
echo "▸ Using: uv $(uv --version)"

# 2. Install dependencies with uv
echo ""
echo "▸ Installing Python dependencies..."
cd "$BACKEND_DIR"
uv sync
echo "  Done!"
cd ..

# 3. Create .env from example
echo ""
echo "▸ Setting up environment..."
if [ ! -f "$BACKEND_DIR/.env" ]; then
    cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
    echo "  Created .env from .env.example"
else
    echo "  .env already exists"
fi

# 4. Create required directories
echo ""
echo "▸ Creating directories..."
mkdir -p "$BACKEND_DIR/data"
mkdir -p "$BACKEND_DIR/checkpoints"
mkdir -p "$BACKEND_DIR/exports"
mkdir -p "$MODELS_DIR"
echo "  Created: data/, checkpoints/, exports/, models/"

# 5. Download pretrained MODNet weights (optional)
echo ""
echo "▸ Pretrained MODNet weights..."
CKPT_PATH="$MODELS_DIR/modnet_webcam_portrait_matting.ckpt"
if [ ! -f "$CKPT_PATH" ]; then
    echo "  To download pretrained weights, run:"
    echo ""
    echo "    curl -L -o $CKPT_PATH \\"
    echo "      https://drive.google.com/uc?export=download\\&id=1mcr7ALciAb8es1YkR3lXq72JBWsbVHKa"
    echo ""
    echo "  Or download manually from: https://github.com/ZHKKKe/MODNet"
else
    echo "  Already downloaded: $CKPT_PATH"
fi

# 6. Verify PyTorch + MPS
echo ""
echo "▸ Verifying PyTorch..."
cd "$BACKEND_DIR"
uv run python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  MPS available: {torch.backends.mps.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
"
cd ..

echo ""
echo "══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Start backend:  cd backend && uv run uvicorn main:app --reload --port 8000"
echo "  Start frontend: pnpm dev"
echo "  Or both:        pnpm dev:all"
echo "══════════════════════════════════════════════"
