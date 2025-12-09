# Remove BG — demo

This folder contains a small demo that integrates the MODNet model via Transformers.js for browser background removal.

How to test locally

1. Start dev server:

```bash
pnpm dev
# or
npm run dev
```

2. Open the demo:

```
http://localhost:3002/remove-bg
```

3. Use the example image (`/example/person.jpg`) or upload an image (file input) to run the model.

Notes
- Model files are expected in `public/models/modnet/` with `config.json` and `preprocessor_config.json` present.
- `public/models/modnet/onnx/` contains the ONNX weights (quantized/fp16/fp32) — download them from Hugging Face if you need more variants.
- The UI has a `Model variant` select: `auto`, `fp32`, `fp16`, or `uint8`.

To update weights using curl:

```bash
mkdir -p public/models/modnet/onnx
curl -L -o public/models/modnet/onnx/model_quantized.onnx "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model_quantized.onnx"
```

If you want a Node API instead of browser inference, see `backend/` (not implemented yet) or use `scripts/test_segmenter_node.mjs` to validate model loading and inference in Node.
