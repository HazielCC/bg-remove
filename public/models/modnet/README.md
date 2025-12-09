# MODNet models

This folder is intended to host ONNX weights used by the browser demo.

Recommended files:
- model_quantized.onnx  (small, fast)
- model_fp16.onnx       (fast on WebGPU)
- model.onnx            (fp32 reference)

Download command examples:

```bash
mkdir -p public/models/modnet/onnx
curl -L -o public/models/modnet/onnx/model_quantized.onnx "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model_quantized.onnx"
```
