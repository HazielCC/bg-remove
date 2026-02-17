"""
ONNX export and quantization for fine-tuned MODNet models.

Supports:
  - Export to ONNX (fp32)
  - Quantize to FP16 (via onnxconverter-common)
  - Quantize to UINT8 (via onnxruntime dynamic quantization)
"""

from pathlib import Path

import torch
import onnx

from ml.modnet import MODNet, MODNetInference


def export_to_onnx(
    ckpt_path: str,
    output_path: str,
    img_size: int = 512,
    device: str = "cpu",
    opset_version: int = 14,
) -> str:
    """
    Export a MODNet checkpoint to ONNX format.

    Args:
        ckpt_path: path to .ckpt file
        output_path: path for output .onnx file
        img_size: input image size (square)
        device: device to use for export
        opset_version: ONNX opset version

    Returns:
        Path to the exported ONNX file.
    """
    model = MODNet.from_pretrained(ckpt_path, device=device, backbone_pretrained=False)
    wrapper = MODNetInference(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        # Ensure weights are embedded in the .onnx file (avoid .onnx.data split)
        # This keeps the model fully self-contained for easy deployment.
    )

    # Verify
    model_onnx = onnx.load(str(out))
    onnx.checker.check_model(model_onnx)

    return str(out)


def quantize_fp16(input_onnx: str, output_onnx: str) -> str:
    """
    Convert ONNX model to FP16.

    Args:
        input_onnx: path to fp32 ONNX model
        output_onnx: path for fp16 ONNX output

    Returns:
        Path to quantized model.
    """
    from onnxconverter_common import float16

    model = onnx.load(input_onnx)
    model_fp16 = float16.convert_float_to_float16(model)

    out = Path(output_onnx)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(out))

    return str(out)


def quantize_uint8(input_onnx: str, output_onnx: str) -> str:
    """
    Dynamic quantization to UINT8.

    Args:
        input_onnx: path to ONNX model
        output_onnx: path for quantized output

    Returns:
        Path to quantized model.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    out = Path(output_onnx)
    out.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        input_onnx,
        str(out),
        weight_type=QuantType.QUInt8,
    )

    return str(out)


def get_onnx_info(onnx_path: str) -> dict:
    """Get basic information about an ONNX model."""
    model = onnx.load(onnx_path)
    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)

    inputs = []
    for inp in model.graph.input:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param
            for d in inp.type.tensor_type.shape.dim
        ]
        inputs.append({"name": inp.name, "shape": shape})

    outputs = []
    for out in model.graph.output:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param
            for d in out.type.tensor_type.shape.dim
        ]
        outputs.append({"name": out.name, "shape": shape})

    return {
        "path": onnx_path,
        "size_mb": round(size_mb, 2),
        "opset_version": model.opset_import[0].version if model.opset_import else None,
        "inputs": inputs,
        "outputs": outputs,
        "n_nodes": len(model.graph.node),
    }
