#!/usr/bin/env python3
# -*- coding: utf-8

import argparse
import torch
import yaml
import onnx
import onnxruntime as ort
from pathlib import Path

from pyjnd.api_helpers import get_model


def export_and_verify(cfg_path: str,
                      onnx_path: str,
                      dummy_shape: tuple,
                      device: torch.device = torch.device('cuda:0'),
                      opset: int = 13):
    """
    Load model configuration, build the model, export to ONNX, and verify numeric consistency.
    """
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model = get_model(cfg, device=device)
    model.eval()

    # Disable input checks
    model.check_input_range = False
    model.is_valid_input = lambda x: x

    # Create dummy input with uniform distribution in [0,1]
    dummy = torch.rand(*dummy_shape, device=device)
    dummy_in = dummy

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_in,
        onnx_path,
        export_params=True,
        opset_version=opset,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'},
        }
    )
    print(f"ONNX model exported: {onnx_path}")

    # Check the ONNX model structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid")

    # Compare outputs between PyTorch and ONNXRuntime
    with torch.no_grad():
        torch_out = model(dummy_in)
        if torch_out.ndim == 3:
            torch_out = torch_out.unsqueeze(0)
        torch_np = torch_out.cpu().numpy()

    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: dummy.cpu().numpy()})[0]
    ort_np = ort_out

    diff = (torch_np - ort_np).ravel()
    print(f"Max absolute error: {diff.max():.3e}, Mean absolute error: {diff.mean():.3e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a PyTorch JND model to ONNX and verify numeric consistency."
    )
    parser.add_argument(
        "--cfg_file", "-i",
        type=Path,
        required=True,
        help="Path to the model configuration file (YAML format)."
    )
    parser.add_argument(
        "--output_onnx", "-o",
        type=Path,
        required=True,
        help="Output path for the exported ONNX model."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for the dummy input (default: 1)."
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=3,
        help="Channel for the dummy input (default: 3)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Height for the dummy input (default: 768)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width for the dummy input (default: 512)."
    )
    args = parser.parse_args()

    dummy_shape = (args.batch, args.channel, args.height, args.width)
    export_and_verify(
        cfg_path=str(args.cfg_file),
        onnx_path=str(args.output_onnx),
        dummy_shape=dummy_shape
    )