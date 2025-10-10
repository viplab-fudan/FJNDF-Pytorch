#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post_train_weight_quanter.py  
(Re-revised 10 Jul 2025 - **Ambiguity-tolerant mapping, complete version**)

Post-training weight quantisation for PyTorch checkpoints **while preserving
*both* the top-level structure and the exact ordered list of parameter keys** so
that the result can be loaded by legacy code without any modifications.

Summary of features
-------------------
* Supports FP16 weight conversion (INT8 placeholder stub).
* Prints parameter dtypes before/after quantisation and simple error metrics.
* When saving, re-inserts quantised weights into the *original* checkpoint
  structure so that `ckpt.keys()` and `ckpt[weight_key].keys()` match exactly.
* Robust suffix-matching + shortest-prefix heuristic resolves prefix diffs like
  `net.conv.weight` vs `conv.weight`.  Ambiguities are reported.

Usage example
-------------
```bash
python post_train_weight_quanter.py \
    -c cfg.yaml \
    -o temp.pth \
    --dtype fp16
```
Where `cfg.yaml` contains at least:
```yaml
name: MyModel
pretrained_model_path: net_best_v0709_l1loss_mseloss_randomcrop_filter_4.pth
```

If the original checkpoint is missing, a *minimal* structure (just a plain
`state_dict`) will be created, so the script still works for fresh models.
"""

from __future__ import annotations

import argparse
import collections
import copy
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.quantization as quant
from torch.quantization.qconfig import QConfig
from torch.quantization.observer import default_observer, default_weight_observer, MinMaxObserver
import yaml

from onnx_converter import get_model  # project-specific

###############################################################################
# Helper utilities
###############################################################################

def print_param_dtypes(model: torch.nn.Module, label: str) -> None:
    """Pretty-print each parameter's dtype with a header *label*."""
    print(f"\n=== Parameter dtypes: {label} ===")
    for name, param in model.state_dict().items():
        print(f"{name:50s} : {param.dtype}")
    print(f"=== End of {label} ===\n")


def quantize_fp16(model: torch.nn.Module) -> torch.nn.Module:
    """Convert model weights to FP16 while keeping BatchNorm layers in FP32."""
    model.half()
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.float()
    return model


def quantize_int8_static(model: torch.nn.Module, dummy_shape: Tuple[int, ...]) -> torch.nn.Module:
    """Perform static INT8 quantization."""
    torch.backends.quantized.engine = "fbgemm"  # Ensure using fbgemm backend
    model.eval()

    # Configure quantization parameters
    default_qconfig = quant.get_default_qconfig("fbgemm")  # Use default configuration

    # Default QConfig (per-tensor quantization)
    default_qconfig = QConfig(
        activation=MinMaxObserver.with_args(reduce_range=False),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,  # Explicitly specify per-tensor quantization
            reduce_range=False
        )
    )
    
    # Replace ConvTranspose2d layers with per-tensor configuration
    convtrans_qconfig = QConfig(
        activation=MinMaxObserver.with_args(reduce_range=False),
        weight=MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False
        )
    )

    model.qconfig = default_qconfig

    # Replace ConvTranspose2d with per-tensor configuration
    for m in model.modules():
        if isinstance(m, torch.nn.ConvTranspose2d):
            m.qconfig = convtrans_qconfig        
 
    # Insert quantization observers (prepare model)
    model_prepared = quant.prepare(model, inplace=False)  # Return new model

    # Calibration (using real data)
    with torch.no_grad():
        model_prepared(torch.rand(dummy_shape))

    # Convert to INT8 model
    model_quantized = quant.convert(model_prepared, inplace=False)
    return model_quantized


def evaluate_error(
    model_fp32: torch.nn.Module,
    model_q: torch.nn.Module,
    dummy_shape: Tuple[int, ...],
) -> None:
    """Print MAE / MaxAE / MRE between original and quantised model outputs."""
    x_fp32 = torch.rand(dummy_shape)

    with torch.no_grad():
        y_ref = model_fp32.eval()(x_fp32)

    x_q = x_fp32.half() if any(p.dtype == torch.float16 for p in model_q.parameters()) else x_fp32
    with torch.no_grad():
        y_q = model_q.eval()(x_q)
        if isinstance(y_q, torch.Tensor) and y_q.dtype == torch.float16:
            y_q = y_q.float()

    diff = (y_ref - y_q).abs()
    mae, maxae = diff.mean().item(), diff.max().item()
    mre = (diff / y_ref.abs().clamp_min_(1e-12)).mean().item()

    print("Quantisation error (random input):")
    print(f"  MAE  : {mae:.6e}")
    print(f"  MaxE : {maxae:.6e}")
    print(f"  MRE  : {mre:.6e}")

###############################################################################
# Checkpoint structure helpers
###############################################################################

def load_original_ckpt_structure(
    ckpt_path: Path | str | None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Return (checkpoint_dict, weight_key). *weight_key* can be 'state_dict',
    'params', or '_root' (special wrapper when the checkpoint itself is the
    state_dict). Returns (None, None) if file not found or not a dict.
    """
    if ckpt_path is None or not Path(ckpt_path).is_file():
        return None, None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return None, None

    if "state_dict" in ckpt:
        return ckpt, "state_dict"
    if "params" in ckpt:
        return ckpt, "params"
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return {"_root": ckpt}, "_root"
    return None, None


def _find_matching_key(target: str, candidates) -> Optional[str]:
    """Return best candidate whose suffix matches *target*.

    * If exactly one candidate matches → return it.
    * If multiple match → choose the one with the *shortest prefix* (i.e.
      len(candidate) - len(target)).  Emit a warning listing all matches.
    """
    hits = [c for c in candidates if c.endswith(target)]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]
    # Ambiguous: pick shortest prefix
    chosen = min(hits, key=lambda x: len(x) - len(target))
    print(
        f"[Ambiguous map] {target} matched {len(hits)} candidates; "
        f"chosen '{chosen}' out of {hits}"
    )
    return chosen


def merge_quantised_weights(
    orig_ckpt: Dict,
    weight_key: str,
    q_state_dict: Dict[str, torch.Tensor],
) -> Dict:
    """Return new checkpoint with weights reordered/mapped to original keys.

    Suffix-matching bridges prefix differences like 'module.' or 'net.'.
    Ambiguities resolved heuristically; diagnostics are printed.
    """
    orig_weights = orig_ckpt[weight_key]
    ordered = collections.OrderedDict()
    missing = []
    q_keys = list(q_state_dict.keys())
    used_q = set()

    for k in orig_weights.keys():
        qk = k if k in q_state_dict else _find_matching_key(k, q_keys)
        if qk is None:
            missing.append(k)
            continue
        ordered[k] = q_state_dict[qk]
        used_q.add(qk)
        if qk != k:
            print(f"[Map] {k} <- {qk}")

    extra = [k for k in q_keys if k not in used_q]

    if missing:
        print("[Debug] Quantised model keys:")
        for kk in q_keys:
            print("   ", kk)
        raise KeyError("Quantised model is missing expected keys: " + ", ".join(missing))
    if extra:
        print("[Warning] Quantised model has additional keys not present in original: " + ", ".join(extra))

    orig_ckpt[weight_key] = ordered
    return orig_ckpt if weight_key != "_root" else orig_ckpt[weight_key]

###############################################################################
# Main
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-training weight quantisation with key-parity preservation",
    )
    parser.add_argument("-c", "--cfg-file", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("-d", "--dtype", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    args = parser.parse_args()

    # Currently only FP16 path is implemented
    # if args.dtype != "fp16":
    #     raise NotImplementedError("INT8 path is not yet implemented.")

    cfg = yaml.safe_load(args.cfg_file.read_text())
    pretrained_path = cfg.get("pretrained_model_path")

    # 1) Build FP32 model
    model_fp32 = get_model(cfg, device="cpu").eval()
    print_param_dtypes(model_fp32, "Before quantisation")

    # 2) Quantise (FP16)
    model_q = copy.deepcopy(model_fp32)
    dummy_shape = (args.batch, args.channel, args.height, args.width)
    if args.dtype == "fp16":
        model_q = quantize_fp16(model_q)
    if args.dtype == 'int8':
        model_q = quantize_int8_static(model_q, dummy_shape)
    print_param_dtypes(model_q, "After quantisation → FP16")

    # 3) Quick error metric on random input
    evaluate_error(model_fp32, model_q, dummy_shape)

    # 4) Load original checkpoint structure (if exists)
    orig_ckpt, weight_key = load_original_ckpt_structure(pretrained_path)
    if orig_ckpt is None:
        print("[Info] No existing checkpoint found - creating minimal structure.")
        weight_key = "_root"
        orig_ckpt = {weight_key: model_fp32.state_dict()}  # preserve fp32 order

    # 5) Merge FP16 weights back into original structure
    new_ckpt = merge_quantised_weights(orig_ckpt, weight_key, model_q.state_dict())

    # 6) Save
    torch.save(new_ckpt, str(args.output))
    print(f"Quantised checkpoint saved → {args.output}")


if __name__ == "__main__":
    main()