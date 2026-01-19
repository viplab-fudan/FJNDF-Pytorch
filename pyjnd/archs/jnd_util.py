from __future__ import annotations

from typing import Any, Dict, Mapping, Union

import torch

def fuse_jnd_maps(
    maps: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    method: Union[str, Mapping[str, Any]]
) -> torch.Tensor:
    """
    Generic JND map fusion function.

    Supports tensor shapes in maps as (C,H,W) or (H,W).

    Notes on `method`:
      - method can be a string (e.g., 'product', 'max', 'sum')
      - or a dict loaded from YAML, e.g.:
          method:
            name: wu2013_overlap_min
            overlap: 0.3
            stages: [["LA", "CM"], ["PM"]]
        In this case, `name` selects the fusion rule and other fields are treated as parameters.

    Args:
      maps   : dict of name->torch.Tensor, each tensor shape is (C,H,W) or (H,W)
      weights: dict of name->float weights, only used when method=='weighted'
      method : 'sum'|'weighted'|'product'|'max'|'wu2013_overlap_min'

    Returns:
      torch.Tensor, fused JND map, shape consistent with input
    """
    if not maps:
        raise ValueError("maps dictionary cannot be empty")

    # Allow method to be either a plain string or a YAML dict.
    method_name: str
    method_params: Mapping[str, Any] = {}
    if isinstance(method, Mapping):
        method_params = method
        method_name = str(method.get('name') or method.get('method') or method.get('type') or '').strip()
        if not method_name:
            raise ValueError("Fusion method dict must contain a 'name' (or 'method'/'type') field")
    else:
        method_name = str(method).strip()

    # Get all branch names and tensors in dictionary order
    names = list(maps.keys())

    # If single channel (H,W), expand to (1,H,W) locally (do NOT mutate caller's dict)
    is_2d = maps[names[0]].dim() == 2
    maps_3d: Dict[str, torch.Tensor] = {}
    for n in names:
        t = maps[n]
        maps_3d[n] = t.unsqueeze(0) if is_2d else t

    # tensors are now all (C,H,W)
    tensors = [maps_3d[n] for n in names]

    # Now tensors are all (C,H,W)
    if method_name == 'product':
        result = torch.ones_like(tensors[0])
        for m in tensors:
            result = result * m

    elif method_name == 'max':
        # Stack then take max along dimension 0
        stacked = torch.stack(tensors, dim=0)    # (N,C,H,W)
        result, _ = torch.max(stacked, dim=0)    # (C,H,W)

    elif method_name in ('sum', 'weighted'):
        # Ensure each branch has corresponding weight
        missing = [n for n in names if n not in weights]
        if missing:
            raise KeyError(f"Missing branch weight settings: {missing}")

        result = torch.zeros_like(tensors[0])
        for name, m in zip(names, tensors):
            w = weights[name]
            result = result + m * w

    elif method_name == 'namm_tmm_wu_2013':
        C = float(method_params.get("overlap", 0.3))
        try:
            la, cm, pm = (maps_3d[k] for k in ("LA", "CM", "PM"))
        except KeyError as e:
            raise KeyError(f"wu2013 fusion requires keys LA/CM/PM, got: {list(maps_3d.keys())}") from e

        # out = A + B - C * min(A, B)
        out = la + cm - C * torch.minimum(la, cm)
        result = out + pm - C * torch.minimum(out, pm)

    elif method_name == 'namm_tip_wu_2017':
        C = float(method_params.get("overlap", 0.3))
        try:
            la, cm, pm = (maps_3d[k] for k in ("LA", "CM", "PM"))
        except KeyError as e:
            raise KeyError(f"wu2017 fusion requires keys LA/CM/PM, got: {list(maps_3d.keys())}") from e

        # MS = max(CM, PM)
        ms = torch.maximum(cm, pm)
        # TJND = LA + MS - C * min(LA, MS)
        result = la + ms - C * torch.minimum(la, ms)

    else:
        raise ValueError(f"Unsupported fusion method '{method}'")

    # If original input was (H,W), squeeze back
    if is_2d:
        result = result.squeeze(0)

    return result