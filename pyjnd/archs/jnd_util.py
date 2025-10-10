from typing import Dict
import torch

def fuse_jnd_maps(
    maps: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    method: str
) -> torch.Tensor:
    """
    Generic JND map fusion function, supports tensor shapes in maps as (C,H,W) or (H,W).

    Args:
      maps   : dict of name->torch.Tensor, each tensor shape is (C,H,W) or (H,W)
      weights: dict of name->float weights, only used when method=='weighted'
      method : 'sum'|'product'|'max'|'weighted'

    Returns:
      torch.Tensor, fused JND map, shape consistent with input
    """
    if not maps:
        raise ValueError("maps dictionary cannot be empty")

    # Get all branch names and tensors in dictionary order
    names = list(maps.keys())

    # If single channel (H,W), first expand in-place to (1,H,W) in maps
    is_2d = False
    if maps[names[0]].dim() == 2:
        is_2d = True
        for n in names:
            maps[n] = maps[n].unsqueeze(0)

    # tensors are now all (C,H,W)
    tensors = [maps[n] for n in names]
    ws = [weights[n] for n in names]

    # Now tensors are all (C,H,W)
    if method == 'product':
        result = torch.ones_like(tensors[0])
        for m in tensors:
            result = result * m

    elif method == 'max':
        # Stack then take max along dimension 0
        stacked = torch.stack(tensors, dim=0)    # (N,C,H,W)
        result, _ = torch.max(stacked, dim=0)    # (C,H,W)

    elif method == 'sum':
        # Ensure each branch has corresponding weight
        missing = [n for n in names if n not in weights]
        if missing:
            raise KeyError(f"Missing branch weight settings: {missing}")

        result = torch.zeros_like(tensors[0])
        for name, m in zip(names, tensors):
            w = weights[name]
            result = result + m * w

    else:
        raise ValueError(f"Unsupported fusion method '{method}'")

    # If original input was (H,W), squeeze back
    if is_2d:
        result = result.squeeze(0)

    return result