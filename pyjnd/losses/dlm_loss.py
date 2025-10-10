import math
import torch
from torch import nn
from pytorch_wavelets import DWTForward

from pyjnd.utils.registry import LOSS_REGISTRY
from .loss_util import _reduction_modes


def safe_crop(tensor: torch.Tensor, border_frac: float) -> torch.Tensor:
    """
    Crop a tensor by a fraction of its spatial size, safely handling small dimensions.
    tensor: (..., H, W) tensor
    border_frac: fraction to crop from each side
    """
    h, w = tensor.shape[-2], tensor.shape[-1]
    pad_h = int(border_frac * h)
    pad_w = int(border_frac * w)
    if pad_h <= 0 or pad_h * 2 >= h or pad_w <= 0 or pad_w * 2 >= w:
        return tensor
    return tensor[..., pad_h:-pad_h, pad_w:-pad_w]


def csf_torch(freqs: torch.Tensor) -> torch.Tensor:
    """Contrast Sensitivity Function in torch."""
    return (0.31 + 0.69 * freqs) * torch.exp(-0.29 * freqs)


def dlm_tensor(img_ref: torch.Tensor,
               img_dist: torch.Tensor,
               wavelet: str = 'db2',
               n_levels: int = 4,
               border_frac: float = 0.2,
               d2h: float = 3.0) -> torch.Tensor:
    """
    Compute DLM ratio (num/den) on single-channel torch tensors, clamped to [0,1],
    and detach gradient when den==0 or num>den to avoid spikes.
    img_ref, img_dist: (1, H, W) or (H, W)
    Returns: scalar tensor
    """
    if img_ref.ndim == 2:
        img_ref = img_ref.unsqueeze(0)
        img_dist = img_dist.unsqueeze(0)

    device = img_ref.device
    dtype = img_ref.dtype
    dwt = DWTForward(J=n_levels, wave=wavelet, mode='symmetric').to(device)
    x_ref = img_ref.unsqueeze(0).to(device)
    x_dist = img_dist.unsqueeze(0).to(device)
    _, Yh_ref = dwt(x_ref)
    _, Yh_dist = dwt(x_dist)
    Yh_ref = Yh_ref[::-1]
    Yh_dist = Yh_dist[::-1]

    H = img_ref.shape[-2]
    factor = torch.tensor(math.pi * H * d2h / 180, device=device, dtype=dtype)
    csf_filters = []
    for i, _ in enumerate(Yh_ref):
        freqs = factor / (2 ** (i+1))
        csf_filters.append(csf_torch(freqs.view(1,1,1,1)))

    num = torch.tensor(0.0, device=device, dtype=dtype)
    den = torch.tensor(0.0, device=device, dtype=dtype)
    eps = 1e-6

    for r_band, d_band, filt in zip(Yh_ref, Yh_dist, csf_filters):
        r = r_band * filt
        d = d_band * filt
        k = torch.clamp(d / (r + eps), 0.0, 1.0)
        rest = k * r
        add = d - rest
        thresh = (rest.abs().mean() + add.abs().mean()) / 30
        rest_m = torch.clamp(rest.abs() - thresh, min=0.0)

        core_r = safe_crop(r.squeeze(0), border_frac)
        core_m = safe_crop(rest_m.squeeze(0), border_frac)
        num = num + core_m.pow(3).sum().pow(1/3)
        den = den + core_r.abs().pow(3).sum().pow(1/3)

    # compute raw ratio and apply clamping to [0,1]
    raw = num / (den + eps)
    clamped = torch.clamp(raw, 0.0, 1.0)
    # mask invalid cases: den==0 or num>den
    valid = (den > 0) & (num <= den)
    # detach gradient for invalid to avoid spikes
    ratio = torch.where(valid, clamped, clamped.detach())
    return ratio


@LOSS_REGISTRY.register()
class DlmLoss(nn.Module):
    """DLM-based loss implemented fully in PyTorch, with autograd support,"""
    """clamped to [0,1], and invalid cases detached."""
    def __init__(self, loss_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = pred.shape[0]
        losses = []
        for i in range(batch_size):
            ref = target[i]
            dist = pred[i]
            if ref.shape[0] > 1:
                ref = ref.mean(dim=0, keepdim=True)
                dist = dist.mean(dim=0, keepdim=True)
            val = dlm_tensor(ref, dist)
            losses.append(val)
        loss_tensor = torch.stack(losses)
        if self.reduction == 'mean':
            loss = loss_tensor.mean()
        elif self.reduction == 'sum':
            loss = loss_tensor.sum()
        else:
            loss = loss_tensor
        return self.loss_weight * loss
