import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor

from pyjnd.utils.registry import LOSS_REGISTRY
from .loss_util import _reduction_modes


def _normalize(N: int) -> Tensor:
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return n @ n.t()


def _harmonics(N: int) -> Tensor:
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def blockify(im, size):
    b, c, h, w = im.shape
    
    im = im.reshape(b*c, 1, h, w)
    im = torch.nn.functional.unfold(im, kernel_size=(size, size), stride=(size, size))
    im = im.transpose(1, 2)
    im = im.reshape(b, c, -1, size, size)

    return im


def block_dct(blocks: Tensor) -> Tensor:
    N = blocks.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if blocks.is_cuda:
        n = n.cuda()
        h = h.cuda()

    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ blocks @ h)

    return coeff


def _zigzag_indices(b: int):
    """return 8x8  zigzag  [(u,v), ...]"""
    order = []
    for s in range(2 * b - 1):
        for u in range(b):
            v = s - u
            if 0 <= v < b:
                order.append((u, v))
    return order


@LOSS_REGISTRY.register()
class FrequencyDomainResidualLoss(nn.Module):
    """
    Corresponds to L_dct^res in the paper.
    Calculates the energy of the residual's DCT coefficients between the
    network output (pred) and the reference (target).
    Serves as a distillation loss in the frequency domain.
    """
    def __init__(self, loss_weight: float = 1.0, reduction: str = 'mean', block: int = 8):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.block = int(block)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        assert target is not None, "FrequencyDomainResidualLoss needs `target`."
        assert pred.shape == target.shape, "pred/target shape mismatch."

        residual = pred - target

        blocks = blockify(residual, self.block)
        coeffs = block_dct(blocks)
        energy = coeffs.pow(2).sum(dim=(-1, -2))
        
        if self.reduction == 'mean':
            loss = energy.mean()
        elif self.reduction == 'sum':
            loss = energy.sum()
        else:
            loss = energy
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class FrequencyConservationConstraintLoss(nn.Module):
    """
    Corresponds to L_dct^cons in the paper.
    Imposes a bidirectional, asymmetric constraint by comparing the network
    output (pred) with the original input (ori).
    - Prevents over-smoothing by penalizing loss of low-frequency energy.
    - Suppresses artifacts by penalizing creation of new high-frequency energy.
    """
    def __init__(self, loss_weight: float = 1.0, reduction: str = 'mean', block: int = 8,
                 zigzag_cutoff: int = 10, low_band_weight: float = 1.0, high_band_weight: float = 1.0):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.block = int(block)
        self.zigzag_cutoff = int(zigzag_cutoff)
        self.low_band_weight = float(low_band_weight)
        self.high_band_weight = float(high_band_weight)

        zz = _zigzag_indices(self.block)
        assert 0 < self.zigzag_cutoff < len(zz), "zigzag_cutoff out of range."
        self.register_buffer(
            'low_idx_mask',
            self._build_mask(self.block, zz[:self.zigzag_cutoff]),
            persistent=False
        )
        self.register_buffer(
            'high_idx_mask',
            self._build_mask(self.block, zz[self.zigzag_cutoff:]),
            persistent=False
        )

    @staticmethod
    def _build_mask(b: int, coords: list[tuple[int, int]]) -> torch.Tensor:
        m = torch.zeros((b, b), dtype=torch.bool)
        for (u, v) in coords:
            m[u, v] = True
        return m

    def forward(self, pred: torch.Tensor, ori: torch.Tensor, **kwargs) -> torch.Tensor:
        assert ori is not None, "FrequencyConservationConstraintLoss needs `ori`."
        assert pred.shape == ori.shape, "pred/ori shape mismatch."

        p_blocks = blockify(pred, self.block)
        x_blocks = blockify(ori, self.block)

        Cp = block_dct(p_blocks)
        Cx = block_dct(x_blocks)

        Cp_abs = Cp.abs()
        Cx_abs = Cx.abs()
        
        # Penalize when low-frequency energy in pred is less than in ori
        low_mask = self.low_idx_mask.view(1, 1, 1, 1, self.block, self.block)
        low_freq_loss = (Cx_abs - Cp_abs).clamp_min(0.0) * low_mask

        # Penalize when high-frequency energy in pred is greater than in ori
        high_mask = self.high_idx_mask.view(1, 1, 1, 1, self.block, self.block)
        high_freq_loss = (Cp_abs - Cx_abs).clamp_min(0.0) * high_mask

        loss_low = low_freq_loss.sum(dim=(-1, -2))
        loss_high = high_freq_loss.sum(dim=(-1, -2))

        loss_map = self.low_band_weight * loss_low + self.high_band_weight * loss_high

        if self.reduction == 'mean':
            loss = loss_map.mean()
        elif self.reduction == 'sum':
            loss = loss_map.sum()
        else:
            loss = loss_map

        return self.loss_weight * loss