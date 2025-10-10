import torch
from torch import nn
from piq import VIFLoss as _VIFLoss

from pyjnd.utils.registry import LOSS_REGISTRY
from .loss_util import _reduction_modes

@LOSS_REGISTRY.register()
class VifLoss(nn.Module):
    """Visual Information Fidelity (VIF) loss.

    Args:
        loss_weight (float): Scalar multiplier for the loss. Default: 1.0.
        data_range (float): Value range of input images (max - min). Default: 1.0.
        sigma_n_sq (float): Variance of the visual distortion models assumed noise. Default: 2.0.
        reduction (str): Specifies how to reduce per-sample losses to a scalar: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        data_range: float = 1.0,
        sigma_n_sq: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        if reduction not in _reduction_modes:
            raise ValueError(f"Unsupported reduction mode: {reduction}. "
                             f"Supported ones are: {_reduction_modes}")
        self.loss_weight = loss_weight
        # we compute VIF score per-sample with no reduction here
        self.vif = _VIFLoss(sigma_n_sq=sigma_n_sq,
                            data_range=data_range,
                            reduction='none')
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            pred (Tensor): Predicted images (N, C, H, W), values in [0, data_range].
            target (Tensor): Ground-truth images (N, C, H, W), same range.
        Returns:
            Tensor: VIF-based loss.
        """
        # compute per-sample VIF score ∈ [0,1]
        vif_score = self.vif(pred, target)
        # convert fidelity score to loss
        loss = 1.0 - torch.clamp(vif_score, 0.0, 1.0)

        # apply reduction
        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            loss = loss.mean()
        else:  # 'sum'
            loss = loss.sum()

        return self.loss_weight * loss