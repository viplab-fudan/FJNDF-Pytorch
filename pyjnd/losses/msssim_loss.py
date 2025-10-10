import torch
from torch import nn
from pytorch_msssim import ms_ssim

from pyjnd.utils.registry import LOSS_REGISTRY
from .loss_util import _reduction_modes, reduce_loss

@LOSS_REGISTRY.register()
class MsssimLoss(nn.Module):
    """MS-SSIM (Multi-Scale Structural Similarity) loss.

    Args:
        loss_weight (float): Scalar multiplier for the loss. Default: 1.0.
        data_range (float): Value range of input images (max - min). Default: 1.0.
        win_size (int): Size of the Gaussian filter window. Default: 11.
        win_sigma (float): Standard deviation of the Gaussian window. Default: 1.5.
        weights (list or None): Scale weights for each level. Default: None (uses default in ms_ssim).
        reduction (str): Specifies how to reduce per-sample losses to a scalar: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(
        self,
        loss_weight=1.0,
        data_range=1.0,
        win_size=11,
        win_sigma=1.5,
        weights=None,
        reduction='mean'
    ):
        super(MsssimLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}'
            )
        self.loss_weight = loss_weight
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.weights = weights
        self.reduction = reduction

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): Predicted images (N, C, H, W), values in [0, data_range].
            target (Tensor): Ground-truth images (N, C, H, W), same range.
        """
        # compute per-sample MSSSIM score
        msssim_val = ms_ssim(
            pred,
            target,
            data_range=self.data_range,
            win_size=self.win_size,
            win_sigma=self.win_sigma,
            weights=self.weights,
            size_average=False
        )
        # convert similarity to loss
        loss = 1 - msssim_val

        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss.sum()

        return self.loss_weight * loss
