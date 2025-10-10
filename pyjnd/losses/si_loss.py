import torch
import torch.nn as nn
import torch.nn.functional as F

from pyjnd.utils.registry import LOSS_REGISTRY
from .loss_util import _reduction_modes


@LOSS_REGISTRY.register()
class SILoss(nn.Module):
    """Spatial Information (SI) loss as a proxy for coding rate cost.

    SI is defined as the population standard deviation of the Sobel-gradient
    magnitude of the image luminance (Y) channel. Higher SI indicates more
    texture/edges (higher bitrate), so minimizing SI encourages smoother
    reconstructions.

    Args:
        loss_weight (float): Scalar multiplier for the SI loss. Default: 1.0.
        reduction (str): Specifies reduction over batch:
            'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )
        self.loss_weight = loss_weight
        self.reduction = reduction

        # Sobel kernels for gradient computation
        kernel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]], dtype=torch.float32
        )
        kernel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [ 1.,  2.,  1.]], dtype=torch.float32
        )
        # shape (1,1,3,3)
        self.register_buffer('sobel_x', kernel_x.view(1,1,3,3))
        self.register_buffer('sobel_y', kernel_y.view(1,1,3,3))

    def forward(self, pred: torch.Tensor, target=None, weight=None, **kwargs) -> torch.Tensor:
        """
        Args:
            pred (Tensor): Input tensor of shape (N, C, H, W), values in [0,1] or [0,255].
            target: Ignored for SI (unreferenced).
            weight: Ignored.
        Returns:
            Tensor: SI loss, reduced per `reduction` and scaled by `loss_weight`.
        """
        # convert to luminance Y if RGB, else assume single-channel
        x = pred
        if x.size(1) == 3:
            # Rec. 601 luma transform
            r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
            y = 0.299*r + 0.587*g + 0.114*b
        else:
            y = x

        # compute Sobel gradients
        gx = F.conv2d(y, self.sobel_x, padding=1)
        gy = F.conv2d(y, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx * gx + gy * gy)

        # flatten spatial dims and compute population std dev per sample
        N, _, H, W = grad_mag.shape
        grad_flat = grad_mag.view(N, -1)
        si_vals = torch.std(grad_flat, dim=1, unbiased=False)  # shape [N]

        # apply reduction over batch
        if self.reduction == 'none':
            loss = si_vals
        elif self.reduction == 'mean':
            loss = si_vals.mean()
        else:  # 'sum'
            loss = si_vals.sum()

        return self.loss_weight * loss
