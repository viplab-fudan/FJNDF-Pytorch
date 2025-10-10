import torch
from torch import nn
import pyiqa
from pyjnd.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss, _reduction_modes

@LOSS_REGISTRY.register()
class PyiqaLoss(nn.Module):
    """Generic IQA loss wrapper based on pyiqa metrics.

    Args:
        metric_name (str): Name of the metric to use (e.g., 'lpips', 'niqe', 'ssim').
        loss_weight (float): Scaling factor for the loss output. Default: 1.0.
        device (str or torch.device): Device on which to run the metric. Default: 'cuda'.
        need_ref (bool): Whether this is a full-reference metric (requires both pred and target).
            If False, only pred is used (no-reference). Default: True.
        lower_better (bool): If True, lower metric scores indicate better quality (loss = score).
            If False, higher metric scores are better (loss = 1 - score). Default: True.
        reduction (str): Specifies batch reduction: 'none' | 'mean' | 'sum'. Default: 'mean'.
        **metric_kwargs: Additional keyword args for `pyiqa.create_metric`.
    """
    def __init__(
        self,
        metric_name: str,
        loss_weight: float = 1.0,
        device='cuda',
        need_ref: bool = True,
        lower_better: bool = True,
        reduction: str = 'mean',
        **metric_kwargs
    ):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.need_ref = need_ref
        self.lower_better = lower_better
        self.reduction = reduction
        # instantiate pyiqa metric with gradient support
        self.metric = pyiqa.create_metric(
            metric_name, device=device, as_loss=True, **metric_kwargs
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred (Tensor): Predicted images, shape (N, C, H, W).
            target (Tensor, optional): Reference images for FR metrics. Ignored if `need_ref=False`.
        Returns:
            Tensor: Loss value (scalar if reduced, else per-sample tensor).
        """
        # compute raw metric score
        if self.need_ref:
            if target is None:
                raise ValueError('`target` must be provided for full-reference metrics.')
            score = self.metric(pred, target)
        else:
            score = self.metric(pred)
        # convert metric to loss: lower is better or vice versa
        loss = score if self.lower_better else (1.0 - score)

        return self.loss_weight * loss
