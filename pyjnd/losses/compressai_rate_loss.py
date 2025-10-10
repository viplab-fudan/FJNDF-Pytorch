import torch
import torch.nn as nn
import math
from pyjnd.utils.registry import LOSS_REGISTRY
import importlib

@LOSS_REGISTRY.register()
class CompressaiRateLoss(nn.Module):
    """
    Rate-distortion loss proxy computing the rate term (bpp) using a CompressAI model specified by name.
    The model is dynamically imported from compressai.zoo and instantiated with given quality and pretrained flag.
    Computes bpp_loss as in RateDistortionLoss:
        bpp_loss = sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                       for likelihoods in output['likelihoods'].values())

    Args:
        model_name (str): Name of the model factory in compressai.zoo (e.g., 'mbt2018_an').
        quality (int): Quality setting passed to the model factory. Default is 5.
        pretrained (bool): Whether to load pretrained weights. Default is True.
        device (str): Device to place the model on. Default is 'cuda'.
        reduction (str): Reduction method over batch: 'none', 'mean', or 'sum'. Default is 'mean'.
        loss_weight (float): Scaling factor for the loss. Default is 1.0.
    """
    def __init__(
        self,
        model_name: str,
        quality: int = 5,
        pretrained: bool = True,
        device: str = 'cuda',
        reduction: str = 'mean',
        loss_weight: float = 1.0,
    ):
        super().__init__()
        # Dynamically import the model factory from compressai.zoo
        zoo_module = importlib.import_module('compressai.zoo')
        if not hasattr(zoo_module, model_name):
            raise ValueError(f"Model '{model_name}' not found in compressai.zoo")
        model_fn = getattr(zoo_module, model_name)
        # Instantiate and freeze the model
        model = model_fn(quality=quality, pretrained=pretrained).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.device = device
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, x: torch.Tensor, target=None, **kwargs) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), values in [0,1] or [0,255].
            target: Ignored for rate computation.
        Returns:
            torch.Tensor: Loss tensor (bpp_loss), scalar or per-sample depending on reduction.
        """
        x = x.to(self.device)
        # For gray inputs, replicate channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Forward through model: get 'likelihoods' for each entropy module
        output = self.model(x)
        N, C, H, W = x.size()
        num_pixels = N * H * W

        # Compute bpp_loss exactly as in RateDistortionLoss
        bpp_losses = []
        for likelihoods in output['likelihoods'].values():
            # likelihoods: Tensor shape (N, H', W')
            log_sum = torch.log(likelihoods).sum()  # per-sample
            bpp = log_sum / (-math.log(2) * num_pixels)
            bpp_losses.append(bpp)

        # Sum contributions from all modules → shape (N,)
        bpp_loss = sum(bpp_losses)

        # Reduction
        if self.reduction == 'none':
            loss = bpp_loss
        elif self.reduction == 'sum':
            loss = bpp_loss.sum()
        else:  # 'mean'
            loss = bpp_loss.mean()

        return self.loss_weight * loss