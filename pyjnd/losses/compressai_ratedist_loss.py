import torch
import torch.nn as nn
import math
from pyjnd.utils.registry import LOSS_REGISTRY
import importlib

@LOSS_REGISTRY.register()
class CompressaiRateDistLoss(nn.Module):
    """
    Rate-distortion loss proxy combining rate (bpp) and distortion using a CompressAI model.
    Dynamically imports a model from compressai.zoo, computes bpp_loss and distortion, and returns
    L = lmbda * distortion + bpp_loss.

    Args:
        model_name (str): Name of the model factory in compressai.zoo (e.g., 'mbt2018_an').
        quality (int): Quality setting passed to the model factory. Default is 5.
        pretrained (bool): Whether to load pretrained weights. Default is True.
        device (str): Device to place the model on. Default is 'cuda'.
        lmbda (float): Lagrangian multiplier for distortion. Default is 0.01.
        metric (str): Distortion metric: 'mse' or 'ms-ssim'. Default is 'mse'.
    """
    def __init__(
        self,
        model_name: str,
        quality: int = 5,
        pretrained: bool = True,
        device: str = 'cuda',
        lmbda: float = 0.01,
        metric: str = 'mse',
        reduction: str = 'mean',
        loss_weight: float = 1.0,
    ):
        super().__init__()
        # import and instantiate compressai model
        zoo = importlib.import_module('compressai.zoo')
        if not hasattr(zoo, model_name):
            raise ValueError(f"Model '{model_name}' not found in compressai.zoo")
        model_fn = getattr(zoo, model_name)
        model = model_fn(quality=quality, pretrained=pretrained).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self.model = model
        self.device = device
        self.lmbda = lmbda
        self.reduction = reduction
        self.loss_weight = loss_weight
        if metric == 'mse':
            self.metric_fn = nn.MSELoss()
        elif metric == 'ms-ssim':
            from pytorch_msssim import ms_ssim
            self.metric_fn = lambda x, y: ms_ssim(x, y, data_range=1)
        else:
            raise NotImplementedError(f"Metric '{metric}' not supported")

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        """
        Args:
            x (Tensor): Input tensor (N,C,H,W), values in [0,1].
            target (Tensor): Ground-truth tensor (N,C,H,W), values in [0,1].
        Returns:
            Tensor: Scalar loss = lmbda*distortion + bpp_loss.
        """
        x = x.to(self.device)
        target = target.to(self.device)
        # replicate grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # forward through compressai model
        output = self.model(x)
        # rate term: sum of log-likelihoods
        N, C, H, W = x.shape
        num_pixels = N * H * W
        bpp_losses = []
        for v in output['likelihoods'].values():
            log_sum = torch.log(v).sum()
            bpp_losses.append(log_sum / (-math.log(2) * num_pixels))
        bpp_loss = sum(bpp_losses).mean()
        # distortion term
        x_hat = output.get('x_hat', None)
        if x_hat is None:
            # decompress if necessary
            comp = self.model.compress(x)
            x_hat = self.model.decompress(comp['strings'], comp['shape']).clamp_(0,1)
        if isinstance(self.metric_fn, nn.MSELoss):
            mse = self.metric_fn(x_hat, target)
            distortion = 255**2 * mse
        else:
            ms = self.metric_fn(x_hat, target)
            distortion = 1 - ms
        # total loss
        loss = self.lmbda * distortion + bpp_loss

        # Reduction
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:  # 'mean'
            loss = loss.mean()

        return self.loss_weight * loss