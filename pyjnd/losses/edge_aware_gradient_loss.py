import torch
import torch.nn as nn
import torch.nn.functional as F

from pyjnd.utils.registry import LOSS_REGISTRY
from .loss_util import _reduction_modes

def _kernels(kind: str, dtype, device):
    if kind == 'sobel':
        kx = torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], dtype=dtype, device=device).view(1,1,3,3)
        ky = torch.tensor([[ 1.,  2.,  1.],
                           [ 0.,  0.,  0.],
                           [-1., -2., -1.]], dtype=dtype, device=device).view(1,1,3,3)
    elif kind == 'scharr':
        kx = torch.tensor([[ 3.,  0., -3.],
                           [10.,  0., -10.],
                           [ 3.,  0., -3.]], dtype=dtype, device=device).view(1,1,3,3)
        ky = torch.tensor([[ 3., 10.,  3.],
                           [ 0.,  0.,  0.],
                           [-3.,-10., -3.]], dtype=dtype, device=device).view(1,1,3,3)
    else:
        raise ValueError(f'Unknown kernel type: {kind}')
    return kx, ky


@LOSS_REGISTRY.register()
class EdgeAwareGradLoss(nn.Module):
    """
    Edge-aware gradient consistency loss (no ROI required).

    It aligns gradients between prediction and target while emphasizing
    edge pixels automatically, using the target's gradient magnitude as
    a per-pixel weight map.

    Args:
        loss_weight (float): global multiplier. Default: 1.0
        reduction (str): 'none' | 'mean' | 'sum'. Default: 'mean'
        kernel_type (str): 'scharr' | 'sobel'. Default: 'scharr'
        mag_weight (float): weight for |∇p|-|∇t| term. Default: 1.0
        ori_weight (float): weight for orientation term (1 - cos). Default: 1.0
        edge_strength (float): in [0,1], how strongly to emphasize edges.
            0 -> uniform, 1 -> fully use normalized edge weights. Default: 0.6
        gamma (float): exponent for edge weights; >1 focuses more on strong edges. Default: 1.0
        scales (tuple): multi-scale pyramid, e.g. (1.0, 0.5). Default: (1.0, 0.5)
        scale_weights (tuple|None): per-scale weights (auto-normalized). Default: None (uniform)
        eps (float): numeric stability. Default: 1e-12
    """
    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 kernel_type: str = 'scharr',
                 mag_weight: float = 1.0,
                 ori_weight: float = 1.0,
                 edge_strength: float = 0.6,
                 gamma: float = 1.0,
                 scales=(1.0, 0.5),
                 scale_weights=None,
                 eps: float = 1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported: {_reduction_modes}')
        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.kernel_type = kernel_type.lower()
        self.mag_weight = float(mag_weight)
        self.ori_weight = float(ori_weight)
        self.edge_strength = float(edge_strength)
        self.gamma = float(gamma)
        self.scales = tuple(scales)
        if scale_weights is None:
            self.scale_weights = [1.0/len(self.scales)] * len(self.scales)
        else:
            sw = torch.tensor(scale_weights, dtype=torch.float32)
            sw = (sw / sw.sum()).tolist()
            self.scale_weights = sw
        self.eps = float(eps)

        kx, ky = _kernels(self.kernel_type, torch.float32, 'cpu')
        self.register_buffer('kx_base', kx, persistent=False)
        self.register_buffer('ky_base', ky, persistent=False)

    @staticmethod
    def _sel_y(x: torch.Tensor, use_y: bool):
        return x[:, :1] if (use_y and x.size(1) >= 1) else x

    @staticmethod
    def _down(x: torch.Tensor, s: float):
        if s == 1.0: return x
        H, W = x.shape[-2:]
        return F.interpolate(x, size=(max(1,int(H*s)), max(1,int(W*s))),
                             mode='bilinear', align_corners=False)

    def _grad(self, x: torch.Tensor):
        # replicate padding avoids border bias
        n, c, h, w = x.shape
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')    

        # make kernels match x's dtype & device
        kx = self.kx_base.to(x).repeat(c, 1, 1, 1)
        ky = self.ky_base.to(x).repeat(c, 1, 1, 1)  

        dx = F.conv2d(x_pad, kx, stride=1, padding=0, groups=c)
        dy = F.conv2d(x_pad, ky, stride=1, padding=0, groups=c)
        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        Args:
            pred, target : (N, C, H, W)
            kwargs:
              test_y_channel (bool): if True, compute on Y only. Default False
        """
        N = pred.size(0)
        loss_ps = pred.new_zeros(N)

        for s, sw in zip(self.scales, self.scale_weights):
            P = self._down(pred, s)
            T = self._down(target, s)

            dxp, dyp = self._grad(P)
            dxt, dyt = self._grad(T)

            # gradient magnitudes
            gp = torch.sqrt(dxp*dxp + dyp*dyp + self.eps)
            gt = torch.sqrt(dxt*dxt + dyt*dyt + self.eps)

            # edge-aware weights from target gradients:
            # normalize by per-sample mean to keep global scale stable
            gt_mean = gt.mean(dim=(1,2,3), keepdim=True)
            w_norm = (gt / (gt_mean + self.eps)) ** self.gamma
            # blend with uniform weight to control strength
            w_map = (1.0 - self.edge_strength) + self.edge_strength * w_norm
            # keep average ≈ 1 to avoid changing global loss scale too much
            w_map = w_map / (w_map.mean(dim=(1,2,3), keepdim=True) + self.eps)

            # magnitude term
            mag_diff = torch.abs(gp - gt) if self.mag_weight > 0 else 0.0

            # orientation term: 1 - cosine(∇p, ∇t)
            if self.ori_weight > 0:
                np_ = torch.sqrt(dxp*dxp + dyp*dyp + self.eps)
                nt_ = torch.sqrt(dxt*dxt + dyt*dyt + self.eps)
                cos = (dxp*dxt + dyp*dyt) / (np_ * nt_)
                ori_diff = 1.0 - cos.clamp(-1.0, 1.0)
            else:
                ori_diff = 0.0

            diff = self.mag_weight * mag_diff + self.ori_weight * ori_diff  # (N,C,h,w) or scalar 0
            diff = diff * w_map                                            # edge emphasis
            loss_ps += sw * diff.mean(dim=(1,2,3))

        if self.reduction == 'none':
            loss = loss_ps
        elif self.reduction == 'mean':
            loss = loss_ps.mean()
        else:
            loss = loss_ps.sum()

        return self.loss_weight * loss
