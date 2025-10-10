# dct_loss.py
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


def adaptive_coefficients(coefficients):
    zero = torch.tensor([0], dtype=torch.float32).cuda()

    mean_coefficients_value = torch.mean(coefficients.clone(), 3)
    mean_coefficients_value = mean_coefficients_value.reshape(coefficients.shape[0], coefficients.shape[1], coefficients.shape[2], 1)

    mean_coefficients = coefficients.clone()
    mean_coefficients[..., :] = mean_coefficients_value

    selected_coefficients = torch.where(coefficients < mean_coefficients, zero, coefficients)

    return selected_coefficients


def upper_band_zigzag(coefficients):
    assert len(coefficients.shape) in (4, 5)

    zigzag_indices = torch.tensor([
      #  0,  1,  8, 16,  9,  2,  3, 10,
      # 17, 24, 32, 25, 18, 11,  4,  5,
      # 12, 19, 26, 33, 40, 48, 41, 34,
      # 27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63 
    ]).long() 

    if len(coefficients.shape) == 4:
        c = coefficients.unsqueeze(0)
    else:
        c = coefficients

    c = c.view(c.shape[0], c.shape[1], c.shape[2], 64)
    c = c[..., zigzag_indices]
    c = adaptive_coefficients(c)

    if len(coefficients.shape) == 3:
        c = c.squeeze(0)
    return c


def upper_band_zigzag_16(coefficients):
    """
    For 16x16 DCT zigzag
    """
    assert len(coefficients.shape) in (4, 5)

    zigzag_indices = torch.tensor([
      # 0, 1, 16, 32, 17, 2, 3, 18, 33, 48, 64, 49, 34, 19, 4, 5, 
      # 20, 35, 50, 65, 80, 96, 81, 66, 51, 36, 21, 6, 7, 22, 37, 52, 
      # 67, 82, 97, 112, 128, 113, 98, 83, 68, 53, 38, 23, 8, 9, 24, 39, 
      # 54, 69, 84, 99, 114, 129, 144, 160, 145, 130, 115, 100, 85, 70, 55, 40, 
      # 25, 10, 11, 26, 41, 56, 71, 86, 101, 116, 131, 146, 161, 176, 192, 177, 
      # 162, 147, 132, 117, 102, 87, 72, 57, 42, 27, 12, 13, 28, 43, 58, 73, 
      # 88, 103, 118, 133, 148, 163, 178, 193, 208, 224, 209, 194, 179, 164, 149, 134, 
      # 119, 104, 89, 74, 59, 44, 29, 14, 15, 30, 45, 60, 75, 90, 105, 120, 
        135, 150, 165, 180, 195, 210, 225, 240, 241, 226, 211, 196, 181, 166, 151, 136, 
        121, 106, 91, 76, 61, 46, 31, 47, 62, 77, 92, 107, 122, 137, 152, 167, 
        182, 197, 212, 227, 242, 243, 228, 213, 198, 183, 168, 153, 138, 123, 108, 93,  
        78, 63, 79, 94, 109, 124, 139, 154, 169, 184, 199, 214, 229, 244, 245, 230, 
        215, 200, 185, 170, 155, 140, 125, 110, 95, 111, 126, 141, 156, 171, 186, 201, 
        216, 231, 246, 247, 232, 217, 202, 187, 172, 157, 142, 127, 143, 158, 173, 188, 
        203, 218, 233, 248, 249, 234, 219, 204, 189, 174, 159, 175, 190, 205, 220, 235,  
        250, 251, 236, 221, 206, 191, 207, 222, 237, 252, 253, 238, 223, 239, 254, 255, 
    ]).long() 

    if len(coefficients.shape) == 4:
        c = coefficients.unsqueeze(0)
    else:
        c = coefficients

    c = c.view(c.shape[0], c.shape[1], c.shape[2], 256)
    c = c[..., zigzag_indices]

    c = adaptive_coefficients(c)

    if len(coefficients.shape) == 3:
        c = c.squeeze(0)
    return c


@LOSS_REGISTRY.register()
class Dct8Loss(nn.Module):
    """Adaptive DCT loss (block size = 8)"""
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Dct8Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target=None, weight=None, **kwargs):
        """
        Args:
            pred (Tensor[N, C, H, W]): Network predicted feature map / image
            target, weight (ignored): Kept for consistent signature with L1Loss, but not used in DCT loss
        Returns:
            Tensor: Scalar loss
        """
        # Block → Calculate 8×8 DCT coefficients → Take high frequency band → L2 energy
        blocks = blockify(pred, 8)
        coeffs = block_dct(blocks)
        upper = upper_band_zigzag(coeffs)
        loss = torch.mean(torch.sum(torch.pow(upper, 2), dim=3))

        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class Dct16Loss(nn.Module):
    """Adaptive DCT loss (block size = 16)"""
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Dct16Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target=None, weight=None, **kwargs):
        """
        Args:
            pred (Tensor[N, C, H, W]): Network predicted feature map / image
            target, weight (ignored): Keep signature, not actually used
        Returns:
            Tensor: Scalar loss
        """
        # Block → Calculate 16×16 DCT coefficients → Take high frequency band → L2 energy
        blocks = blockify(pred, 16)
        coeffs = block_dct(blocks)
        upper = upper_band_zigzag_16(coeffs)
        loss = torch.mean(torch.sum(torch.pow(upper, 2), dim=3))

        return self.loss_weight * loss