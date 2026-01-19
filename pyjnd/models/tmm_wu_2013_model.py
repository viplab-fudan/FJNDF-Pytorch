"""
Reproduction of the JND estimation algorithm presented in:

    Jinjian Wu, Guangming Shi, Weisi Lin, Anmin Liu, and Fei Qi,
    "Just Noticeable Difference Estimation for Images With Free-Energy Principle,"
    IEEE Transactions on Multimedia, vol. 15, no. 7, pp. 1705-1710, Nov. 2013.

This implementation is aligned to the authors' released MATLAB reference code
(e.g., demo_JND_estimation.m and the accompanying func_*.m files).

For full details, please refer to the original publication.
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pyjnd.models.base_model import SpatialEffectBase, register_spatial_effect


# -----------------------------------------------------------------------------
# MATLAB-compat helpers
# -----------------------------------------------------------------------------
def _gaussian_kernel2d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Equivalent to fspecial('gaussian', size, sigma) (normalized)."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size // 2)
    g1 = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    ker = g1[:, None] * g1[None, :]
    return ker


def _pad_symmetric_2d(x: torch.Tensor, pad: int) -> torch.Tensor:
    """
    Approximate MATLAB padarray(x, [pad pad], 'symmetric') for NCHW tensors.

    PyTorch does not provide 'symmetric' padding (edge-inclusive reflection),
    so we build it via explicit mirroring that repeats the border samples.
    """
    if pad <= 0:
        return x

    # Width
    left = torch.flip(x[..., :, :pad], dims=[-1])
    right = torch.flip(x[..., :, -pad:], dims=[-1])
    x = torch.cat([left, x, right], dim=-1)

    # Height
    top = torch.flip(x[..., :pad, :], dims=[-2])
    bottom = torch.flip(x[..., -pad:, :], dims=[-2])
    x = torch.cat([top, x, bottom], dim=-2)
    return x


def _imfilter_same(x: torch.Tensor, ker: torch.Tensor, pad: int, boundary: str) -> torch.Tensor:
    """
    imfilter(x, ker, 'same') with a boundary option.

    boundary:
      - 'replicate'  : replicate border pixels
      - 'zero'       : zero padding
    """
    if boundary == "replicate":
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        return F.conv2d(x, ker, padding=0)
    if boundary == "zero":
        return F.conv2d(x, ker, padding=pad)
    raise ValueError(f"Unsupported boundary: {boundary}")


# 5x5 background luminance kernel B / 32 (shared by several blocks)
_BG_KERNEL = torch.tensor(
    [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 2, 0, 2, 1],
        [1, 2, 2, 2, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=torch.float32,
) / 32.0


def _bg_lum(img: torch.Tensor) -> torch.Tensor:
    """func_bg_lum: floor(filter2(B, img) / 32) with zero padding."""
    ker = _BG_KERNEL.to(img.device).view(1, 1, 5, 5)
    return torch.floor(F.conv2d(img, ker, padding=2))


def _bg_adjust(bg_lum0: torch.Tensor, min_lum: float) -> torch.Tensor:
    """func_bg_adjust: adjust low luminance region (<=127)."""
    th = 127.0
    bg = bg_lum0
    scale = (th - float(min_lum)) / th
    return torch.where(bg <= th, torch.round(float(min_lum) + bg * scale), bg)


def bg_lum_jnd(img: torch.Tensor, min_lum: float = 32.0) -> torch.Tensor:
    """
    func_bg_lum_jnd: luminance adaptation JND (LA).
    img: (1,1,H,W) float32 in [0,255]
    """
    th = 127.0
    T0 = 17.0
    gamma = 3.0 / 128.0

    bg0 = _bg_lum(img)
    bg = _bg_adjust(bg0, min_lum)

    return torch.where(
        bg <= th,
        T0 * (1.0 - torch.sqrt(bg / th)) + 3.0,
        gamma * (bg - th) + 3.0,
    )


def statistic_value_std(img: torch.Tensor, r: int, boundary: str = "replicate") -> torch.Tensor:
    """
    func_statistic_value: local standard deviation in a (2r+1)x(2r+1) mean window.
    img: (1,1,H,W)
    """
    k = 2 * r + 1
    ker = torch.ones((k, k), dtype=torch.float32, device=img.device) / float(k * k)
    ker = ker.view(1, 1, k, k)

    mean = _imfilter_same(img, ker, pad=r, boundary=boundary)
    mean_sq = mean * mean
    sq_mean = _imfilter_same(img * img, ker, pad=r, boundary=boundary)
    var = torch.clamp(sq_mean - mean_sq, min=0.0)
    return torch.sqrt(var)


def luminance_diff(img: torch.Tensor) -> torch.Tensor:
    """
    func_luminance_diff: max abs response among four 5x5 directional filters (filter2 / 16).
    img: (1,1,H,W)
    """
    device = img.device

    G1 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 3, 8, 3, 1],
            [0, 0, 0, 0, 0],
            [-1, -3, -8, -3, -1],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
        device=device,
    ) / 16.0

    G2 = torch.tensor(
        [
            [0, 0, 1, 0, 0],
            [0, 8, 3, 0, 0],
            [1, 3, 0, -3, -1],
            [0, 0, -3, -8, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=torch.float32,
        device=device,
    ) / 16.0

    G3 = torch.tensor(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 3, 8, 0],
            [-1, -3, 0, 3, 1],
            [0, -8, -3, 0, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=torch.float32,
        device=device,
    ) / 16.0

    G4 = torch.tensor(
        [
            [0, 1, 0, -1, 0],
            [0, 3, 0, -3, 0],
            [0, 8, 0, -8, 0],
            [0, 3, 0, -3, 0],
            [0, 1, 0, -1, 0],
        ],
        dtype=torch.float32,
        device=device,
    ) / 16.0

    kernels = [G1, G2, G3, G4]
    grads = [F.conv2d(img, k.view(1, 1, 5, 5), padding=2).abs() for k in kernels]
    return torch.max(torch.stack(grads, dim=0), dim=0)[0]


def contrast_mask_jnd(img: torch.Tensor) -> torch.Tensor:
    """
    func_contrast_mask_jnd: contrast masking JND (CM).
    img: (1,1,H,W)
    """
    lum_diff = luminance_diff(img)

    thre = 80.0
    lum_diff_ = thre * torch.log10(1.0 + lum_diff / thre) / math.log10(4.0)

    bg = _bg_lum(img)

    LANDA = 0.5
    alpha = 0.0001 * bg + 0.115
    beta = LANDA - 0.01 * bg
    return torch.abs(lum_diff_ * alpha + beta)


def superedge_map(img0_u8: np.ndarray, thre: float = 0.7) -> np.ndarray:
    """
    func_edge: Canny -> dilation with disk radius 2 -> (1 - 0.8*edge) -> Gaussian blur.
    Returns float32 array in the same size as img0.
    """
    # MATLAB edge(I,'canny',thre): treat scalar threshold as high threshold; low ~= 0.4*high.
    high = float(thre) * 255.0
    low = 0.4 * high

    edges = cv2.Canny(img0_u8, threshold1=int(low), threshold2=int(high))

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # disk radius 2
    dil = cv2.dilate(edges, se)

    sup = 1.0 - 0.8 * (dil.astype(np.float32) / 255.0)

    # filter2 with fspecial('gaussian',5,0.8) (zero padding)
    ker = _gaussian_kernel2d(5, 0.8, device=torch.device("cpu")).cpu().numpy().astype(np.float32)
    sup = cv2.filter2D(sup, ddepth=-1, kernel=ker, borderType=cv2.BORDER_CONSTANT)
    return sup.astype(np.float32)


def disorder_jnd(img0: torch.Tensor, residual: torch.Tensor, r: int = 3, var_thr: float = 10.0) -> torch.Tensor:
    """
    func_disorder_jnd:
      - residual smoothing with Gaussian (2r+1, sigma=(2r+1)/3)
      - low-variance gating (variance window fixed to 3 in MATLAB demo)
      - super-edge weighting
    img0/residual: (1,1,H,W)
    """
    device = img0.device

    # vari_map = func_statistic_value(img0, 3)
    vari_map = statistic_value_std(img0, r=3, boundary="replicate")

    # img_mean = imfilter(residual, gaussian_ker)
    gsize = 2 * r + 1
    sigma = gsize / 3.0
    gker = _gaussian_kernel2d(gsize, sigma, device=device).view(1, 1, gsize, gsize)
    res_mean = _imfilter_same(residual, gker, pad=r, boundary="replicate")

    res_min = torch.minimum(res_mean, residual)

    jnd_dis = residual.clone()
    jnd_dis = torch.where(vari_map < float(var_thr), res_min, jnd_dis)

    # superedge from original image (CPU / OpenCV), then multiply
    img0_u8 = img0.detach().clamp(0, 255).round()[0, 0].to("cpu").numpy().astype(np.uint8)
    sup_np = superedge_map(img0_u8, thre=0.7)
    sup = torch.from_numpy(sup_np).to(device=device, dtype=torch.float32).view(1, 1, *sup_np.shape)

    return jnd_dis * sup


# -----------------------------------------------------------------------------
# AR prediction (func_ar_predict_decomp + func_ar_nl)
# -----------------------------------------------------------------------------
class ARPredictor(nn.Module):
    def __init__(self, min_thr: float = 5.0, r: int = 3, R: int = 10):
        super().__init__()
        self.min_thr = float(min_thr)
        self.r = int(r)
        self.R = int(R)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Y: (1,1,H,W) float32 luminance in [0,255]
        Returns: (1,1,H,W) float32 (integer-valued, rounded like MATLAB uint8 cast)
        """
        if Y.ndim != 4 or Y.shape[0] != 1 or Y.shape[1] != 1:
            raise ValueError("Y must be shaped as (1,1,H,W)")

        r, R = self.r, self.R
        H, W = Y.shape[-2], Y.shape[-1]

        # min_sigma = func_bg_lum_jnd(img0); min_sigma(min_sigma<min_thr)=min_thr;
        min_sigma = bg_lum_jnd(Y, min_lum=32.0)
        min_sigma = torch.clamp(min_sigma, min=self.min_thr)

        # vari = func_statistic_value(img_in, r)  (std, not variance)
        vari = statistic_value_std(Y, r=r, boundary="replicate")

        # sigma_value = min_sigma.^2; sigma_value(vari>min_sigma) = (...).^2
        sigma_value = min_sigma * min_sigma
        mask = vari > min_sigma
        vari_safe = torch.clamp(vari, min=1e-12)
        adj = (min_sigma * torch.sqrt(min_sigma / vari_safe)) ** 2
        sigma_value = torch.where(mask, adj, sigma_value)

        # padarray(img_in, [R+r,R+r], 'symmetric')
        pad = R + r
        Y_pad = _pad_symmetric_2d(Y, pad=pad)

        img_pad = Y_pad[:, :, R : R + H + 2 * r, R : R + W + 2 * r]

        ker = torch.ones((2 * r + 1, 2 * r + 1), dtype=torch.float32, device=Y.device) / float((2 * r + 1) ** 2)
        ker = ker.view(1, 1, 2 * r + 1, 2 * r + 1)

        img_reco = torch.zeros_like(Y)
        weight_mat = torch.zeros_like(Y)
        max_weight = torch.zeros_like(Y)

        for u in range(-R, R + 1):
            for v in range(-R, R + 1):
                if u == 0 and v == 0:
                    continue

                mv = Y_pad[:, :, R + u : R + u + H + 2 * r, R + v : R + v + W + 2 * r]
                mat_dif = (img_pad - mv) ** 2
                sum_val = F.conv2d(mat_dif, ker, padding=0)
                sim = torch.exp(-sum_val / sigma_value)

                patch = mv[:, :, r : r + H, r : r + W]
                img_reco += patch * sim
                weight_mat += sim
                max_weight = torch.maximum(max_weight, sim)

        img_recon = img_reco + max_weight * Y
        weight = weight_mat + max_weight
        Y_pred = img_recon / torch.clamp(weight, min=1e-12)

        # MATLAB: uint8(img_recon ./ weight_mat)
        Y_pred = torch.clamp(torch.round(Y_pred), 0.0, 255.0)
        return Y_pred


# -----------------------------------------------------------------------------
# Effect blocks (kept compatible with pyjnd's SpatialEffectBase interface)
# -----------------------------------------------------------------------------
@register_spatial_effect("LA", "TMMwu2013")
class TMMwu2013_LA(SpatialEffectBase):
    """Luminance adaptation (func_bg_lum_jnd) computed on the AR-predicted image."""

    def __init__(self, min_sigma: float = 8.0, min_lum: float = 32.0, r: int = 3, R: int = 10):
        super().__init__(min_sigma=min_sigma, min_lum=min_lum, r=r, R=R)
        self.predictor = ARPredictor(min_thr=min_sigma, r=r, R=R)
        self.min_lum = float(min_lum)

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        Y = yuv[0:1].unsqueeze(0).float()
        Yp = self.predictor(Y)
        return bg_lum_jnd(Yp, min_lum=self.min_lum).squeeze(0).squeeze(0)


@register_spatial_effect("CM", "TMMwu2013")
class TMMwu2013_CM(SpatialEffectBase):
    """Contrast masking (func_contrast_mask_jnd) computed on the AR-predicted image."""

    def __init__(self, min_sigma: float = 8.0, r: int = 3, R: int = 10):
        super().__init__(min_sigma=min_sigma, r=r, R=R)
        self.predictor = ARPredictor(min_thr=min_sigma, r=r, R=R)

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        Y = yuv[0:1].unsqueeze(0).float()
        Yp = self.predictor(Y)
        return contrast_mask_jnd(Yp).squeeze(0).squeeze(0)


@register_spatial_effect("PM", "TMMwu2013")
class TMMwu2013_PM(SpatialEffectBase):
    """Disorderly concealment effect (func_disorder_jnd) computed from AR residual."""

    def __init__(self, min_sigma: float = 8.0, r: int = 3, R: int = 10, k: float = 1.0):
        super().__init__(min_sigma=min_sigma, r=r, R=R, k=k)
        self.predictor = ARPredictor(min_thr=min_sigma, r=r, R=R)
        self.k = float(k)
        self.r = int(r)

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        Y = yuv[0:1].unsqueeze(0).float()
        Yp = self.predictor(Y)
        res = (Y - Yp).abs()
        jnd_dis = disorder_jnd(Y, res, r=self.r, var_thr=10.0)
        return (self.k * jnd_dis).squeeze(0).squeeze(0)