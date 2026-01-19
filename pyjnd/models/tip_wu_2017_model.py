"""
Reproduction of the algorithm presented in:

    Jinjian Wu, Leida Li, Weisheng Dong, Guangming Shi, Weisi Lin, and C.-C. Jay Kuo,
    "Enhanced Just Noticeable Difference Model for Images With Pattern Complexity,"
    IEEE Transactions on Image Processing, 2017.

This file implements the JND-based model as described in the above paper.
For full details, please refer to the original publication.
"""


import math
import cv2
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

try:
    from scipy import ndimage as ndi
    from skimage.feature import canny as sk_canny
    from skimage.morphology import binary_dilation, disk
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
from pyjnd.models.base_model import SpatialEffectBase, register_spatial_effect

# Luminance Adaptation (LA)
@register_spatial_effect('LA', 'TIPwu2017')
class TIPwu2017_LA(SpatialEffectBase):
    """Luminance adaptation (Wu et al., TIP 2017)."""
    def __init__(self, 
                 T0: float = 17.0, 
                 slope: float = 3.0/128.0, 
                 bg_threshold: float = 127.0, 
                 min_lum: float = 32.0
                ):
        super().__init__(T0=T0, slope=slope, bg_threshold=bg_threshold, min_lum=min_lum)
        self.T0 = T0
        self.slope = slope
        self.th  = bg_threshold
        self.ml  = min_lum
        # 5x5 low-pass kernel B/32 (same as Chou '95)
        B = torch.tensor([
          [1,1,1,1,1],
          [1,2,2,2,1],
          [1,2,0,2,1],
          [1,2,2,2,1],
          [1,1,1,1,1],
        ], dtype=torch.float32) / 32.0
        self.register_buffer('bg_kernel', B.view(1,1,5,5))

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        # Y channel in range [0,255]
        Y = yuv[0:1].unsqueeze(0)              # (1,1,H,W)

        # Background luminance (5x5 low-pass, floor)
        bg = torch.floor(F.conv2d(Y, self.bg_kernel, padding=2).squeeze(0).squeeze(0))

        # 3) adjust bg according to wu:when bg0 <= threshold, linearly map to [min_lum, threshold]
        mask     = (bg <= self.th)
        scale    = (self.th - self.ml) / self.th
        bg       = torch.where(mask,
                               torch.round(self.ml + bg * scale),
                               bg)

        # 3) NAMM's LA formula (Eq.6–8 in Wu et al.)
        T0, slp, th = self.T0, self.slope, self.th
        f2 = torch.where(bg <= th,
                         T0 * (1 - torch.sqrt(bg / th)) + 3.0,
                         slp * (bg - th) + 3.0)
        return f2



# Luminance Contrast Masking (CM)
@register_spatial_effect('CM', 'TIPwu2017')
class TIPwu2017_CM(SpatialEffectBase):
    """Luminance contrast masking (Wu et al., TIP 2017)."""
    def __init__(self,
                 R: int = 2,
                 a1: float = 0.115 * 16,
                 a2: float = 26.0):
        super().__init__(R=R, a1=a1, a2=a2)
        ksize = 2*R + 1
        ker = torch.ones((ksize, ksize), dtype=torch.float32) / (ksize**2)
        self.register_buffer('stat_ker', ker.view(1,1,ksize,ksize))
        self.R = R
        self.a1 = a1
        self.a2 = a2

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        Y = yuv[0:1].unsqueeze(0).float()             # (1,1,H,W)
        # local variance
        mean    = F.conv2d(Y, self.stat_ker, padding=self.R)
        mean_sq = F.conv2d(Y*Y, self.stat_ker, padding=self.R)
        var     = mean_sq - mean*mean
        var     = torch.clamp(var, min=0.0)
        # valid mask to zero out borders
        _,_,H,W = Y.shape
        vm = torch.zeros_like(var)
        vm[:,:,self.R:H-self.R, self.R:W-self.R] = 1.0
        var = var * vm
        Lc = torch.sqrt(var).squeeze(0).squeeze(0)    # (H,W)
        num = self.a1 * (Lc**2.4)
        den = (Lc**2 + self.a2**2)
        return num / den


# Pattern & Edge Protection (PM)
@register_spatial_effect('PM', 'TIPwu2017')
class TIPwu2017_PM(SpatialEffectBase):
    """Pattern masking with edge protection (Wu et al., TIP 2017)."""
    def __init__(self,
                 R: int = 2,
                 r: int = 3,
                 otr: int = 6,
                 a3: float = 0.3,
                 a4: float = 2.7,
                 a5: float = 1.0):
        super().__init__(R=R, r=r, otr=otr, a3=a3, a4=a4, a5=a5)
        # for computing L_c
        ksize = 2*R + 1
        ker = torch.ones((ksize, ksize), dtype=torch.float32) / (ksize**2)
        self.register_buffer('stat_ker', ker.view(1,1,ksize,ksize))
        self.R  = R
        self.r  = r
        self.otr= otr
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        # Gaussian smoothing for the pattern-complexity map (MATLAB: fspecial('gaussian', 3, 1))
        g_size = int(r)  # MATLAB uses r=3 as the kernel size (3x3), not a radius.
        sigma = 1.0
        coords = torch.arange(g_size, dtype=torch.float32) - (g_size // 2)
        g1 = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        g1 = g1 / g1.sum()
        g2 = torch.matmul(g1.unsqueeze(1), g1.unsqueeze(0))
        self.register_buffer('cmlx_smooth', g2.view(1, 1, g_size, g_size))
        self.cmlx_pad = g_size // 2
        # Gaussian for edge-protect smoothing
        size2, sig2 = 5, 0.8
        c2 = torch.arange(size2, dtype=torch.float32) - (size2//2)
        h1 = torch.exp(-(c2**2)/(2*sig2*sig2))
        h1 = h1 / h1.sum()
        h2 = torch.matmul(h1.unsqueeze(1), h1.unsqueeze(0))
        self.register_buffer('edge_gauss', h2.view(1,1,size2,size2))

        # Directional gradient kernels (5x5) for edge-height estimation
        G1 = torch.tensor([[0,0,0,0,0],
                           [1,3,8,3,1],
                           [0,0,0,0,0],
                           [-1,-3,-8,-3,-1],
                           [0,0,0,0,0]], dtype=torch.float32)/16
        G2 = torch.tensor([[0,0,1,0,0],
                           [0,8,3,0,0],
                           [1,3,0,-3,-1],
                           [0,0,-3,-8,0],
                           [0,0,-1,0,0]], dtype=torch.float32)/16
        G3 = torch.tensor([[0,0,1,0,0],
                           [0,0,3,8,0],
                           [-1,-3,0,3,1],
                           [0,-8,-3,0,0],
                           [0,0,-1,0,0]], dtype=torch.float32)/16
        G4 = torch.tensor([[0,1,0,-1,0],
                           [0,3,0,-3,0],
                           [0,8,0,-8,0],
                           [0,3,0,-3,0],
                           [0,1,0,-1,0]], dtype=torch.float32)/16
        self.register_buffer('G1', G1.view(1,1,5,5))
        self.register_buffer('G2', G2.view(1,1,5,5))
        self.register_buffer('G3', G3.view(1,1,5,5))
        self.register_buffer('G4', G4.view(1,1,5,5))

    def _compute_complexity(self, I: np.ndarray) -> np.ndarray:
        """
        Replicates func_ori_cmlx_compute + func_cmlx_num_compute.
        I: HxW numpy float image
        returns: HxW complexity map
        """
        r, nb, otr = 1, 8, self.otr
        # --- orientation samples around circle ---
        angles = np.arange(nb) * (2*np.pi/nb)
        sps = np.stack([[-r*np.sin(a), r*np.cos(a)] for a in angles], axis=0)
        # pad
        Ipad = np.pad(I, pad_width=r, mode='symmetric')
        # gradients
        kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)/3
        ky = kx.T
        Gx = cv2.filter2D(Ipad, -1, kx)
        Gy = cv2.filter2D(Ipad, -1, ky)
        Cimg = np.sqrt(Gx*Gx + Gy*Gy)
        Cvimg = (Cimg >= 5.0).astype(np.uint8)
        # angle map
        Oimg = np.round(np.degrees(np.arctan2(Gy, Gx)))
        Oimg[Oimg > 90]  -= 180
        Oimg[Oimg < -90] += 180
        Oimg += 90
        Oimg[Cvimg==0] = 180 + 2*otr
        # crop center
        Oimgc  = Oimg[r:-r, r:-r]
        Cvimgc = Cvimg[r:-r, r:-r]
        # quantize
        Onorm  = np.round(Oimg / (2*otr)).astype(int)
        Oc_norm= np.round(Oimgc/ (2*otr)).astype(int)
        onum = int(round(180.0/(2*otr))) + 1
        bins = onum + 1
        Hc, Wc = Oimgc.shape
        ssr = np.zeros((Hc, Wc, bins), dtype=np.int32)
        # central
        for b in range(bins):
            ssr[:,:,b] += (Oc_norm == b)
        # neighbors
        for dx,dy in sps:
            ix = int(round(r + dx))
            iy = int(round(r + dy))
            patch = Onorm[ix:ix+Hc, iy:iy+Wc]
            for b in range(bins):
                ssr[:,:,b] += (patch == b)
        # complexity
        ssr_nz = (ssr != 0)
        cmlx   = ssr_nz.sum(axis=2)
        # enforce plain and borders =1
        cmlx[Cvimgc==0] = 1
        cmlx[:r,:]      = 1
        cmlx[-r:,:]     = 1
        cmlx[:,:r]      = 1
        cmlx[:,-r:]     = 1
        return cmlx

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        Y = yuv[0:1].unsqueeze(0).float()             # (1,1,H,W)
        _,_,H,W = Y.shape
        # --- 1) compute L_c (as in CM) ---
        mean    = F.conv2d(Y, self.stat_ker, padding=self.R)
        mean_sq = F.conv2d(Y*Y, self.stat_ker, padding=self.R)
        var     = torch.clamp(mean_sq - mean*mean, min=0.0)
        vm      = torch.zeros_like(var)
        vm[:,:,self.R:H-self.R, self.R:W-self.R] = 1.0
        Lc = torch.sqrt((var * vm)).squeeze(0).squeeze(0)  # (H,W)
        # --- 2) compute P_c ---
        I = Y[0,0].cpu().numpy()
        cmlx_map = self._compute_complexity(I)             # HxW numpy
        # smooth
        Pc = torch.from_numpy(cmlx_map).to(Y.device).unsqueeze(0).unsqueeze(0).float()
        Pc = F.conv2d(Pc, self.cmlx_smooth, padding=self.cmlx_pad).squeeze(0).squeeze(0)
        # complexity transducer
        Ct = (self.a3 * (Pc**self.a4)) / (Pc**2 + self.a5**2)
        # pattern masking
        jnd_pm = Lc * Ct                                    # (H,W)
        # --- 3) edge protection (MATLAB: func_edge_protect) ---
        # Edge height (MATLAB: func_edge_height).
        e1 = F.conv2d(Y, self.G1, padding=2).abs()
        e2 = F.conv2d(Y, self.G2, padding=2).abs()
        e3 = F.conv2d(Y, self.G3, padding=2).abs()
        e4 = F.conv2d(Y, self.G4, padding=2).abs()
        edge_height = torch.max(torch.max(e1, e2), torch.max(e3, e4)).squeeze(0).squeeze(0)

        # MATLAB keeps a 2-pixel boundary at zero: edge_height(3:end-2,3:end-2)=..., others=0.
        edge_height = edge_height.clone()
        edge_height[:2, :] = 0
        edge_height[-2:, :] = 0
        edge_height[:, :2] = 0
        edge_height[:, -2:] = 0

        max_val = float(edge_height.max().detach().cpu().item())
        max_val = max(max_val, 1e-6)
        edge_threshold = min(60.0 / max_val, 0.8)

        # MATLAB: edge(img,'canny',edge_threshold), scalar implies low=0.4*high.
        # We approximate MATLAB's behavior using a gradient-magnitude based Canny.
        Im = Y[0, 0].detach().cpu().numpy().astype(np.float64)

        if _HAS_SKIMAGE:
            sm = ndi.gaussian_filter(Im, sigma=1.0)
            gx = ndi.sobel(sm, axis=1, mode="constant")
            gy = ndi.sobel(sm, axis=0, mode="constant")
            mag = np.hypot(gx, gy)
            mag_max = float(mag.max()) if mag.size else 0.0
            mag_max = max(mag_max, 1e-12)

            high = edge_threshold * mag_max
            low = 0.4 * high
            edge_region = sk_canny(Im, sigma=1.0, low_threshold=low, high_threshold=high)

            img_edge = binary_dilation(edge_region, disk(self.r)).astype(np.float32)
        else:
            # Fallback to OpenCV when SciPy / scikit-image is unavailable.
            Im8 = np.clip(Im, 0, 255).astype(np.uint8)
            high = edge_threshold * 255.0
            low = 0.4 * high
            edge_region = cv2.Canny(Im8, int(low), int(high)) > 0

            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.r + 1, 2 * self.r + 1))
            img_edge = cv2.dilate(edge_region.astype(np.uint8), se).astype(np.float32)

        # MATLAB: h = fspecial('gaussian',5,0.8); img_edge = filter2(h,img_edge)
        h = self.edge_gauss[0, 0].detach().cpu().numpy().astype(np.float32)
        img_edge_blur = cv2.filter2D(img_edge, -1, h, borderType=cv2.BORDER_CONSTANT)

        edge_protect = 1.0 - img_edge_blur
        edge_protect = np.clip(edge_protect, 0.0, 1.0)
        edge_protect_t = torch.from_numpy(edge_protect).to(Y.device).float()

        # final jnd_PM_p
        return (jnd_pm * edge_protect_t).clamp(min=0.0)
