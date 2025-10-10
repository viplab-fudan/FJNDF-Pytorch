"""
Reproduction of the algorithm presented in:

    Kang, Byeongkeun and Kim, Wonha,
    “Human perception-oriented enhancement and smoothing for perceptual video coding,”
    IEEE Transactions on Broadcasting, vol. 69, no. 3, pp. 767-778, 2023.

This file implements the human perception-oriented enhancement and smoothing techniques as described in the above paper.
For full details, please refer to the original publication.
"""

import math
import torch
import cv2
import numpy as np

from pyjnd.models.base_model import FrequencyEffectBase, register_frequency_effect

@register_frequency_effect('CSF', 'TOBkang2023')
class TOB_Kang_2023_CSF(FrequencyEffectBase):
    """
    J_base (Spatial CSF) part in TOB Kang & Kim 2023:
    - Same as TIP_Bae_2016_CSF
    Returns (H,W) J_base map.
    """
    def __init__(self, s, dct_size, r_vh, screen_h):
        super().__init__(s=s, dct_size=dct_size,
                         r_vh=r_vh,
                         screen_h=screen_h)
        self.s = s
        self.dct_size = dct_size
        self.r_vh = r_vh
        self.screen_h = screen_h

        # Pre-calculate spatial frequency f and direction angle θ (equations 3,4)
        self.freq = {}
        self.theta = math.atan(1.0 / (2 * self.r_vh * self.screen_h)) * (180.0 / math.pi)
        for u in range(dct_size):
            for v in range(dct_size):
                if u == 0 and v == 0:
                    f = 0.0
                else:
                    f = math.hypot(u, v) / (2 * self.dct_size * self.theta)
                self.freq[(u, v)] = f

        # Fitting coefficients a2, a1, a0 for vertical/diagonal directions in equation (7)
        self.a_diag = [0.0293, -0.1382, 1.75]
        self.a_vert = [0.0238, -0.1771, 1.75]

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        Y = yuv[0]
        H, W = Y.shape
        jnd = torch.zeros_like(Y)
        for i in range(0, H, self.dct_size):
            for j in range(0, W, self.dct_size):
                for u in range(self.dct_size):
                    for v in range(self.dct_size):
                        # 1) Get frequency
                        f_uv  = self.freq[(u, v)]
                        f_u0  = self.freq[(u, 0)]
                        f_0v  = self.freq[(0, v)]

                        # 2) Calculate J_v, J_d (vertical/diagonal polynomials)
                        Jv = self.a_vert[0]*f_uv*f_uv + self.a_vert[1]*f_uv + self.a_vert[2]
                        Jd = self.a_diag[0]*f_uv*f_uv + self.a_diag[1]*f_uv + self.a_diag[2]

                        # 3) Calculate sin(phi)^2 directly using simplified form of equation 5
                        #    sin(phi) = 2*f_u0*f_0v / (f_uv^2)
                        if (u == 0 and v ==0):
                            sin_phi = sin2 = 0
                        else:
                            sin_phi = 2.0 * f_u0 * f_0v / (f_uv * f_uv)
                            sin2   = sin_phi * sin_phi

                        # 4) Calculate J_base according to equation 8
                        Jbase = (Jd - Jv) * sin2 + Jv

                        jnd[i+u, j+v] = Jbase * self.s * self.dct_size
        return jnd

@register_frequency_effect('LA', 'TOBkang2023')
class TOB_Kang_2023_LA(FrequencyEffectBase):
    """
    J_la part in TOB Kang & Kim 2023:
    - Same as TIP_Bae_2016_LA
    Returns (H,W) J_LA map.
    """
    def __init__(self, threshold_split, dct_size, r_vh, screen_h):
        super().__init__(threshold_split=threshold_split,
                         dct_size=dct_size,
                         r_vh=r_vh,
                         screen_h=screen_h)
        # threshold_split: boundary luminance point
        self.threshold = threshold_split
        self.dct_size = dct_size
        self.r_vh = r_vh
        self.screen_h = screen_h

        self.freq = {}
        self.theta = math.atan(1.0 / (2 * self.r_vh * self.screen_h)) * (180.0 / math.pi)
        for u in range(dct_size):
            for v in range(dct_size):
                if u == 0 and v == 0:
                    f = 0.0
                else:
                    f = math.hypot(u, v) / (2 * self.dct_size * self.theta)
                self.freq[(u, v)] = f

        # Fitting coefficients q1, q2, q3 for low/high luminance in equation (7)
        self.q_01 = [2.468e-4, 4.466e-3, 1.14]
        self.q_09 = [1.230e-3, 1.433e-2, 1.34]

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        Y = yuv[0]
        H, W = Y.shape
        m = torch.zeros_like(Y)
        for i in range(0, H, self.dct_size):
            for j in range(0, W, self.dct_size):
                block = Y[i:i+self.dct_size, j:j+self.dct_size]
                bg = float(block.mean().item()) / 255.0 # for 8 bit input
                for u in range(self.dct_size):
                    for v in range(self.dct_size):
                        f = self.freq[(u,v)]
                        # Calculate by segments according to bg (equations 10–12)
                        M01 = self.q_01[0]*f*f + self.q_01[1]*f + self.q_01[2]
                        M09 = self.q_09[0]*f*f + self.q_09[1]*f + self.q_09[2]
                        if bg <= self.threshold:
                            delta = abs((bg - self.threshold) / 0.2)
                            term  = 1 + (M01 - 1) * (delta ** 0.8)
                        else:
                            delta = abs((bg - self.threshold) / 0.6)
                            term  = 1 + (M09 - 1) * (delta ** 0.6)
                        m[i+u, j+v] = term
        return m

@register_frequency_effect('CM', 'TOBkang2023')
class TOB_Kang_2023_CM(FrequencyEffectBase):
    """
    TOB Kang & Kim 2023 J_cm
    - Same as TIP_Bae_2016_CM
    """
    # -------------------- Initialization -------------------- #
    def __init__(
        self,
        threshold_split: float,   # Keep consistent with FrequencyEffectBase (unused)
        dct_size: int,
        r_vh: float,
        screen_h: int
    ):
        super().__init__(threshold_split=threshold_split,
                         dct_size=dct_size,
                         r_vh=r_vh,
                         screen_h=screen_h)
        self.threshold_split = threshold_split
        self.dct_size   = dct_size
        self.r_vh       = r_vh
        self.screen_h   = screen_h
        self.canny_low  = 50
        self.canny_high = 150

        # Pre-calculate ω (cycles/deg) —— Same as LA class
        theta = math.atan(1.0 / (2 * self.r_vh * self.screen_h)) * (180.0 / math.pi)
        self.freq = {(u, v): (0.0 if (u == 0 and v == 0)
                              else math.hypot(u, v) / (2 * dct_size * theta))
                     for u in range(dct_size) for v in range(dct_size)}

    # -------------------- Helper functions -------------------- #
    @staticmethod
    def _G(omega: float) -> float:
        """Frequency response G(ω) (equation 17)."""
        return 1.558 * math.exp(-((omega - 4.12) / 7.16) ** 2) + 2.817

    def _canny_edges(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (H,W) uint8 / float Tensor
        Returns binary edge map (H,W) float32 (0/1)
        OpenCV runs on CPU; if y is on GPU, move to CPU first.
        """
        if y.dtype != torch.uint8:
            y_8 = (y.clamp(0, 255).round()).to(torch.uint8)
        else:
            y_8 = y
        y_cpu = y_8.detach().to('cpu').numpy()           # (H,W) uint8
        edges = cv2.Canny(y_cpu, self.canny_low, self.canny_high)  # 0/255
        edges = torch.from_numpy(edges).float().to(y.device) / 255.0
        return edges                                     # (H,W) 0/1 float

    # -------------------- Main interface -------------------- #
    @torch.no_grad()
    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Input: yuv (C,H,W), only use Y(0); supports uint8 or float32
        Output: (H,W) M_CM Tensor (float)
        """
        Y = yuv[0]
        H, W = Y.shape
        edge_map = self._canny_edges(Y)                  # (H,W) 0/1
        m = torch.ones_like(Y, dtype=torch.float32)      # Output map

        for i in range(0, H, self.dct_size):
            for j in range(0, W, self.dct_size):
                # —— μ_e : edge density within block
                mu_e = edge_map[i:i+self.dct_size, j:j+self.dct_size].mean().item()

                # —— α : binary segmentation according to μ_e
                alpha = 2.0 if mu_e <= self.threshold_split else 1.0

                # —— Fill each DCT coefficient corresponding pixel position
                for u in range(self.dct_size):
                    for v in range(self.dct_size):
                        omega = self.freq[(u, v)]
                        m_cm  = 1.0 + self._G(omega) * mu_e / alpha
                        m[i + u, j + v] = m_cm

        return m

@register_frequency_effect('SA', 'TOBkang2023')
class TOB_Kang_2023_SA(FrequencyEffectBase):
    """
    Block-level Saliency Adaptation (based on Kang 2023, equation 21)

        · First divide into dct_size x dct_size blocks, use OpenCV-saliency (SpectralResidual)
          and Otsu threshold to detect SA pixels.
        · If block contains ≥1 SA pixel → this block is SA block (J_SA = 1).
          Otherwise calculate distance from block center ↔ nearest SA block center:
              d  = distance_blocks · dct_size · pixel_pitch  (mm)
              J  = min{ exp(d / (2·viewing_distance)), 1.15 }

        Output (H,W) J_SA map, same pixel values within block.
    """

    def __init__(
        self,
        pixel_pitch: float,        # mm / px  (screen physical pixel pitch)
        viewing_distance: float,   # mm
        dct_size: int = 8,
        saliency_model: str = "spectral",  # Optional 'spectral' / 'fine'
    ):
        super().__init__(
                        pixel_pitch=pixel_pitch,
                        viewing_distance=viewing_distance,
                        dct_size=dct_size,
                        saliency_model=saliency_model)

        self.pixel_pitch     = float(pixel_pitch)
        self.viewing_distance = float(viewing_distance)
        self.dct_size        = dct_size

        # —— OpenCV saliency handle —— #
        if saliency_model.lower() == "spectral":
            self.sal_handler = cv2.saliency.StaticSaliencySpectralResidual_create()
        elif saliency_model.lower() == "fine":
            self.sal_handler = cv2.saliency.StaticSaliencyFineGrained_create()
        else:
            raise ValueError(f"Unsupported saliency model: {saliency_model}")

    # ------------------------------------------------------------------ #
    # 1) Saliency detection → SA block boolean grid
    # ------------------------------------------------------------------ #
    def _sa_blocks_mask(self, y_u8: np.ndarray) -> np.ndarray:
        """
        y_u8: (H,W) uint8 grayscale
        Returns sa_blocks: (H_blk, W_blk) bool, True=SA block
        """
        H, W = y_u8.shape
        # —— 1.1 Generate saliency map 0–255 —— #
        _, sal_map = self.sal_handler.computeSaliency(
            cv2.cvtColor(y_u8, cv2.COLOR_GRAY2BGR)
        )
        sal_map = (sal_map * 255).astype(np.uint8)

        # —— 1.2 Otsu binarization to salient pixels —— #
        _, sal_bin = cv2.threshold(
            sal_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )  # salient=255

        # —— 1.3 Block statistics —— #
        blk = self.dct_size
        H_blk = (H + blk - 1) // blk
        W_blk = (W + blk - 1) // blk
        sa_blocks = np.zeros((H_blk, W_blk), dtype=bool)

        for by in range(H_blk):
            for bx in range(W_blk):
                y0, y1 = by * blk, min((by + 1) * blk, H)
                x0, x1 = bx * blk, min((bx + 1) * blk, W)
                if sal_bin[y0:y1, x0:x1].any():            # Block contains salient pixels
                    sa_blocks[by, bx] = True
        return sa_blocks

    # ------------------------------------------------------------------ #
    # 2) Distance transform (using blocks as pixels) to get distance from each block to nearest SA block (block units)
    # ------------------------------------------------------------------ #
    def _block_distances(self, sa_blocks: np.ndarray) -> np.ndarray:
        """
        sa_blocks: (H_blk, W_blk) bool, True=SA
        Returns dist_blocks: (H_blk, W_blk) float32, block unit L2 distance
        """
        # OpenCV distanceTransform 计算到最近“零像素”的距离
        # We set SA blocks to 0, non-SA blocks to 255
        mask = np.where(sa_blocks, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
        return dist  # same shape, unit: blocks

    # ------------------------------------------------------------------ #
    # 3) Main prediction interface
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Input yuv: (C,H,W) Tensor
        Output J_SA_map: (H,W) Tensor
        """
        Y = yuv[0]
        H, W = Y.shape
        blk = self.dct_size

        # ------- 3.1 Prepare CPU uint8 grayscale image ------- #
        Y_cpu = Y.detach().to('cpu')
        Y_u8  = (Y_cpu.clamp(0, 255).round()).byte().numpy() \
                if Y_cpu.dtype != torch.uint8 else Y_cpu.numpy()

        # ------- 3.2 Get SA block boolean grid ------- #
        sa_blocks = self._sa_blocks_mask(Y_u8)          # (H_blk, W_blk)
        H_blk, W_blk = sa_blocks.shape

        # If entire frame has no salient blocks, directly return all 1 (considered as distance 0)
        if not sa_blocks.any():
            return torch.ones_like(Y, dtype=Y.dtype, device=Y.device)

        # ------- 3.3 Distance transform (block units) ------- #
        dist_blocks = self._block_distances(sa_blocks)  # (H_blk, W_blk)

        # ------- 3.4 Equation (21): block-level J_SA ------- #
        d_mm = dist_blocks * blk * self.pixel_pitch     # mm
        J_blk = np.exp(d_mm / (2.0 * self.viewing_distance))
        J_blk = np.clip(J_blk, 1.0, 1.15).astype(np.float32)  # (H_blk, W_blk)

        # SA blocks (dist==0) should be 1; clip already ensures
        # ------- 3.5 Tile block-level J_SA back to pixel resolution ------- #
        J_img = np.repeat(np.repeat(J_blk, blk, axis=0), blk, axis=1)  # (H',W')
        J_img = J_img[:H, :W]                                          # Crop

        # ------- 3.6 Convert to Tensor and return ------- #
        return torch.from_numpy(J_img).to(Y.device).type_as(Y)