"""
Reproduction of the algorithm presented in:

    Wei, Zhenyu and Ngan, King N,
    “Spatio-temporal just noticeable distortion profile for grey scale image/video in DCT domain,”
    IEEE Transactions on Circuits and Systems for Video Technology, vol. 19, no. 3, pp. 337-346, 2009.

This file implements the DCT-based JND model as described in the above paper.
For full details, please refer to the original publication.
"""
import math
import torch
import torch_dct as tdct
import numpy as np
import cv2

from pyjnd.models.base_model import FrequencyEffectBase, register_frequency_effect

@register_frequency_effect('CSF', 'TCSVTwei2009')
class TCSVT_Wei_2009_CSF(FrequencyEffectBase):
    """
    Implements the spatial Contrast Sensitivity Function (CSF) from Wei & Ngan (2009),
    corresponding to equations (6)-(12), (9), and (10) in the paper.
    This class pre-computes the basic JND threshold, T_basic.
    """

    def __init__(self,
                 dct_size: int = 8,
                 r_vh: float = 4.0,
                 screen_h: int = None,
                 a: float = 1.33,
                 b: float = 0.11,
                 c: float = 0.18,
                 oblique_factor: float = 0.6,
                 s: float = 0.25):
        super(TCSVT_Wei_2009_CSF, self).__init__()

        self.dct_size = dct_size
        self.r_vh = r_vh
        if screen_h is None:
            raise ValueError("screen_h (screen height in pixels) must be provided.")
        self.screen_h = float(screen_h)
        self.a = a
        self.b = b
        self.c = c
        self.r = oblique_factor
        self.s = s

        # Pre-compute T_basic values for each frequency component (i, j)
        # and store them in a lookup table.
        lookup = np.zeros((self.dct_size, self.dct_size), dtype=np.float32)
        self.theta = math.atan(1.0 / (2 * self.r_vh * self.screen_h)) * (180.0 / math.pi)
        for i in range(dct_size):
            for j in range(dct_size):
                if i == 0 and j == 0:
                    omega_ij = 0.0
                else:
                    omega_ij = math.hypot(i, j) / (2 * self.dct_size * self.theta)

                phi_i = math.sqrt(1.0 / self.dct_size) if i == 0 else math.sqrt(2.0 / self.dct_size)
                phi_j = math.sqrt(1.0 / self.dct_size) if j == 0 else math.sqrt(2.0 / self.dct_size)

                omega_i0 = 0.0 if i == 0 else math.hypot(i, 0) / (2 * self.dct_size * self.theta)
                omega_0j = 0.0 if j == 0 else math.hypot(0, j) / (2 * self.dct_size * self.theta)

                if omega_ij == 0.0:
                    sin_phi_ij = 0.0
                else:
                    sin_phi_ij = (2.0 * omega_i0 * omega_0j) / (omega_ij ** 2)
                    sin_phi_ij = max(min(sin_phi_ij, 1.0), -1.0)

                cos_sq = 1.0 - sin_phi_ij * sin_phi_ij

                denom = (self.a + self.b * omega_ij)
                if denom <= 1e-6:
                    continue

                T_prime = (1.0 / (phi_i * phi_j)) * (math.exp(self.c * omega_ij) / denom) / (self.r + (1.0 - self.r) * cos_sq)

                lookup[i, j] = self.s * T_prime

        self.register_buffer('T_basic_lookup', torch.from_numpy(lookup))

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Generates a JND map by tiling the pre-computed T_basic lookup table.
        Only the Y channel from the input yuv tensor is used to determine the output size.
        """
        Y = yuv[0]
        H, W = Y.shape
        jnd_map = torch.zeros_like(Y)

        for row in range(0, H, self.dct_size):
            for col in range(0, W, self.dct_size):
                block_h = min(self.dct_size, H - row)
                block_w = min(self.dct_size, W - col)
                jnd_map[row:row + block_h, col:col + block_w] = self.T_basic_lookup[:block_h, :block_w]

        return jnd_map

@register_frequency_effect('LA', 'TCSVTwei2009')
class TCSVT_Wei_2009_LA(FrequencyEffectBase):
    """
    Implements the Luminance Adaptation (LA) model from Wei & Ngan (2009),
    corresponding to equation (19). This factor adjusts JND based on the
    average intensity of each block.
    """

    def __init__(self, dct_size: int = 8):
        super(TCSVT_Wei_2009_LA, self).__init__()
        self.dct_size = dct_size

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Computes the F_lum map for the input YUV image.
        """
        Y = yuv[0]
        H, W = Y.shape
        lum_map = torch.zeros_like(Y)

        for row in range(0, H, self.dct_size):
            for col in range(0, W, self.dct_size):
                row_end = min(row + self.dct_size, H)
                col_end = min(col + self.dct_size, W)
                block = Y[row:row_end, col:col_end]
                avg_intensity = float(block.mean().item())

                if avg_intensity <= 60.0:
                    F_lum = (60.0 - avg_intensity) / 150.0 + 1.0
                elif avg_intensity < 170.0:
                    F_lum = 1.0
                else:
                    F_lum = (avg_intensity - 170.0) / 425.0 + 1.0

                lum_map[row:row_end, col:col_end] = F_lum

        return lum_map


@register_frequency_effect('CM', 'TCSVTwei2009')
class TCSVT_Wei_2009_CM(FrequencyEffectBase):
    """
    Implements the Contrast Masking (CM) model from Wei & Ngan (2009),
    corresponding to equations (20)-(23). This factor models how image content
    (plane, edge, texture) masks distortion.
    """

    def __init__(self,
                 dct_size: int = 8,
                 r_vh: float = 4.0,
                 screen_h: int = None,
                 a: float = 1.33,
                 b: float = 0.11,
                 c: float = 0.18,
                 oblique_factor: float = 0.6,
                 s: float = 0.25,
                 alpha: float = 0.1,
                 beta: float = 0.2):
        super(TCSVT_Wei_2009_CM, self).__init__()

        self.dct_size = dct_size
        self.alpha = alpha
        self.beta = beta

        # Instantiate a CSF model internally to get the T_basic lookup table.
        self.csf_model = TCSVT_Wei_2009_CSF(
            dct_size=dct_size, r_vh=r_vh, screen_h=screen_h,
            a=a, b=b, c=c, oblique_factor=oblique_factor, s=s
        )
        self.register_buffer('T_basic_lookup', self.csf_model.T_basic_lookup)

    @staticmethod
    def _compute_F_lum_for_block(block: np.ndarray) -> float:
        """
        Computes F_lum for a single block based on its average intensity (Eq. 19).
        """
        avg_intensity = float(block.mean())
        if avg_intensity <= 60.0:
            return (60.0 - avg_intensity) / 150.0 + 1.0
        elif avg_intensity < 170.0:
            return 1.0
        else:
            return (avg_intensity - 170.0) / 425.0 + 1.0

    @staticmethod
    def _compute_block_type(rho_edge: float, alpha: float, beta: float) -> str:
        """
        Determines block type (plane, edge, texture) based on edge density (Eq. 21).
        """
        if rho_edge <= alpha:
            return "plane"
        elif rho_edge <= beta:
            return "edge"
        else:
            return "texture"

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Computes the F_contrast map for the input YUV image.
        """
        Y = yuv[0]
        device = Y.device
        H, W = Y.shape

        Y_cpu = Y.detach().cpu().float().numpy()
        edge_map = cv2.Canny(Y_cpu.astype(np.uint8), 100, 200)
        Y_dct_cpu = torch.from_numpy(Y_cpu).float()
        contrast_map = torch.zeros((H, W), dtype=torch.float32, device=device)
        T_basic_lookup_np = self.T_basic_lookup.cpu().numpy()

        N = self.dct_size
        alpha = self.alpha
        beta = self.beta

        for row in range(0, H, N):
            for col in range(0, W, N):
                row_end = min(row + N, H)
                col_end = min(col + N, W)
                h_block, w_block = row_end - row, col_end - col

                # 1. Classify block type based on edge density
                block_edge = edge_map[row:row_end, col:col_end]
                rho_edge = np.count_nonzero(block_edge) / float(h_block * w_block)
                block_type = self._compute_block_type(rho_edge, alpha, beta)

                # 2. Compute luminance adaptation factor for the block
                pixel_block = Y_cpu[row:row_end, col:col_end]
                F_lum_block = self._compute_F_lum_for_block(pixel_block)

                # 3. Compute DCT coefficients for the block
                block_for_dct = Y_dct_cpu[row:row_end, col:col_end]
                if h_block < N or w_block < N: # Pad if not a full block
                    padded = torch.zeros((N, N), dtype=torch.float32)
                    padded[:h_block, :w_block] = block_for_dct
                    block_for_dct = padded
                dct_coeff_np = tdct.dct_2d(block_for_dct).numpy()

                # 4. Compute F_contrast for each pixel in the block
                for i in range(h_block):
                    for j in range(w_block):
                        # Calculate Psi(i,j) factor (Eq. 22)
                        if block_type in ("plane", "edge"):
                            psi = 1.0
                        else: # texture block
                            psi = 2.25 if (i * i + j * j) <= 16 else 1.25

                        # Calculate final F_contrast value (Eq. 23)
                        if block_type in ("plane", "edge") and (i * i + j * j) <= 16:
                            F_contrast_val = psi
                        else:
                            C_ij = abs(dct_coeff_np[i, j])
                            T_basic_ij = T_basic_lookup_np[i, j]
                            denom = T_basic_ij * F_lum_block
                            
                            if denom <= 1e-6:
                                clipped = 1.0
                            else:
                                ratio = C_ij / denom
                                pow_r = ratio ** 0.36
                                clipped = min(4.0, max(1.0, pow_r))
                            F_contrast_val = psi * clipped

                        contrast_map[row + i, col + j] = F_contrast_val

        return contrast_map

@register_frequency_effect('TM', 'TCSVTwei2009')
class TCSVT_Wei_2009_TM(FrequencyEffectBase):
    """
    Temporal Modulation (TM) factor. This is a placeholder implementation.
    A full implementation would consider temporal aspects as described in
    equations (24)-(26) of the paper.
    """

    def __init__(self):
        super(TCSVT_Wei_2009_TM, self).__init__()

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Returns a map of ones, as temporal effects are not implemented.
        """
        Y = yuv[0]
        tm_map = torch.ones_like(Y, dtype=torch.float32, device=Y.device)
        return tm_map