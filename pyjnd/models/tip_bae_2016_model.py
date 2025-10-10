"""
Reproduction of the algorithm presented in:

    Bae, Sung-Ho; Kim, Jaeil; and Kim, Munchurl,
    “HEVC-based perceptually adaptive video coding using a DCT-based local distortion detection probability model,”
    IEEE Transactions on Image Processing, vol. 25, no. 7, pp. 3343-3357, 2016.

This file implements the perceptually adaptive HEVC coding method as described in the above paper.
For full details, please refer to the original publication.
"""

import math
import torch
import cv2
import numpy as np

from pyjnd.models.base_model import FrequencyEffectBase, register_frequency_effect

@register_frequency_effect('CSF', 'TIPbae2016')
class TIP_Bae_2016_CSF(FrequencyEffectBase):
    """
    J_base (Spatial CSF) part in TIP Bae & Kim 2016:
    - Is 1/N of SPL_Bae_2013_CSF
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

        # Precompute frequency values
        freq_u, freq_v = torch.meshgrid(torch.arange(self.dct_size, dtype=torch.float32),
                                        torch.arange(self.dct_size, dtype=torch.float32), indexing='ij')
        freq_u, freq_v = freq_u.to(Y.device), freq_v.to(Y.device)  # Move to GPU if needed

        # Calculate the frequency map for all u, v
        f_uv = torch.hypot(freq_u, freq_v) / (2 * self.dct_size * self.theta)
        f_u0 = f_uv[:, 0]
        f_0v = f_uv[0, :]

        # Calculate Jv and Jd for all u, v values
        Jv = self.a_vert[0] * f_uv**2 + self.a_vert[1] * f_uv + self.a_vert[2]
        Jd = self.a_diag[0] * f_uv**2 + self.a_diag[1] * f_uv + self.a_diag[2]

        # Calculate sin2 values using the simplified formula
        sin_phi = 2.0 * f_u0[:, None] * f_0v[None, :] / (f_uv**2 + 1e-8)  # Add small value to avoid div by 0
        sin2 = sin_phi**2

        # Calculate the Jbase map
        Jbase = (Jd - Jv) * sin2 + Jv

        # Reshape Jbase to match the original dimensions (H, W)
        for i in range(0, H, self.dct_size):
            for j in range(0, W, self.dct_size):
                jnd[i:i+self.dct_size, j:j+self.dct_size] = Jbase * self.s * self.dct_size
        return jnd

@register_frequency_effect('LA', 'TIPbae2016')
class TIP_Bae_2016_LA(FrequencyEffectBase):
    """
    J_base (Spatial CSF) part in TIP Bae & Kim 2016:
    - Same as SPL_Bae_2013_CSF
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
        
        # Precompute the frequency map for all (u,v)
        freq_u, freq_v = torch.meshgrid(torch.arange(self.dct_size), torch.arange(self.dct_size), indexing='ij')
        freq_u, freq_v = freq_u.float(), freq_v.float()  # Convert to float
        self.freq_map = torch.hypot(freq_u, freq_v) / (2 * self.dct_size * self.theta)

        # Coefficients for low and high brightness
        self.q_01 = torch.tensor([2.468e-4, 4.466e-3, 1.14], dtype=torch.float32)
        self.q_09 = torch.tensor([1.230e-3, 1.433e-2, 1.34], dtype=torch.float32)

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        device = yuv.device
        
        Y = yuv[0].to(device).to(torch.float32)  # Convert Y to float32 for operations
        H, W = Y.shape
        m = torch.zeros_like(Y, device=device)

        # Precompute the frequency map for all u, v
        f_uv = self.freq_map.to(device) 

        # Calculate the threshold and brightness delta
        threshold = self.threshold
        delta_low = 1 / 0.2  # For low brightness (bg <= threshold)
        delta_high = 1 / 0.6  # For high brightness (bg > threshold)

        # Loop through image blocks (optimized)
        for i in range(0, H, self.dct_size):
            for j in range(0, W, self.dct_size):
                block = Y[i:i+self.dct_size, j:j+self.dct_size]
                bg = block.mean() / 255.0  # Normalize to [0, 1]
                
                # Vectorized calculations for M01, M09
                M01 = self.q_01[0]*f_uv**2 + self.q_01[1]*f_uv + self.q_01[2]
                M09 = self.q_09[0]*f_uv**2 + self.q_09[1]*f_uv + self.q_09[2]

                if bg <= threshold:
                    delta = abs((bg - threshold) * delta_low)
                    term = 1 + (M01 - 1) * (delta ** 0.8)
                else:
                    delta = abs((bg - threshold) * delta_high)
                    term = 1 + (M09 - 1) * (delta ** 0.6)

                # Update the output map
                m[i:i+self.dct_size, j:j+self.dct_size] = term

        return m

@register_frequency_effect('CM', 'TIPbae2016')
class TIP_Bae_2016_CM(FrequencyEffectBase):
    def __init__(self, threshold_split: float, dct_size: int, r_vh: float, screen_h: int):
        super().__init__(threshold_split=threshold_split, dct_size=dct_size, r_vh=r_vh, screen_h=screen_h)
        self.threshold_split = threshold_split
        self.dct_size = dct_size
        self.r_vh = r_vh
        self.screen_h = screen_h
        self.canny_low = 50
        self.canny_high = 150

        # Precompute frequency map for all (u, v)
        theta = math.atan(1.0 / (2 * self.r_vh * self.screen_h)) * (180.0 / math.pi)
        self.freq_map = torch.tensor(
            [[0.0 if u == 0 and v == 0 else math.hypot(u, v) / (2 * dct_size * theta) 
              for v in range(dct_size)] 
             for u in range(dct_size)], dtype=torch.float32)

    @staticmethod
    def _G(omega: torch.Tensor) -> torch.Tensor:
        """ Frequency response G(ω) (Equation 17). """
        return (1.558 * torch.exp(-((omega - 4.12) / 7.16) ** 2) + 2.817).to(omega.device)

    def _canny_edges(self, y: torch.Tensor) -> torch.Tensor:
        """ Canny edge detection. """
        y_8 = (y.clamp(0, 255).round()).to(torch.uint8) if y.dtype != torch.uint8 else y
        y_cpu = y_8.detach().to('cpu').numpy()
        edges = cv2.Canny(y_cpu, self.canny_low, self.canny_high)  # 0/255
        edges = torch.from_numpy(edges).float().to(y.device) / 255.0
        return edges

    @torch.no_grad()
    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Input: yuv (C,H,W), only use Y(0); supports uint8 or float32
        Output: (H,W) M_CM Tensor (float)
        """
        device = yuv.device

        Y = yuv[0].to(device).to(torch.float32)  # Convert Y to float32 for operations
        H, W = Y.shape
        edge_map = self._canny_edges(Y).to(device)   # Get the edge map (H, W)
        m = torch.ones_like(Y, dtype=torch.float32, device=device)  # Output map initialized with 1s

        # Vectorize: We can compute the μ_e (edge density) for all blocks simultaneously.
        for i in range(0, H, self.dct_size):
            for j in range(0, W, self.dct_size):
                block = edge_map[i:i+self.dct_size, j:j+self.dct_size]
                mu_e = block.mean()  # Average edge density for the block
                alpha = 2.0 if mu_e <= self.threshold_split else 1.0

                # Vectorized calculation of m_cm for all u, v in one block
                u, v = torch.meshgrid(torch.arange(self.dct_size, device=device), 
                                      torch.arange(self.dct_size, device=device), indexing='ij')
                omega = self.freq_map[u, v].to(device)  # Frequencies for all u, v
                m_cm = 1.0 + self._G(omega) * mu_e / alpha  # Vectorized m_cm for the whole block

                # Assign the m_cm to the correct positions in the output map
                m[i:i+self.dct_size, j:j+self.dct_size] = m_cm

        return m