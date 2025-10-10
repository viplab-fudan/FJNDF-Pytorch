"""
Reproduction of the algorithm presented in:

    Bae, Sung-Ho and Kim, Munchurl,
    “A novel DCT-based JND model for luminance adaptation effect in DCT frequency,”
    IEEE Signal Processing Letters, vol. 20, no. 9, pp. 893-896, 2013.

This file implements the DCT-based JND model as described in the above paper.
For full details, please refer to the original publication.
"""

import math
import torch
from pyjnd.models.base_model import FrequencyEffectBase, register_frequency_effect

@register_frequency_effect('CSF', 'SPLbae2013')
class SPL_Bae_2013_CSF(FrequencyEffectBase):
    """
    J_base (Spatial CSF) part in SPL Bae & Kim 2013:
    - Pre-calculate spatial frequency and direction angle for each DCT frequency component (equations 3,4)
    - Get T(f) through quadratic polynomial fitting (equations 6,7)
    - Add oblique effect direction factor (equations 5,8)
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

@register_frequency_effect('LA', 'SPLbae2013')
class SPL_Bae_2013_LA(FrequencyEffectBase):
    """
    MLA part (Luminance Adaptation) in SPL Bae & Kim 2013:
    - Calculate average luminance bg for each 8x8 block
    - Use quadratic polynomial to fit in low/high luminance intervals respectively (equations 10,11,12) to get bg_term
    Returns (H,W) LA map.
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