"""
Reproduction of the algorithm presented in:

    Chou, Chun-Hsien and Li, Yun-Chin,
    “A perceptually tuned subband image coder based on the measure of just-noticeable-distortion profile,”
    IEEE Transactions on Circuits and Systems for Video Technology, vol. 5, no. 6, pp. 467-476, 1995.

This file implements the JND-based model as described in the above paper.
For full details, please refer to the original publication.
"""

import torch
import torch.nn.functional as F
from pyjnd.models.base_model import SpatialEffectBase, register_spatial_effect


@register_spatial_effect('LA', 'TCSVTchou1995')
class TCSVT_Chou_1995_LA(SpatialEffectBase):
    """
    Luminance adaptation (LA) effect based on Chou & Li, IEEE TCSVT 1995.

    f2(bg) = {
      T0 * (1 - sqrt(bg / 127)) + 3,    for bg <= 127
      slope * (bg - 127) + 3,           for bg > 127
    }

    bg(x, y) = (1/32) * sum_{i,j=1..5} p(x-3+i, y-3+j) * B[i,j]

    where B = [[1 1 1 1 1],
               [1 2 2 2 1],
               [1 2 0 2 1],
               [1 2 2 2 1],
               [1 1 1 1 1]]
    """
    def __init__(self, T0: float, slope: float, bg_threshold: float = 127.0):
        super().__init__(T0=T0, slope=slope, bg_threshold=bg_threshold)
        self.T0 = T0
        self.slope = slope
        self.bg_threshold = bg_threshold

        # Build 5×5 low-pass weighting kernel B/32
        B = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ], dtype=torch.float32) / 32.0
        # Shape: (out_channels=1, in_channels=1, 5, 5)
        self.register_buffer('bg_kernel', B.view(1, 1, 5, 5))

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Compute the luminance adaptation JND map for the Y channel.

        Args:
            yuv: Input YUV tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Single-channel LA JND map of shape (H, W).
        """
        # Take Y channel and add batch + channel dimensions -> (1, 1, H, W)
        Y = yuv[0:1].unsqueeze(0)

        # Compute background luminance bg via 5×5 convolution with padding=2
        bg = F.conv2d(Y, self.bg_kernel, padding=2).squeeze(0).squeeze(0)

        # Apply f2(bg) as defined in the paper
        T0 = self.T0
        th = self.bg_threshold
        slp = self.slope
        f2 = torch.where(
            bg <= th,
            T0 * (1 - torch.sqrt(bg / th)) + 3.0,
            slp * (bg - th) + 3.0
        )

        return f2


@register_spatial_effect('CM', 'TCSVTchou1995')
class TCSVT_Chou_1995_CM(SpatialEffectBase):
    """
    Contrast masking (CM) effect based on Chou & Li, IEEE TCSVT 1995.

    f1(bg, mg) = mg * alpha(bg) + beta(bg)                 (Eq.2)
      alpha(bg) = bg * a_coeff + constant                  (Eq.4)
      beta(bg)  = intercept - bg * bg_coeff                (Eq.5)

    mg(x,y) = max_k |grad_k(x,y)|                           (Eq.7)
      grad_k = (1/16) * conv2d(Y, G_k), k = 1..4            (Eq.8)

    bg(x,y) = (1/32) * conv2d(Y, B)                         (Eq.9)
      B = [[1,1,1,1,1],
           [1,2,2,2,1],
           [1,2,0,2,1],
           [1,2,2,2,1],
           [1,1,1,1,1]]
    """
    def __init__(
        self,
        intercept: float,    # intercept term in beta(bg)
        bg_coeff: float,     # slope coefficient in beta(bg) = intercept - bg * bg_coeff
        a_coeff: float,      # slope coefficient in alpha(bg) = bg * a_coeff + constant
        constant: float,     # constant offset in alpha(bg)
    ):
        super().__init__(
            intercept=intercept,
            bg_coeff=bg_coeff,
            a_coeff=a_coeff,
            constant=constant,
        )
        # Store parameters
        self.intercept = intercept
        self.bg_coeff = bg_coeff
        self.a_coeff = a_coeff
        self.constant = constant

        # 5×5 background low-pass kernel B/32
        B = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 0, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ], dtype=torch.float32) / 32.0
        self.register_buffer('bg_kernel', B.view(1, 1, 5, 5))

        # 5×5 directional gradient kernels G1..G4, each divided by 16
        G1 = torch.tensor([
            [ 0,  0,  0,  0,  0],
            [ 1,  3,  8,  3,  1],
            [ 0,  0,  0,  0,  0],
            [-1, -3, -8, -3, -1],
            [ 0,  0,  0,  0,  0],
        ], dtype=torch.float32) / 16.0
        G2 = torch.tensor([
            [ 0,  0,  1,  0,  0],
            [ 0,  8,  3,  0,  0],
            [ 1,  3,  0, -3, -1],
            [ 0,  0, -3, -8,  0],
            [ 0,  0, -1,  0,  0],
        ], dtype=torch.float32) / 16.0
        G3 = torch.tensor([
            [ 0,  0,  1,  0,  0],
            [ 0,  0,  3,  8,  0],
            [-1, -3,  0,  3,  1],
            [ 0, -8, -3,  0,  0],
            [ 0,  0, -1,  0,  0],
        ], dtype=torch.float32) / 16.0
        G4 = torch.tensor([
            [ 0,  1,  0, -1,  0],
            [ 0,  3,  0, -3,  0],
            [ 0,  8,  0, -8,  0],
            [ 0,  3,  0, -3,  0],
            [ 0,  1,  0, -1,  0],
        ], dtype=torch.float32) / 16.0

        for idx, G in enumerate((G1, G2, G3, G4), start=1):
            # Register as grad1_kernel, ..., grad4_kernel
            self.register_buffer(f'grad{idx}_kernel', G.view(1, 1, 5, 5))

    def predict(self, yuv: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrast masking JND map for the Y channel.

        Args:
            yuv: Input YUV tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Single-channel CM JND map of shape (H, W).
        """
        # Take Y channel and add batch + channel dimensions -> (1, 1, H, W)
        Y = yuv[0:1].unsqueeze(0)

        # 1) Compute background luminance bg(x, y) using B convolution
        bg = F.conv2d(Y, self.bg_kernel, padding=2).squeeze(0).squeeze(0)

        # 2) Compute directional gradients grad_k and take max magnitude mg(x, y)
        grads = []
        for k in range(1, 5):
            Gk = getattr(self, f'grad{k}_kernel')
            gradk = F.conv2d(Y, Gk, padding=2)  # (1, 1, H, W)
            grads.append(gradk.abs())
        mg = torch.max(torch.stack(grads, dim=0), dim=0)[0]
        mg = mg.squeeze(0).squeeze(0)  # -> (H, W)

        # 3) Compute alpha(bg) and beta(bg)
        alpha = bg * self.a_coeff + self.constant
        beta = self.intercept - bg * self.bg_coeff

        # 4) f1(bg, mg) = mg * alpha(bg) + beta(bg)
        f1 = mg * alpha + beta

        return f1