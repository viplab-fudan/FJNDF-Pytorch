import os
import yaml
import torch
import torch.nn as nn
from pyjnd.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class PUCNet(nn.Module):
    """
    PUCNet: Simple down-up sampling network inspired by SpaceToDepth and ConvTranspose.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        downscale_factor (int): Spatial downscale factor for pixel unshuffle/transpose operations.
    Inputs:
        X (torch.Tensor): Input tensor of shape (N, C, H, W), RGB channel order for color images.

    Returns:
        torch.Tensor: Output tensor of shape (N, C, H, W), RGB channel order for color images.
    """
    def __init__(self, 
                 in_channels=3, 
                 downscale_factor=4, 
                 only_train_y=False,
                 effect_yml_path=None,
                 precision=None
                 ):
        super(PUCNet, self).__init__()
        if effect_yml_path is not None:
            if not effect_yml_path or not os.path.isfile(effect_yml_path):
                raise FileNotFoundError(f"PUCNet: Configuration file does not exist:{effect_yml_path}")
            with open(effect_yml_path, 'r', encoding='utf-8') as f:
                opt = yaml.safe_load(f)
            in_channels=opt['in_channels']
            downscale_factor=opt['downscale_factor']
            only_train_y=opt['only_train_y']

        self.in_channels = 1 if only_train_y else in_channels
        self.only_train_y = only_train_y
        self.factor = downscale_factor
        # Downscale using PixelUnshuffle: channels -> channels * factor^2
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
        # 3x3 convolution on expanded channel dimension
        ch_expanded = self.in_channels * (downscale_factor ** 2)
        self.conv = nn.Conv2d(
            in_channels=ch_expanded,
            out_channels=ch_expanded,
            kernel_size=3,
            padding=1
        )
        # ConvTranspose to upsample back to original resolution
        self.deconv = nn.ConvTranspose2d(
            in_channels=ch_expanded,
            out_channels=self.in_channels,
            kernel_size=downscale_factor,
            stride=downscale_factor
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PUCNet.

        Args:
            X: Input tensor with shape (N, C, H, W), values assumed in [0,1].

        Returns:
            Output tensor with shape (N, C, H, W), values mapped to [0,1].
        """
        if self.only_train_y:
            y = X[:, :1]      # [N,1,H,W]
            uv= X[:, 1:3]     # [N,2,H,W]
            x_in = y
        else:
            x_in = X         # [N,3,H,W]

        # Spatial downscale + channel expansion
        x_sd = self.pixel_unshuffle(x_in) # [N, C*factor^2, H/factor, W/factor]
        # Convolution
        x_conv = self.conv(x_sd) # same shape as x_sd
        # Weighted fusion: 0.1 * Conv + 0.9 * Identity
        x_fused = 0.1 * x_conv + 0.9 * x_sd  # fused features
        # Upsample via transposed convolution
        x_up = self.deconv(x_fused) # [N, C, H, W]

        # Scale to [0,255] and clamp
        # x_scaled = x_up * 255.0
        # x_clipped = x_scaled.clamp(0.0, 255.0)
        x_clipped = x_up
        
        # uv_scaled = uv * 255.0
        # uv_clipped = uv_scaled.clamp(0.0, 255.0)
        uv_clipped = uv
        if self.only_train_y:
            return torch.cat([x_clipped, uv_clipped], dim=1)
        else:
            return x_clipped