"""
Reproduction of the algorithm presented in:

    Sun, Yu-Han, Lee, Chiang Lo-Hsuan, and Chang, Tian-Sheuan,
    "IQNet: Image quality assessment guided just noticeable difference prefiltering for versatile video coding,"
    IEEE Open Journal of Circuits and Systems, vol. 5, pp. 17-27, 2023.

This file implements the IQNet model as described in the above paper.
For full details, please refer to the original publication.
"""
import os
import yaml
import torch
import torch.nn as nn
from pyjnd.utils.registry import MODEL_REGISTRY

class PixelAttentionBlock(nn.Module):
    """
    Implementation of the Pixel Attention (PA) block.
    This block uses a gating mechanism where an attention map is generated and
    multiplied element-wise with a feature map.
    """
    def __init__(self, in_channels=32):
        super(PixelAttentionBlock, self).__init__()
    
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

        # Branch to generate the attention map
        self.pa = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
        # Branch to extract features
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input feature map with shape (N, 32, H, W).
        Returns:
            Output feature map with shape (N, 32, H, W).
        """        
        # --- Attention Mechanism ---
        # Generate the feature map
        feature_map = self.conv1(X)

        # Generate the attention map
        attention_map = self.pa(feature_map)

        # Apply attention: element-wise multiplication (gating)
        gated_features = feature_map * attention_map

        # Extract features from the gated map
        features = self.conv2(gated_features)

        # Add the residual connection from the original input
        output = X + features
        
        return output

@MODEL_REGISTRY.register()
class IQNet(nn.Module):
    """
    IQNet: A lightweight Image Quality Assessment guided Just Noticeable Difference prefiltering network.
    This implementation strictly follows the architecture from Figure 6 in the paper.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for YUV or RGB).
        only_train_y (bool): If True, only the Y channel (the first channel) will be processed.
        effect_yml_path (str, optional): Path to the YAML file for model configuration.
    """
    def __init__(self,
                 in_channels=3,
                 only_train_y=True,
                 effect_yml_path=None,
                 precision=None  # Kept for interface compatibility
                 ):
        super(IQNet, self).__init__()

        # Load configuration from YAML file if provided
        if effect_yml_path is not None:
            if not os.path.isfile(effect_yml_path):
                raise FileNotFoundError(f"IQNet: Configuration file not found: {effect_yml_path}")
            with open(effect_yml_path, 'r', encoding='utf-8') as f:
                opt = yaml.safe_load(f)
            in_channels = opt.get('in_channels', in_channels)
            only_train_y = opt.get('only_train_y', only_train_y)

        self.only_train_y = only_train_y
        self.in_channels = 1 if only_train_y else in_channels

        self.net_in_channels = self.in_channels
        self.net_out_channels = self.in_channels

        # Initial feature extraction layer
        self.conv_in = nn.Conv2d(
            in_channels=self.net_in_channels,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        
        # Pixel Attention (PA) block
        self.pa_block = PixelAttentionBlock(in_channels=32)

        # Output layer to learn the JND residual
        self.conv_out = nn.Conv2d(
            in_channels=32,
            out_channels=self.net_out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of IQNet.

        Args:
            X: Input tensor with shape (N, C, H, W), with values assumed to be in the [0,1] range.
        Returns:
            Output tensor with shape (N, C, H, W).
        """
        if self.only_train_y:
            # Separate Y and UV channels
            y_in = X[:, :1]
            uv = X[:, 1:]
            net_input = y_in
        else:
            net_input = X

        # Input for the global residual connection
        global_shortcut = net_input

        # Network flow corresponding to Figure 6 in the paper
        x1 = self.conv_in(net_input)
        x2 = self.pa_block(x1)
        x3_residual = self.conv_out(x2) # The learned JND residual

        # Add the global residual connection
        output = global_shortcut + x3_residual

        if self.only_train_y:
            # Concatenate the processed Y channel with the original UV channels
            return torch.cat([output, uv], dim=1)
        else:
            return output