"""
Reproduction of the algorithm presented in:

    Ma, Chengqian; Wu, Zhiqiang; Cai, Chunlei; Zhang, Pengwei; Wang, Yi; Zheng, Long; Chen, Chao; and Zhou, Quan,
    “Rate-perception optimized preprocessing for video coding,”
    arXiv preprint arXiv:2301.10455, 2023.

This file implements the rate-perception optimized preprocessing method as described in the above paper.
For full details, please refer to the original preprint.

This implementation also references the authors'official open-source code available at:
    https://github.com/submissions-ai/Rate-Perception-Optimized-Preprocessing-for-Video-Coding
"""

import os
import yaml
import torch
import torch.nn.functional as F
from torch import nn

from pyjnd.utils.registry import MODEL_REGISTRY

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    """Create a Conv2d layer with automatic padding."""
    padding = (kernel_size - 1) // 2 * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                     padding=padding, bias=True, dilation=dilation, groups=groups)

def get_valid_padding(kernel_size, dilation):
    """Compute padding so that output spatial dims match input."""
    kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
    return (kernel - 1) // 2

def pad(pad_type, padding):
    """Return a padding layer of the specified type."""
    if padding == 0 or pad_type == 'zero':
        return None
    if pad_type == 'reflect':
        return nn.ReflectionPad2d(padding)
    if pad_type == 'replicate':
        return nn.ReplicationPad2d(padding)
    raise NotImplementedError(f'Padding [{pad_type}] is not implemented')

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """Factory for activation layers."""
    t = act_type.lower()
    if t == 'relu':
        return nn.ReLU(inplace)
    if t == 'lrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    if t == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    raise NotImplementedError(f'Activation [{act_type}] is not found')

def sequential(*modules):
    """Flatten nested nn.Sequential modules into one."""
    layers = []
    for m in modules:
        if isinstance(m, nn.Sequential):
            layers.extend(list(m.children()))
        elif isinstance(m, nn.Module):
            layers.append(m)
    return nn.Sequential(*layers)

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1,
               groups=1, bias=True, pad_type='zero', act_type='relu'):
    """Convolution + optional padding + activation block."""
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding)
    c = nn.Conv2d(in_nc, out_nc, kernel_size, stride,
                  padding=padding if pad_type == 'zero' else 0,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    return sequential(p, c, a)

def pixelshuffle_block(in_channels, out_channels, downscale_factor=2, kernel_size=3, stride=1):
    """Conv2d -> PixelShuffle(downscale_factor)."""
    conv = conv_layer(in_channels, out_channels * (downscale_factor**2), kernel_size, stride)
    ps = nn.PixelShuffle(downscale_factor)
    return sequential(conv, ps)

class ESA(nn.Module):
    """Enhanced Spatial Attention module for channel-wise feature refinement."""
    def __init__(self, n_feats, conv):
        super().__init__()
        f = n_feats // 4
        self.conv1   = conv(n_feats, f, 1)
        self.conv_f  = conv(f, f, 1)
        self.conv_max= conv(f, f, 3, padding=1)
        self.conv2   = conv(f, f, 3, stride=2)
        self.conv3   = conv(f, f, 3, padding=1)
        self.conv3_  = conv(f, f, 3, padding=1)
        self.conv4   = conv(f, n_feats, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_    = self.conv1(x)
        c1     = self.conv2(c1_)
        v_max  = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range= self.relu(self.conv_max(v_max))
        c3     = self.relu(self.conv3(v_range))
        c3     = self.conv3_(c3)
        c3     = F.interpolate(c3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf     = self.conv_f(c1_)
        c4     = self.conv4(c3 + cf)
        m      = self.sigmoid(c4)
        return x * m

class RFDB(nn.Module):
    """Residual Feature Distillation Block with ESA attention."""
    def __init__(self, in_channels, distillation_rate=0.25):
        super().__init__()
        dc = in_channels // 2
        rc = in_channels
        # distilled and remaining convs
        self.c1_d = conv_layer(in_channels, dc, 1)
        self.c1_r = conv_layer(in_channels, rc, 3)
        self.c2_d = conv_layer(rc, dc, 1)
        self.c2_r = conv_layer(rc, rc, 3)
        self.c4   = conv_layer(rc, dc, 3)
        self.act  = activation('lrelu', neg_slope=0.05)
        self.c5   = conv_layer(dc * 3, in_channels, 1)

    def forward(self, x):
        d1 = self.act(self.c1_d(x))
        r1 = self.act(self.c1_r(x) + x)
        d2 = self.act(self.c2_d(r1))
        r2 = self.act(self.c2_r(r1) + r1)
        r3 = self.act(self.c4(r2))
        out= torch.cat([d1, d2, r3], dim=1)
        return self.c5(out)

@MODEL_REGISTRY.register()
class RPPNet(nn.Module):
    """
    RPPNet: Rate-Perception Preprocessing Network (RPP) from Ma et al. 2023.

    Args:
        in_channels (int): Number of input channels (e.g. 3 for RGB).
        num_features (int): Number of feature channels in each RFDB (default=16).
        num_modules (int): Number of RFDB blocks to stack (default=3).
        downscale_factor (int): Down/up-sampling factor (default=2).
        only_train_y (bool): If True, only process Y channel and concatenate UV back.
        effect_yml_path (str, optional): Path to YAML config.
        precision (unused): Placeholder.
    """
    def __init__(self,
                 in_channels=3,
                 downscale_factor=2,
                 num_features=16,
                 num_modules=3,
                 only_train_y=False,
                 effect_yml_path=None,
                 precision=None
                ):
        super(RPPNet, self).__init__()
        if effect_yml_path is not None:
            if not effect_yml_path or not os.path.isfile(effect_yml_path):
                raise FileNotFoundError(f"RPPNet: config file not found: {effect_yml_path}")
            with open(effect_yml_path, 'r', encoding='utf-8') as f:
                opt = yaml.safe_load(f)
            in_channels       = opt['in_channels']
            downscale_factor  = opt['downscale_factor']
            num_features      = opt['num_features']
            num_modules       = opt['num_modules']
            only_train_y      = opt['only_train_y']

        self.in_channels      = 1 if only_train_y else in_channels
        self.downscale_factor = downscale_factor
        self.num_features     = num_features
        self.num_modules      = num_modules
        self.only_train_y     = only_train_y

        # number of channels after PixelUnshuffle
        ch_expanded = self.in_channels * (downscale_factor ** 2)

        # shallow feature extractor (downsample spatial → channels)
        self.fea_unshuffle = nn.PixelUnshuffle(self.downscale_factor)
        # project to num_features
        self.fea_conv_c    = conv_block(ch_expanded,
                                        num_features,
                                        kernel_size=1,
                                        act_type='lrelu')

        # RFDB blocks for feature enhancement
        self.B1 = RFDB(num_features)

        # low-resolution convolution
        self.LR_conv   = conv_layer(num_features, num_features, kernel_size=3)
        # upsampler to reconstruct original resolution
        self.upsampler = pixelshuffle_block(num_features,
                                            self.in_channels,
                                            downscale_factor=self.downscale_factor)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # if only training Y channel, split UV
        if self.only_train_y:
            y    = X[:, :1]
            uv   = X[:, 1:3]
            x_in = y
        else:
            x_in = X

        # extract shallow features (no projection yet)
        f_shallow = self.fea_unshuffle(x_in)      # [N, C*sf^2, H/sf, W/sf]
        # project to num_features
        f         = self.fea_conv_c(f_shallow)    # [N, num_features, H/sf, W/sf]

        # RFDB enhancement + residual fusion
        b1     = self.B1(f)
        merged = b1 + f

        # low-resolution conv
        res = self.LR_conv(merged)                # [N, num_features, H/sf, W/sf]

        # fuse shallow and residual features at low resolution
        fused = 0.1 * res + 0.9 * f_shallow # required C = 1, sf = 4, num_features = 16

        # upsample fused features back to original resolution
        out = self.upsampler(fused)               # [N, in_channels, H, W]
        
        out = 0.01 * out + 0.99 * x_in

        # directly output fused result (plus UV if needed)
        if self.only_train_y:
            return torch.cat([out, uv], dim=1)
        else:
            return out