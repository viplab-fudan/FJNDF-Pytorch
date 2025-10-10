import cv2
import numpy as np
import torch
from torch.nn import functional as F


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening (NumPy version).

    Args:
        img (numpy.ndarray): Input image in HWC format, float32 in [0, 1].
        weight (float), radius (int), threshold (int)
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10, test_y_channel=True):
        """
        Args:
            img (Tensor): (B, C, H, W) in YUV format, float in [0, 1]
            weight (float), threshold (int)
            test_y_channel (bool): if True, only sharpen Y channel
        Returns:
            Tensor: sharpened output of same shape
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Full-image sharpening
        def _sharpen(x):
            blur = filter2D(x, self.kernel)
            residual = x - blur
            mask = (torch.abs(residual) * 255 > threshold).float()
            soft_mask = filter2D(mask, self.kernel)
            sharp = torch.clip(x + weight * residual, 0.0, 1.0)
            return soft_mask * sharp + (1 - soft_mask) * x

        if test_y_channel:
            # Split Y and UV
            y = img[:, 0:1, :, :]
            uv = img[:, 1:, :, :]
            # Sharpen only Y
            out_y = _sharpen(y)
            # Concatenate back UV channels unchanged
            out = torch.cat([out_y, uv], dim=1)
        else:
            # Sharpen all channels
            out = _sharpen(img)

        # If original input was 3D, squeeze batch dim
        if out.size(0) == 1:
            out = out.squeeze(0)
        return out
