import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_gaussian_kernel_2d(size, sigma, normalize=True):
    assert size % 2 == 1, "size must be odd"
    center = size // 2
    kernel = torch.zeros(size, size)

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    if normalize:
        kernel = kernel / kernel.sum()

    return kernel


class SpatialDC3D(nn.Module):
    """
    空间差分卷积（3D版本，时间维度独立）
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=2, groups=1, bias=False, mode='center'):
        super().__init__()
        assert mode in ['center', 'local', 'global'], "mode must be 'center', 'local', or 'global'"

        self.mode = mode
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == 'center':
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=(1, 1, 1),
                stride=(1, stride, stride), padding=(0, 0, 0),
                groups=groups, bias=bias
            )
        elif mode == 'local':
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=(1, 3, 3),
                stride=(1, stride, stride), padding=(0, 1, 1),
                groups=groups, bias=bias
            )
        else:
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=(1, 5, 5),
                stride=(1, stride, stride), padding=(0, 2, 2),
                groups=groups, bias=bias
            )

        self._init_gaussian_weights()

    def _init_gaussian_weights(self):
        with torch.no_grad():
            if self.mode == 'center':
                self.conv.weight.data.fill_(1.0)
            elif self.mode == 'local':
                gaussian_kernel = create_gaussian_kernel_2d(3, sigma=1.0, normalize=True)
                for i in range(self.conv.weight.shape[0]):
                    for j in range(self.conv.weight.shape[1]):
                        nn.init.kaiming_normal_(self.conv.weight[i:i + 1, j:j + 1], mode='fan_out', nonlinearity='relu')
                        self.conv.weight.data[i, j, 0, :, :] *= gaussian_kernel
            else:
                gaussian_kernel = create_gaussian_kernel_2d(5, sigma=2.0, normalize=True)
                for i in range(self.conv.weight.shape[0]):
                    for j in range(self.conv.weight.shape[1]):
                        nn.init.kaiming_normal_(self.conv.weight[i:i + 1, j:j + 1], mode='fan_out', nonlinearity='relu')
                        self.conv.weight.data[i, j, 0, :, :] *= gaussian_kernel

    def _sdiff_kernel_3d(self, weight: torch.Tensor) -> torch.Tensor:
        k = weight.size(-1)
        assert k % 2 == 1, "kernel size must be odd for SDifference"
        c = k // 2

        sd_w = -weight.clone()
        sum_w = weight.sum(dim=(3, 4), keepdim=True).squeeze(2)
        sd_w[:, :, 0, c:c + 1, c:c + 1] = sum_w
        return sd_w

    def forward(self, x):
        if self.mode == 'center':
            return self.conv(x)

        sd_weight = self._sdiff_kernel_3d(self.conv.weight)
        return F.conv3d(
            x, sd_weight, self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups
        )


class RepConv3D_Spatial(nn.Module):
    """
    仅保留训练路径的3D空间差分卷积模块。
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=2, groups=1, use_fusion_conv=True):
        super().__init__()
        self.use_fusion_conv = use_fusion_conv

        self.center_tdc = nn.Sequential(
            SpatialDC3D(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, mode='center'),
            nn.BatchNorm3d(out_channels)
        )
        self.local_tdc = nn.Sequential(
            SpatialDC3D(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, mode='local'),
            nn.BatchNorm3d(out_channels)
        )
        self.global_tdc = nn.Sequential(
            SpatialDC3D(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, mode='global'),
            nn.BatchNorm3d(out_channels)
        )

        if self.use_fusion_conv:
            self.fusion_conv = nn.Sequential(
                nn.Conv3d(out_channels * 3, out_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0), bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0), bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.fusion_conv = None

    def forward(self, x):
        out = torch.cat([self.center_tdc(x), self.local_tdc(x), self.global_tdc(x)], dim=1)
        if self.fusion_conv is not None:
            out = self.fusion_conv(out)
        return F.relu(out)
