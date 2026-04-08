import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.utils import _triple, _reverse_repeat_tuple
import math
from networks.layers.RepCDC import RepConv3D_SpatialTemporal_v3

class DSCIM(nn.Module):
    """
    Deep–Shallow Constrained Interaction, DS-CIM

    核心思想：
    1. Deep → Shallow：低分辨率特征只生成 gate（约束），不参与内容生成
    2. Shallow → Deep：高分辨率特征只以“差分 + 稀疏证据”形式参与
    3. 中分辨率特征始终作为主干（anchor feature）
    """

    def __init__(self, in_channels, out_channels, deploy=False):
        super(DSCIM, self).__init__()

        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )

        # ===============================
        # Deep → Shallow：Gate 生成（只用低分辨率）
        # ===============================
        self.deep_gate = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # ===============================
        # Shallow → Deep：差分证据压缩
        # ===============================
        self.diff_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            RepConv3D_SpatialTemporal_v3(out_channels, out_channels, stride=1, deploy=deploy),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1)
        )

        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *xs):
        """
        支持 1 / 2 / 3 个尺度输入

        约定：
        - 三尺度时： (x_high, x_mid, x_low)
        - 中分辨率特征始终作为 anchor
        """

        # ------------------------------------------------------------
        # 双尺度
        # ------------------------------------------------------------
        if len(xs) == 2:
            x1, x2 = xs
            _, _, t1, h1, w1 = x1.shape
            _, _, t2, h2, w2 = x2.shape

            # 情况 A：x1 高分辨率，x2 低分辨率（Encoder 前段）
            if h1 > h2:
                # Deep → gate
                gate = self.deep_gate(x2)
                gate = F.interpolate(gate, size=(t1, h1, w1), mode="trilinear", align_corners=True)

                # Shallow difference
                x2_up = F.interpolate(x2, size=(t1, h1, w1), mode="trilinear", align_corners=True)

                diff = torch.abs(x1 - x2_up)
                diff = self.diff_conv(diff)
                diff = diff * gate.detach()

                x_fused = x1 + diff

            # 情况 B：x1 高分辨率，x2 低分辨率（Encoder 末段）
            else:
                x1_down = self.pool(x1)

                gate = self.deep_gate(x2)

                diff = torch.abs(x1_down - x2)
                diff = self.diff_conv(diff)
                diff = diff * gate.detach()

                x_fused = x2 + diff

        # ------------------------------------------------------------
        # 三尺度（核心交互位置）
        # ------------------------------------------------------------
        elif len(xs) == 3:
            x_high, x_mid, x_low = xs
            _, _, t_mid, h_mid, w_mid = x_mid.shape

            # ---------- Deep → Shallow gate ----------
            gate = self.deep_gate(x_low)
            gate = F.interpolate(
                gate,
                size=(t_mid, h_mid, w_mid),
                mode="trilinear",
                align_corners=True
            )

            # ---------- Shallow difference ----------
            x_high_down = self.pool(x_high)

            diff = torch.abs(x_high_down - x_mid)
            diff = self.diff_conv(diff)
            diff = diff * gate.detach()

            # ---------- Fusion ----------
            x_fused = x_mid + diff

        else:
            raise ValueError(
                f"Feature_Block expected 1, 2 or 3 inputs, got {len(xs)}"
            )

        residual = x_fused
        out = self.conv2(x_fused)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out