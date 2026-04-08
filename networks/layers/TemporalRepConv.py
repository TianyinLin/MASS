import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDC(nn.Module):
    """
    时间差分卷积（Temporal Difference Convolution）

    三种时间模式，每个模式使用不同时间跨度的卷积核：
    - mode='short': 时间跨度1（只关注当前帧）
    - mode='medium': 时间跨度3（关注局部时序）
    - mode='long': 时间跨度5（关注更长时序）

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 时间跨度（1, 3, 或 5，根据mode自动确定）
        stride: 步长
        padding: 填充（根据kernel_size自动计算）
        groups: 分组卷积数
        bias: 是否使用偏置
        mode: 时间模式，'short', 'medium', 'long'
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=0, groups=1, bias=False, mode='short'):
        super().__init__()
        assert mode in ['short', 'medium', 'long'], "mode must be 'short', 'medium', or 'long'"

        self.mode = mode
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 根据模式确定实际的时间跨度和填充
        if mode == 'short':
            # 模式1：时间跨度1
            self.actual_kernel_size = 1
            self.actual_padding = 0
            self.conv = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=(1, 1, 1),
                                  stride=(stride, 1, 1),
                                  padding=(0, 0, 0),
                                  groups=groups, bias=bias)
        elif mode == 'medium':
            # 模式2：时间跨度3
            self.actual_kernel_size = 3
            self.actual_padding = 1
            self.conv = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=(3, 1, 1),
                                  stride=(stride, 1, 1),
                                  padding=(1, 0, 0),
                                  groups=groups, bias=bias)
        elif mode == 'long':
            # 模式3：时间跨度5
            self.actual_kernel_size = 5
            self.actual_padding = 2
            self.conv = nn.Conv3d(in_channels, out_channels,
                                  kernel_size=(5, 1, 1),
                                  stride=(stride, 1, 1),
                                  padding=(2, 0, 0),
                                  groups=groups, bias=bias)

        # 初始化权重（使用kaiming初始化，不使用高斯）
        self._init_weights()

    def _init_weights(self):
        """使用kaiming初始化权重"""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def _tdiff_kernel_3d(self, weight: torch.Tensor) -> torch.Tensor:
        """
        将普通 conv3d weight 转换为时间差分形式（3D版本）：

        对于 medium (3x1x1): 时间维度参数 [2, -1, -1]
        对于 long (5x1x1): 时间维度参数 [-1/12, 4/3, -5/2, 4/3, -1/12]

        Args:
            weight: [out_channels, in_channels//groups, T, 1, 1] （Parameter 或 Tensor）
        Returns:
            td_w: 同 shape 的 Tensor（可微分，与 weight 有计算图连接）
        """
        t = weight.size(2)  # 时间维度大小（3 或 5）

        if t == 3:
            # medium 模式：时间维度参数 [2, -1, -1]
            # 这意味着：中心帧权重=2，前后帧权重=-1
            td_w = weight.clone()
            # 原始权重形状: [out, in, 3, 1, 1]
            # 应用差分系数：[2, -1, -1]
            td_w[:, :, 0, :, :] = weight[:, :, 0, :, :] * (-1)  # 前帧: -1
            td_w[:, :, 1, :, :] = weight[:, :, 1, :, :] * 2  # 中心帧: 2
            td_w[:, :, 2, :, :] = weight[:, :, 2, :, :] * (-1)  # 后帧: -1
        elif t == 5:
            # long 模式：时间维度参数 [-1/12, 4/3, -5/2, 4/3, -1/12]
            td_w = weight.clone()
            # 原始权重形状: [out, in, 5, 1, 1]
            # 应用差分系数：[-1/12, 4/3, -5/2, 4/3, -1/12]
            td_w[:, :, 0, :, :] = weight[:, :, 0, :, :] * (1 / 12)  # t-2: -1/12
            td_w[:, :, 1, :, :] = weight[:, :, 1, :, :] * (-4 / 3)  # t-1: 4/3
            td_w[:, :, 2, :, :] = weight[:, :, 2, :, :] * (5 / 2)  # t: -5/2
            td_w[:, :, 3, :, :] = weight[:, :, 3, :, :] * (-4 / 3)  # t+1: 4/3
            td_w[:, :, 4, :, :] = weight[:, :, 4, :, :] * (1 / 12)  # t+2: -1/12
        else:
            # short 模式或其他：不应用差分
            td_w = weight

        return td_w

    def forward(self, x):
        """
        前向传播

        对于 medium (3x1x1) 和 long (5x1x1) 模式，应用时间差分卷积
        对于 short (1x1x1) 模式，直接使用普通卷积

        Args:
            x: 输入特征 [B, in_channels, T, H, W]

        Returns:
            x_temporal: 时间处理后的特征 [B, out_channels, T, H, W]
        """
        if self.mode == 'short':
            # 1x1x1 模式：直接使用普通卷积（不需要差分）
            return self.conv(x)
        else:
            # medium (3x1x1) 和 long (5x1x1) 模式：应用时间差分卷积
            weight = self.conv.weight  # Parameter [out, in, T, 1, 1]
            td_weight = self._tdiff_kernel_3d(weight)  # Tensor，保留计算图

            # 使用 conv 的属性进行卷积
            return F.conv3d(
                x, td_weight, self.conv.bias,
                stride=self.conv.stride, padding=self.conv.padding,
                dilation=self.conv.dilation, groups=self.conv.groups
            )

    def get_temporal_weight(self):
        weight = self.conv.weight
        bias = self.conv.bias

        temporal_weight = torch.zeros(
            self.out_channels,
            self.in_channels // self.groups,
            5, 1, 1,
            device=weight.device,
            dtype=weight.dtype
        )

        if self.mode == 'short':
            # 1x1x1 -> 放到中心帧
            temporal_weight[:, :, 2:3, :, :] = weight

        elif self.mode == 'medium':
            # 3x1x1 -> 放到中心 3 帧
            temporal_weight[:, :, 1:4, :, :] = weight

        elif self.mode == 'long':
            temporal_weight = weight

        if bias is None:
            bias = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)

        return temporal_weight, bias


class RepConv3D(nn.Module):
    """
    重参数化的3D卷积（时间维度）

    包含三个并行分支，每个分支使用不同时间跨度的卷积核：
    - short_tdc: 时间跨度1（当前帧）
    - medium_tdc: 时间跨度3（局部时序）
    - long_tdc: 时间跨度5（长时序）

    训练时：三个分支并行计算后相加
    推理时：可以重参数化为单个时间跨度5的卷积

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 时间跨度，固定为5（用于重参数化）
        stride: 步长
        padding: 填充，默认2
        groups: 分组卷积数
        use_fusion_conv: 是否使用1x1x1卷积进行通道融合（默认True）
    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=2, groups=1, use_fusion_conv=True):
        super(RepConv3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.use_fusion_conv = use_fusion_conv
        # 三个并行分支
        self.short_tdc = nn.Sequential(
            TemporalDC(in_channels, out_channels, kernel_size, stride, padding,
                       groups=groups, bias=False, mode='short'),
            nn.BatchNorm3d(out_channels)
        )
        self.medium_tdc = nn.Sequential(
            TemporalDC(in_channels, out_channels, kernel_size, stride, padding,
                       groups=groups, bias=False, mode='medium'),
            nn.BatchNorm3d(out_channels)
        )
        self.long_tdc = nn.Sequential(
            TemporalDC(in_channels, out_channels, kernel_size, stride, padding,
                       groups=groups, bias=False, mode='long'),
            nn.BatchNorm3d(out_channels)
        )

        # 可选的1x1x1卷积用于通道融合
        if self.use_fusion_conv:
            self.fusion_conv = nn.Sequential(
                nn.Conv3d(out_channels * 3, out_channels, (1, 1, 1), (1, 1, 1), (0, 0, 0), bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.fusion_conv = None

    def forward(self, x):
        """
        前向传播

        - 使用三个并行分支：short_tdc (时间跨度1), medium_tdc (时间跨度3), long_tdc (时间跨度5)
        - 三个分支的输出拼接后融合
        - 可选：通过1x1x1卷积进行通道融合
            - 经过ReLU激活

        Args:
            x: 输入特征 [B, in_channels, T, H, W]

        Returns:
            out: 输出特征 [B, out_channels, T, H, W]
        """
        out = torch.cat([self.short_tdc(x), self.medium_tdc(x), self.long_tdc(x)], dim=1)

        # 可选：使用1x1x1卷积进行通道融合
        if self.fusion_conv is not None:
            out = self.fusion_conv(out)

        out = F.relu(out)
        return out


if __name__ == '__main__':
    # 测试代码
    import torch

    # 创建测试输入 [B, C, T, H, W]
    x = torch.randn(2, 32, 10, 64, 64)

    # 创建RepConv3D模块
    model = RepConv3D(in_channels=32, out_channels=32, use_fusion_conv=False)

    # 训练模式前向传播
    print("训练模式测试...")
    with torch.no_grad():
        out = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {out.shape}")

    print("\n仅保留训练路径，不再支持 deploy 切换。")

