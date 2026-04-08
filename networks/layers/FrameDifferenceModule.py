import torch
import torch.nn as nn

# ==================== 帧差分模块（类似TDifferenceConv）===================
class FrameDifferenceModule(nn.Module):
    """
    对对齐后的特征和参考帧进行差分操作（类似TDifferenceConv）

    输入: f0_warp, f1_warp, feat2 [B, C, H, W]
    操作:
      1. 堆叠成 [B, C, 3, H, W] (f0_warp, f1_warp, feat2)
      2. 应用差分操作: 2 * feat2 - f0_warp - f1_warp (在时序维度)
      3. BatchNorm + ReLU
    输出: [B, C, H, W]
    """

    def __init__(self, feature_channels):
        super(FrameDifferenceModule, self).__init__()
        self.feature_channels = feature_channels
        # 使用3D卷积来实现差分操作（类似TDifferenceConv）
        # kernel_size=(3,1,1) 表示在时序维度上进行操作
        # 输入: [B, C, 3, H, W] -> 输出: [B, C, 1, H, W]
        self.diff_conv = nn.Conv3d(
            feature_channels, feature_channels,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),  # 不使用padding，输出时序维度为1
            bias=False
        )
        # 初始化权重以实现差分操作: 2 * feat2 - f0_warp - f1_warp
        # 权重形状: [feature_channels, feature_channels, 3, 1, 1]
        # 对于每个通道，权重应该是: [2, -1, -1] (对应 feat2, f0_warp, f1_warp)
        with torch.no_grad():
            # 初始化权重矩阵
            weight = torch.zeros(feature_channels, feature_channels, 3, 1, 1)
            for i in range(feature_channels):
                weight[i, i, 0, 0, 0] = -1.0  # f0_warp 的权重
                weight[i, i, 1, 0, 0] = -1.0  # f1_warp 的权重
                weight[i, i, 2, 0, 0] = 2.0  # feat2 的权重
            self.diff_conv.weight.copy_(weight)
            self.diff_conv.weight.requires_grad = True  # 允许学习

        self.bn = nn.BatchNorm2d(feature_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f0_warp, f1_warp, feat2):
        """
        Args:
            f0_warp: [B, C, H, W] - 对齐后的第0帧
            f1_warp: [B, C, H, W] - 对齐后的第1帧
            feat2: [B, C, H, W] - 参考帧（第2帧）

        Returns:
            diff_feat: [B, C, H, W] - 差分特征（经过BN和ReLU）
        """
        # 堆叠成 [B, C, 3, H, W]
        stacked = torch.stack([f0_warp, f1_warp, feat2], dim=2)  # [B, C, 3, H, W]

        # 应用差分卷积（类似TDifferenceConv）
        diff_3d = self.diff_conv(stacked)  # [B, C, 1, H, W]

        # 移除时序维度
        diff_feat = diff_3d.squeeze(dim=2)  # [B, C, H, W]

        # BatchNorm + ReLU
        diff_feat = self.bn(diff_feat)
        diff_feat = self.relu(diff_feat)

        return diff_feat