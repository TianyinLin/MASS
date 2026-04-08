import torch
import torch.nn as nn

# ==================== 改进的特征提取模块 ====================
class ResidualBlock2D(nn.Module):
    """2D残差块（参考RAFT）"""

    def __init__(self, in_planes, planes, stride=1, norm_fn='batch'):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            # 如果stride != 1 或通道数变化，都需要norm3用于downsample
            if stride != 1 or planes != in_planes:
                self.norm3 = nn.BatchNorm2d(planes)
            else:
                self.norm3 = None
        else:
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if stride != 1 or planes != in_planes:
                self.norm3 = nn.Sequential()
            else:
                self.norm3 = None

        # 如果stride != 1 或通道数变化，都需要downsample
        if stride == 1 and planes == in_planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

#
class MultiScaleFeatureExtractor(nn.Module):
    """
    多尺度特征提取器
    共享中间层特征，避免重复计算
    """

    def __init__(self, input_channels=1, base_channels=64, norm_fn='batch'):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.base_channels = base_channels

        # 共享的初始特征提取
        if norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)
        else:
            self.norm1 = nn.Sequential()

        # stride=1保持原始分辨率
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # 分支1: 原始分辨率 (1x) - 从 [B, 32, H, W] 处理到 [B, base_channels, H, W]
        self.branch_1x = nn.Sequential(
            ResidualBlock2D(32, 32, stride=1, norm_fn='batch'),
            ResidualBlock2D(32, base_channels, stride=1, norm_fn='batch'),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 共享层1: 从 [B, 32, H, W] 下采样到 [B, 48, H/2, W/2]
        self.shared_layer_1 = ResidualBlock2D(32, 48, stride=2, norm_fn='batch')
        # 分支2: 1/2分辨率 - 从 [B, 48, H/2, W/2] 处理到 [B, base_channels, H/2, W/2]
        self.branch_2x = nn.Sequential(
            ResidualBlock2D(48, base_channels, stride=1, norm_fn='batch'),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 共享层2: 从 [B, 48, H/2, W/2] 下采样到 [B, 64, H/4, W/4]
        self.shared_layer_2 = ResidualBlock2D(48, 64, stride=2, norm_fn='batch')
        # 分支3: 1/4分辨率 - 从 [B, 64, H/4, W/4] 处理到 [B, base_channels, H/4, W/4]
        self.branch_4x = nn.Sequential(
            ResidualBlock2D(64, base_channels, stride=1, norm_fn='batch'),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 共享层3: 从 [B, 64, H/4, W/4] 下采样到 [B, 64, H/8, W/8]
        self.shared_layer_3 = ResidualBlock2D(64, 64, stride=2, norm_fn='batch')
        # 分支4: 1/8分辨率 - 从 [B, 64, H/8, W/8] 处理到 [B, base_channels, H/8, W/8]
        self.branch_8x = nn.Sequential(
            ResidualBlock2D(64, base_channels, stride=1, norm_fn='batch'),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # 输出卷积
        self.conv_out_1x = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.conv_out_2x = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.conv_out_4x = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        self.conv_out_8x = nn.Conv2d(base_channels, base_channels, kernel_size=1)

    def forward(self, x):
        """
        输入: x [B, C, H, W] - 单帧图像
        输出:
            feat_1x [B, base_channels, H, W]
            feat_2x [B, base_channels, H/2, W/2]
            feat_4x [B, base_channels, H/4, W/4]
            feat_8x [B, base_channels, H/8, W/8]
        """
        # 初始特征提取
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)  # [B, 32, H, W] - 保持原始分辨率

        # 分支1: 原始分辨率
        feat_1x = self.conv_out_1x(self.branch_1x(x))  # [B, base_channels, H, W]

        # 共享层1: 下采样到1/2
        x_2x = self.shared_layer_1(x)  # [B, 48, H/2, W/2]
        feat_2x = self.conv_out_2x(self.branch_2x(x_2x))  # [B, base_channels, H/2, W/2]

        # 共享层2: 下采样到1/4
        x_4x = self.shared_layer_2(x_2x)  # [B, 64, H/4, W/4]
        feat_4x = self.conv_out_4x(self.branch_4x(x_4x))  # [B, base_channels, H/4, W/4]

        # 共享层3: 下采样到1/8
        x_8x = self.shared_layer_3(x_4x)  # [B, 64, H/8, W/8]
        feat_8x = self.conv_out_8x(self.branch_8x(x_8x))  # [B, base_channels, H/8, W/8]

        return feat_1x, feat_2x, feat_4x, feat_8x