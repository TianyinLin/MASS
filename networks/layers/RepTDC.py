import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 1. TDifferenceConv（时间差分卷积）
# ============================================================
class TDifferenceConv(nn.Module):
    """
    Temporal Difference Convolution
    diff_weight[t] = 2*W[t] - W[t-1] - W[t+1]
    Boundary: replicate
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 1, 1),
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size)
        )
        self.bias = None if not bias else nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def temporal_difference(self):
        """
        返回 TD 之后的真实卷积核
        """
        w = self.weight                      # [O, I, T, 1, 1]
        w_prev = torch.roll(w, 1, dims=2)
        w_next = torch.roll(w, -1, dims=2)

        # 边界复制（与 min/max 等价）
        w_prev[:, :, 0] = w[:, :, 0]
        w_next[:, :, -1] = w[:, :, -1]

        return 2 * w - w_prev - w_next

    def forward(self, x):
        w_td = self.temporal_difference()
        return F.conv3d(
            x,
            w_td,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )


# ============================================================
# 2. Conv3d + BN 融合
# ============================================================
def fuse_conv_bn_3d(weight, bn: nn.BatchNorm3d):
    """
    weight: [O, I, T, H, W]
    """
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    weight_fused = weight * (gamma / std).reshape(-1, 1, 1, 1, 1)
    bias_fused = beta - gamma * mean / std
    return weight_fused, bias_fused


# ============================================================
# 3. dilation kernel (3) -> dense kernel (7)
# ============================================================
def dilated_3_to_7(kernel_3, dilation):
    """
    kernel_3: [O, I, 3, 1, 1]
    return:   [O, I, 7, 1, 1]
    """
    out_c, in_c, _, _, _ = kernel_3.shape
    device = kernel_3.device
    kernel_7 = torch.zeros((out_c, in_c, 7, 1, 1), device=device)

    center = 3
    for i in range(3):
        kernel_7[:, :, center + (i - 1) * dilation, 0, 0] = kernel_3[:, :, i, 0, 0]

    return kernel_7


# ============================================================
# 4. TD-RepConv3D-Dilated（训练 / 推理双态）
# ============================================================
class TDRepConv3D_Dilated(nn.Module):
    """
    Train:
        3 branches:
            TDConv(k=3, d=1) + BN
            TDConv(k=3, d=2) + BN
            TDConv(k=3, d=3) + BN
        Sum + ReLU

    Deploy:
        Single Conv3D(k=7, d=1)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 groups=1,
                 deploy=False):
        super().__init__()

        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride

        if deploy:
            self.conv_reparam = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(7, 1, 1),
                stride=(stride, 1, 1),
                padding=(3, 0, 0),
                groups=groups,
                bias=True
            )
        else:
            # ===== dilation = 1 =====
            self.branch_d1 = TDifferenceConv(
                in_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                dilation=(1, 1, 1),
                groups=groups
            )
            self.bn_d1 = nn.BatchNorm3d(out_channels)

            # ===== dilation = 2 =====
            self.branch_d2 = TDifferenceConv(
                in_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(2, 0, 0),
                dilation=(2, 1, 1),
                groups=groups
            )
            self.bn_d2 = nn.BatchNorm3d(out_channels)

            # ===== dilation = 3 =====
            self.branch_d3 = TDifferenceConv(
                in_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                padding=(3, 0, 0),
                dilation=(3, 1, 1),
                groups=groups
            )
            self.bn_d3 = nn.BatchNorm3d(out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv_reparam(x))

        out = (
            self.bn_d1(self.branch_d1(x)) +
            self.bn_d2(self.branch_d2(x)) +
            self.bn_d3(self.branch_d3(x))
        )
        return self.act(out)


    def switch_to_deploy(self):
        if self.deploy:
            return

        kernels = []
        biases = []

        branches = [
            (self.branch_d1, self.bn_d1, 1),
            (self.branch_d2, self.bn_d2, 2),
            (self.branch_d3, self.bn_d3, 3),
        ]

        for branch, bn, dilation in branches:
            # 1) TD -> 实际卷积核
            k_td = branch.temporal_difference()

            # 2) BN 融合
            k_fused, b_fused = fuse_conv_bn_3d(k_td, bn)

            # 3) dilation -> dense 7
            k_7 = dilated_3_to_7(k_fused, dilation)

            kernels.append(k_7)
            biases.append(b_fused)

        kernel_final = sum(kernels)
        bias_final = sum(biases)

        # 创建推理卷积
        self.conv_reparam = nn.Conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(7, 1, 1),
            stride=(self.stride, 1, 1),
            padding=(3, 0, 0),
            groups=self.groups,
            bias=True
        )

        self.conv_reparam.weight.data = kernel_final
        self.conv_reparam.bias.data = bias_final

        # 删除训练分支
        del self.branch_d1, self.bn_d1
        del self.branch_d2, self.bn_d2
        del self.branch_d3, self.bn_d3

        self.deploy = True


class Rep3DT(nn.Module):
    """
    普通 Conv 的时间多尺度 Rep 版本（用于和 TD 对比）
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 groups=1,
                 deploy=False):
        super().__init__()

        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride

        if deploy:
            self.conv_reparam = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(7, 1, 1),
                stride=(stride, 1, 1),
                padding=(3, 0, 0),
                groups=groups,
                bias=True
            )
        else:
            self.branch_d1 = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                dilation=(1, 1, 1),
                groups=groups,
                bias=False
            )
            self.bn_d1 = nn.BatchNorm3d(out_channels)

            self.branch_d2 = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(3, 1, 1),
                padding=(2, 0, 0),
                dilation=(2, 1, 1),
                groups=groups,
                bias=False
            )
            self.bn_d2 = nn.BatchNorm3d(out_channels)

            self.branch_d3 = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(3, 1, 1),
                padding=(3, 0, 0),
                dilation=(3, 1, 1),
                groups=groups,
                bias=False
            )
            self.bn_d3 = nn.BatchNorm3d(out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv_reparam(x))

        out = (
            self.bn_d1(self.branch_d1(x)) +
            self.bn_d2(self.branch_d2(x)) +
            self.bn_d3(self.branch_d3(x))
        )
        return self.act(out)

    def switch_to_deploy(self):
        if self.deploy:
            return

        kernels = []
        biases = []

        branches = [
            (self.branch_d1, self.bn_d1, 1),
            (self.branch_d2, self.bn_d2, 2),
            (self.branch_d3, self.bn_d3, 3),
        ]

        for conv, bn, dilation in branches:
            # 1) Conv + BN 融合
            w = conv.weight
            w_fused, b_fused = fuse_conv_bn_3d(w, bn)

            # 2) dilation -> dense 7
            w_7 = dilated_3_to_7(w_fused, dilation)

            kernels.append(w_7)
            biases.append(b_fused)

        kernel_final = sum(kernels)
        bias_final = sum(biases)

        self.conv_reparam = nn.Conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(7, 1, 1),
            stride=(self.stride, 1, 1),
            padding=(3, 0, 0),
            groups=self.groups,
            bias=True
        )

        self.conv_reparam.weight.data = kernel_final
        self.conv_reparam.bias.data = bias_final

        del self.branch_d1, self.bn_d1
        del self.branch_d2, self.bn_d2
        del self.branch_d3, self.bn_d3

        self.deploy = True



# ============================================================
# 6. 简单测试（可选）
# ============================================================
if __name__ == "__main__":
    x = torch.randn(1, 32, 10, 64, 64)

    model = TDRepConv3D_Dilated(32, 32, deploy=False)
    model.eval()
    with torch.no_grad():
        y_train = model(x)

    model.switch_to_deploy()
    model.eval()
    with torch.no_grad():
        y_deploy = model(x)

    print("Max diff:", (y_train - y_deploy).abs().max().item())
