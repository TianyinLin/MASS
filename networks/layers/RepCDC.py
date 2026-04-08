import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

# ================= Conv3d + BN 融合 =================
def fuse_conv_bn_3d(conv: nn.Conv3d, bn: nn.BatchNorm3d):
    W = conv.weight
    if conv.bias is None:
        bias = torch.zeros(W.size(0), device=W.device)
    else:
        bias = conv.bias
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    W_fused = W * (gamma / std).reshape(-1, 1, 1, 1, 1)
    b_fused = beta + (bias - mean) * gamma / std
    return W_fused, b_fused

# ================= Spatial SD for 3D kernel (only H,W) =================
def spatial_sd_3d(weight: torch.Tensor):
    """
    weight: [out, in, T, kH, kW]
    输出 sd_w: same shape
    """
    sd_w = -weight.clone()
    sum_hw = weight.sum(dim=(3, 4), keepdim=True)  # [out,in,T,1,1]
    sd_w[:, :, :, weight.size(3)//2, weight.size(4)//2] = sum_hw[:, :, :, 0, 0]
    return sd_w

def spatial_sd_to_dense_3x5x5(weight: torch.Tensor):
    """
    weight: [out, in, 3, kH, kW], kH,kW = 3 or 5
    输出 dense: [out, in, 3, 5, 5]
    """
    out_c, in_c, T, kH, kW = weight.shape
    device = weight.device
    dense = torch.zeros((out_c, in_c, T, 5, 5), device=device, dtype=weight.dtype)

    cH_src, cW_src = kH // 2, kW // 2
    cH_dst, cW_dst = 2, 2  # center of 5x5

    sd = spatial_sd_3d(weight)

    for t in range(T):
        dense[:, :, t,
              cH_dst - cH_src:cH_dst + cH_src + 1,
              cW_dst - cW_src:cW_dst + cW_src + 1] = sd[:, :, t]
    return dense

# ================= RepConv3D_SpatialTemporal_v3 =================
class RepConv3D_SpatialTemporal_v3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups

        if deploy:
            self.conv_reparam = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(3, 5, 5),
                stride=(1, stride, stride),
                padding=(1, 2, 2),
                groups=groups,
                bias=True
            )
        else:
            # center branch
            self.conv_center = nn.Conv3d(in_channels, out_channels, kernel_size=(3,1,1),
                                         stride=(1,stride,stride), padding=(1,0,0),
                                         groups=groups, bias=False)
            self.bn_center = nn.BatchNorm3d(out_channels)

            # local SD branch
            self.conv_local = nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3),
                                        stride=(1,stride,stride), padding=(1,1,1),
                                        groups=groups, bias=False)
            self.bn_local = nn.BatchNorm3d(out_channels)

            # global SD branch
            self.conv_global = nn.Conv3d(in_channels, out_channels, kernel_size=(3,5,5),
                                         stride=(1,stride,stride), padding=(1,2,2),
                                         groups=groups, bias=False)
            self.bn_global = nn.BatchNorm3d(out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv_reparam(x))
        out = (self.bn_center(self.conv_center(x)) +
               self.bn_local(self._sd_forward(self.conv_local, x)) +
               self.bn_global(self._sd_forward(self.conv_global, x)))
        return self.act(out)

    @staticmethod
    def _sd_forward(conv, x):
        sd_w = spatial_sd_3d(conv.weight)
        return F.conv3d(x, sd_w, None, stride=conv.stride,
                        padding=conv.padding, dilation=conv.dilation,
                        groups=conv.groups)

    def switch_to_deploy(self):
        if self.deploy:
            return

        # fuse conv+bn
        w_c, b_c = fuse_conv_bn_3d(self.conv_center, self.bn_center)
        w_l, b_l = fuse_conv_bn_3d(self.conv_local, self.bn_local)
        w_g, b_g = fuse_conv_bn_3d(self.conv_global, self.bn_global)

        # convert to dense 5x5 spatial kernel
        w_c_5 = torch.zeros_like(w_g)
        w_c_5[:, :, :, 2, 2] = w_c[:, :, :, 0, 0]
        w_l_5 = spatial_sd_to_dense_3x5x5(w_l)
        w_g_5 = spatial_sd_to_dense_3x5x5(w_g)

        # merge
        kernel = w_c_5 + w_l_5 + w_g_5
        bias = b_c + b_l + b_g

        self.conv_reparam = nn.Conv3d(
            self.in_channels, self.out_channels,
            kernel_size=(3,5,5),
            stride=(1,self.stride,self.stride),
            padding=(1,2,2),
            groups=self.groups,
            bias=True
        )
        self.conv_reparam.weight.data = kernel
        self.conv_reparam.bias.data = bias

        # delete training branches
        del self.conv_center, self.bn_center
        del self.conv_local, self.bn_local
        del self.conv_global, self.bn_global

        self.deploy = True

# ================= RepConv3D_SpatialOnly_v3 (1x5x5 kernel) =================
class RepConv3D_SpatialOnly_v3(nn.Module):
    """
    与 RepConv3D_SpatialTemporal_v3 类似，但时序维度 kernel size 为 1（即 1x5x5）
    不进行时序卷积，仅做空间多分支重参数化
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups

        if deploy:
            self.conv_reparam = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(1, 5, 5),
                stride=(1, stride, stride),
                padding=(0, 2, 2),
                groups=groups,
                bias=True
            )
        else:
            # center branch: 1x1x1
            self.conv_center = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1),
                                         stride=(1, stride, stride), padding=(0, 0, 0),
                                         groups=groups, bias=False)
            self.bn_center = nn.BatchNorm3d(out_channels)

            # local SD branch: 1x3x3
            self.conv_local = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                                        stride=(1, stride, stride), padding=(0, 1, 1),
                                        groups=groups, bias=False)
            self.bn_local = nn.BatchNorm3d(out_channels)

            # global SD branch: 1x5x5
            self.conv_global = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5),
                                         stride=(1, stride, stride), padding=(0, 2, 2),
                                         groups=groups, bias=False)
            self.bn_global = nn.BatchNorm3d(out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv_reparam(x))
        out = (self.bn_center(self.conv_center(x)) +
               self.bn_local(self._sd_forward(self.conv_local, x)) +
               self.bn_global(self._sd_forward(self.conv_global, x)))
        return self.act(out)

    @staticmethod
    def _sd_forward(conv, x):
        """空间差分前向（与 RepConv3D_SpatialTemporal_v3 相同）"""
        sd_w = spatial_sd_3d(conv.weight)
        return F.conv3d(x, sd_w, None, stride=conv.stride,
                        padding=conv.padding, dilation=conv.dilation,
                        groups=conv.groups)

    def switch_to_deploy(self):
        if self.deploy:
            return

        # fuse conv+bn
        w_c, b_c = fuse_conv_bn_3d(self.conv_center, self.bn_center)
        w_l, b_l = fuse_conv_bn_3d(self.conv_local, self.bn_local)
        w_g, b_g = fuse_conv_bn_3d(self.conv_global, self.bn_global)

        # convert to dense 5x5 spatial kernel (T=1)
        w_c_5 = torch.zeros_like(w_g)
        w_c_5[:, :, :, 2, 2] = w_c[:, :, :, 0, 0]
        w_l_5 = self._spatial_sd_to_dense_1x5x5(w_l)
        w_g_5 = self._spatial_sd_to_dense_1x5x5(w_g)

        # merge
        kernel = w_c_5 + w_l_5 + w_g_5
        bias = b_c + b_l + b_g

        self.conv_reparam = nn.Conv3d(
            self.in_channels, self.out_channels,
            kernel_size=(1, 5, 5),
            stride=(1, self.stride, self.stride),
            padding=(0, 2, 2),
            groups=self.groups,
            bias=True
        )
        self.conv_reparam.weight.data = kernel
        self.conv_reparam.bias.data = bias

        # delete training branches
        del self.conv_center, self.bn_center
        del self.conv_local, self.bn_local
        del self.conv_global, self.bn_global

        self.deploy = True

    @staticmethod
    def _spatial_sd_to_dense_1x5x5(weight: torch.Tensor):
        """
        weight: [out, in, 1, kH, kW], kH,kW = 3 or 5
        输出 dense: [out, in, 1, 5, 5]
        """
        out_c, in_c, T, kH, kW = weight.shape
        device = weight.device
        dense = torch.zeros((out_c, in_c, T, 5, 5), device=device, dtype=weight.dtype)

        cH_src, cW_src = kH // 2, kW // 2
        cH_dst, cW_dst = 2, 2  # center of 5x5

        sd = spatial_sd_3d(weight)

        for t in range(T):
            dense[:, :, t,
                  cH_dst - cH_src:cH_dst + cH_src + 1,
                  cW_dst - cW_src:cW_dst + cW_src + 1] = sd[:, :, t]
        return dense


class CConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups

        if deploy:
            self.conv_reparam = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(3, 5, 5),
                stride=(1, stride, stride),
                padding=(1, 2, 2),
                groups=groups,
                bias=True
            )
        else:
            # center branch
            self.conv_center = nn.Conv3d(in_channels, out_channels, kernel_size=(3,1,1),
                                         stride=(1,stride,stride), padding=(1,0,0),
                                         groups=groups, bias=False)
            self.bn_center = nn.BatchNorm3d(out_channels)

            # local SD branch
            self.conv_local = nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3),
                                        stride=(1,stride,stride), padding=(1,1,1),
                                        groups=groups, bias=False)
            self.bn_local = nn.BatchNorm3d(out_channels)

            # global SD branch
            self.conv_global = nn.Conv3d(in_channels, out_channels, kernel_size=(3,5,5),
                                         stride=(1,stride,stride), padding=(1,2,2),
                                         groups=groups, bias=False)
            self.bn_global = nn.BatchNorm3d(out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv_reparam(x))
        out = (self.bn_center(self.conv_center(x)) +
               self.bn_local(self.conv_local(x)) +
               self.bn_global(self.conv_global(x)))
        return self.act(out)


    def switch_to_deploy(self):
        if self.deploy:
            return

        # fuse conv+bn
        w_c, b_c = fuse_conv_bn_3d(self.conv_center, self.bn_center)
        w_l, b_l = fuse_conv_bn_3d(self.conv_local, self.bn_local)
        w_g, b_g = fuse_conv_bn_3d(self.conv_global, self.bn_global)

        # convert to dense 5x5 spatial kernel
        w_c_5 = torch.zeros_like(w_g)
        w_c_5[:, :, :, 2, 2] = w_c[:, :, :, 0, 0]
        w_l_5 = spatial_sd_to_dense_3x5x5(w_l)
        w_g_5 = spatial_sd_to_dense_3x5x5(w_g)

        # merge
        kernel = w_c_5 + w_l_5 + w_g_5
        bias = b_c + b_l + b_g

        self.conv_reparam = nn.Conv3d(
            self.in_channels, self.out_channels,
            kernel_size=(3,5,5),
            stride=(1,self.stride,self.stride),
            padding=(1,2,2),
            groups=self.groups,
            bias=True
        )
        self.conv_reparam.weight.data = kernel
        self.conv_reparam.bias.data = bias

        # delete training branches
        del self.conv_center, self.bn_center
        del self.conv_local, self.bn_local
        del self.conv_global, self.bn_global

        self.deploy = True
