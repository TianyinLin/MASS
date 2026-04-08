import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================自制背景对齐细化模块===============================
class DynamicConvRefine(nn.Module):
    def __init__(self, in_channels, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        hidden = 32

        # 动态卷积核生成器：输入guidence，输出kernel
        # 输出通道数为 k*k*2，因为flow有2个通道（x和y），需要为每个通道生成独立的卷积核
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(in_channels, hidden,  3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, kernel_size * kernel_size * 2, 1)
        )

    def forward(self, flow, guidance):
        B, _, H, W = flow.size()
        kernels = self.kernel_gen(guidance) # 生成kernel(B, 2*k*k, H, W)

        k = self.kernel_size
        flow_unfold = F.unfold(flow, k, padding=k//2)
        flow_unfold = flow_unfold.view(B, 2, k*k, H, W)

        kernels = kernels.view(B, 2, k*k, H, W)
        refined = (kernels * flow_unfold).view(B, 2, k*k, H, W).sum(dim=2)

        return refined

class SpatialRefine(nn.Module):
    """
    空间分支：检查局部邻域运动一致性
    """
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.dynamic = DynamicConvRefine(in_channels=3, kernel_size=kernel_size)

    def forward(self, flow):
        B, _, H, W = flow.size()
        k = self.kernel_size

        # 计算邻域局部variance
        flow_unfold = F.unfold(flow, k, padding=k//2)   # (B, 2*k*k, HW)
        flow_unfold = flow_unfold.view(B, 2, k*k, H, W)
        var = flow_unfold.var(dim=2)    # B, 2, H, W
        consistency = torch.exp(-var.norm(dim=1, keepdim=True)) # (B, 1, H, W)

        guidance = torch.cat([flow, consistency], dim=1)
        refined = self.dynamic(flow, guidance)

        return refined

class TemporalRefine(nn.Module):
    """
    时序分支：基于前后一致性
    """
    def __init__(self):
        super().__init__()
        self.dynamic = DynamicConvRefine(in_channels=3, kernel_size=5)

    def warp_with_flow(self, flow_to_warp, flow):
        """
        Warp a flow field using another flow field

        Args:
            flow_to_warp: [B, 2, H, W] - flow field to be warped
            flow: [B, 2, H, W] - flow field used for warping
        Returns:
            warped_flow: [B, 2, H, W]
        """
        B, _, H, W = flow_to_warp.shape
        device = flow_to_warp.device

        # 生成坐标网格（兼容不同版本的 PyTorch）
        try:
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij'
            )
        except TypeError:
            # 旧版本 PyTorch 不支持 indexing 参数
            y_coords, x_coords = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device)
            )
            # 旧版本默认是 'xy' 模式，需要转置
            y_coords, x_coords = y_coords.t(), x_coords.t()

        grid = torch.stack((x_coords, y_coords), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)

        # flow normalize to [-1, 1] range
        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0] = flow[:, 0] / (W / 2)
        flow_norm[:, 1] = flow[:, 1] / (H / 2)

        sample_grid = (grid + flow_norm).permute(0, 2, 3, 1)  # (B, H, W, 2)
        return F.grid_sample(flow_to_warp, sample_grid, align_corners=True)

    def forward(self, v0_2, v2_0):
        v2_0_warp = self.warp_with_flow(v2_0, v0_2) # warp backward flow to frame0

        fb = v0_2 + v2_0_warp   # B, 2, H, W
        cons = fb.norm(dim=1, keepdim=True) # B, 1, H, W
        w  = torch.exp(-cons)

        guidance = torch.cat([fb, w], dim=1)    # B, 3, H, W
        refined = self.dynamic(v0_2, guidance)

        return refined

class GateFuse(nn.Module):
    # 像素级gate
    def __init__(self, channels=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, v_s, v_t):
        g = self.gate(torch.cat([v_s, v_t], dim=1))
        gs = g[:, 0:1]
        gt = g[:, 1:2]
        return gs*v_s + gt*v_t

class FlowRefine(nn.Module):
    def __init__(self):
        super().__init__()
        self.Local_spatial = SpatialRefine()
        self.Temporal = TemporalRefine()
        self.fuse = GateFuse(channels=4)

    def forward(self, v0_2, v2_0):
        v_s = self.Local_spatial(v0_2)
        v_t = self.Temporal(v0_2, v2_0)
        refined_final = self.fuse(v_s, v_t)
        return refined_final