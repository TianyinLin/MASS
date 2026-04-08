import torch
import torch.nn as nn
import torch.nn.functional as F

def coords_grid(b, h, w, device=None):
    """生成坐标网格"""
    try:
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    except TypeError:
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    grid = torch.stack([x, y], dim=0).float()
    grid = grid[None].repeat(b, 1, 1, 1)
    return grid

def bilinear_sample(feature, sample_coords, padding_mode='zeros'):
    """双线性采样"""
    b, c, h, w = feature.size()
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1)
    return F.grid_sample(feature, grid, mode='bilinear', padding_mode=padding_mode, align_corners=True)


# --------- Hierarchical Flow Refinement Modules ---------
class FlowRefinementBlock(nn.Module):
    """
    在某一分辨率上的 refinement block。
    输入：
        feat: 当前分辨率的特征  [B, C, H, W]
        feat_ref: 参考帧在同一分辨率的特征 [B, C, H, W]
        coarse_flow: 上一层上采样来的粗光流 [B, 2, H, W]
    输出：
        refined_flow = coarse_flow + delta_flow  (B,2,H,W)
    """

    def __init__(self, in_channels, hidden=64):
        super(FlowRefinementBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1)  # predict delta flow (dx,dy)
        )

    def forward(self, feat, feat_ref, coarse_flow):
        """
        feat: target frame feature at this scale (e.g. feat2 at scale)
        feat_ref: reference frame feature at this scale (the other frame we want to align from)
        coarse_flow: upsampled coarse flow [B,2,H,W]
        """
        b, c, h, w = feat.size()
        # coords grid + coarse_flow -> normalized sampling coords for grid_sample
        coords = coords_grid(b, h, w, device=coarse_flow.device) + coarse_flow  # [B,2,H,W]
        feat_ref_warp = bilinear_sample(feat_ref, coords, padding_mode='zeros')  # [B,C,H,W]

        x = torch.cat([feat, feat_ref_warp], dim=1)  # [B, 2C, H, W]
        # print(x.shape)
        delta = self.net(x)  # [B,2,H,W]
        # small init scale to stabilize training
        return coarse_flow + 0.1 * delta




class HierarchicalFlowRefiner(nn.Module):
    """
    使用多个 FlowRefinementBlock 在每个分辨率上逐步 refine。
    设计与调用约定（与你模型一致）：
      - 输入 initial_flow_8x: 在 1/8 分辨率估计得到的粗 flow (B,2,h8,w8)
      - feats: dict 包含 features at different scales: { '8x': (feat0_8, feat1_8, feat2_8), '4x':..., '2x':..., '1x':... }
      - 返回 refined flows at each scale：flow_8x, flow_4x, flow_2x, flow_1x
    """

    def __init__(self, channels_map, upsample_mode='bilinear'):
        """
        channels_map: dict，指定每层特征通道数，例如 {'8x':64, '4x':64, '2x':64, '1x':64}
        """
        super(HierarchicalFlowRefiner, self).__init__()
        self.upsample_mode = upsample_mode
        self.refine_4x = FlowRefinementBlock(in_channels=channels_map['4x'])
        self.refine_2x = FlowRefinementBlock(in_channels=channels_map['2x'])
        self.refine_1x = FlowRefinementBlock(in_channels=channels_map['1x'])

    def _upsample_to(self, flow, size):
        # flow: [B,2,h,w] -> upsample to size (H,W)
        # align_corners True to be consistent with bilinear_sample usage
        return F.interpolate(flow, size=size, mode='bilinear', align_corners=True)

    def forward(self, initial_flow_8x, feats, ref_idx=0):
        """
        feats: dict: keys '8x','4x','2x','1x', each value: tuple (feat0, feat1, feat2) for the three frames
        initial_flow_8x: the coarse flow computed on 8x resolution  (B,2,h8,w8)
        ref_idx: 0 for feat0 as reference, 1 for feat1 as reference
        Returns:
            flow_8x (same as initial)
            flow_4x, flow_2x, flow_1x
        """
        flow_8x = initial_flow_8x  # coarse
        # refine at 4x
        feat0_4, feat1_4, feat2_4 = feats['4x']
        feat_ref_4 = feat0_4 if ref_idx == 0 else feat1_4
        b, _, h4, w4 = feat2_4.shape
        flow_4_coarse = self._upsample_to(flow_8x, (h4, w4))
        # use feat2 as target and feat_ref as reference for background alignment
        flow_4 = self.refine_4x(feat2_4, feat_ref_4, flow_4_coarse)

        # refine at 2x
        feat0_2, feat1_2, feat2_2 = feats['2x']
        feat_ref_2 = feat0_2 if ref_idx == 0 else feat1_2
        b, _, h2, w2 = feat2_2.shape
        flow_2_coarse = self._upsample_to(flow_4, (h2, w2))
        flow_2 = self.refine_2x(feat2_2, feat_ref_2, flow_2_coarse)

        # refine at 1x
        feat0_1, feat1_1, feat2_1 = feats['1x']
        feat_ref_1 = feat0_1 if ref_idx == 0 else feat1_1
        b, _, h1, w1 = feat2_1.shape
        flow_1_coarse = self._upsample_to(flow_2, (h1, w1))
        flow_1 = self.refine_1x(feat2_1, feat_ref_1, flow_1_coarse)

        return flow_8x, flow_4, flow_2, flow_1