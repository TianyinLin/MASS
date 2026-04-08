"""
背景对齐模块
包含BackgroundAlignmentModule及其相关组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks.layers.FlowRefine import FlowRefine

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


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    """生成局部窗口网格"""
    try:
        x, y = torch.meshgrid(
            torch.linspace(w_min, w_max, len_w, device=device),
            torch.linspace(h_min, h_max, len_h, device=device),
            indexing='xy'
        )
    except TypeError:
        x, y = torch.meshgrid(
            torch.linspace(w_min, w_max, len_w, device=device),
            torch.linspace(h_min, h_max, len_h, device=device)
        )
        x, y = x.t(), y.t()
    grid = torch.stack([x, y], dim=-1).float()
    return grid


def normalize_coords(coords, h, w):
    """归一化坐标到[-1, 1]"""
    c = torch.tensor([(w - 1) / 2., (h - 1) / 2.], device=coords.device, dtype=torch.float32)
    return (coords - c) / c


def feature_warp(feature, flow, padding_mode='zeros'):
    """特征warp操作"""
    b, c, h, w = feature.size()
    assert flow.size(1) == 2
    grid = coords_grid(b, h, w, device=flow.device) + flow
    return bilinear_sample(feature, grid, padding_mode=padding_mode)

def local_correlation_softmax(feature0, feature1, local_radius, padding_mode='zeros'):
    """局部相关性计算（softmax）"""
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w, device=feature0.device)
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)
    window_grid = window_grid.reshape(-1, 2).unsqueeze(0).unsqueeze(0).repeat(b, 1, 1, 1)
    sample_coords = coords.unsqueeze(-2) + window_grid
    sample_coords_softmax = sample_coords
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)
    valid = valid_x & valid_y
    sample_coords_norm = normalize_coords(sample_coords, h, w)
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode=padding_mode, align_corners=True
                                   ).permute(0, 2, 1, 3)
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)
    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)
    corr[~valid] = -1e9
    prob = F.softmax(corr, -1)
    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2)
    v = correspondence - coords_init
    match_prob = prob
    return v, match_prob

# ==================== 背景对齐模块 ====================
class BackgroundAlignmentModule(nn.Module):
    """背景对齐模块 - 计算背景运动分量并对齐特征"""

    def __init__(self, feature_channels=64, local_radius=2):
        super(BackgroundAlignmentModule, self).__init__()
        self.feature_channels = feature_channels
        self.local_radius = local_radius
        self.ultimate_refiner = FlowRefine()

    def align_features(self, feature0, feature1, feature2, v0_2, v1_2):
        """对齐特征"""
        f0_warp = feature_warp(feature0, v0_2)
        f1_warp = feature_warp(feature1, v1_2)
        return f0_warp, f1_warp

    def compute_refined_vector(self, feature0, feature1, feature2):
        v_0_2, _ = local_correlation_softmax(feature0, feature2, self.local_radius)
        v_1_2, _ = local_correlation_softmax(feature1, feature2, self.local_radius)

        v_2_0, _ = local_correlation_softmax(feature2, feature0, self.local_radius)
        v_2_1, _ = local_correlation_softmax(feature2, feature1, self.local_radius)

        v0_refined = self.ultimate_refiner(v_0_2.detach(), v_2_0.detach())
        v1_refined = self.ultimate_refiner(v_1_2.detach(), v_2_1.detach())
        # v0_refined = v_0_2.detach()
        # v1_refined = v_1_2.detach()
        return v0_refined, v1_refined



