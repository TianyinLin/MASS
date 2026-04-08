import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks.layers.basic import DSCIM
from networks.layers.TD_RepConv import TDRepConv3D_Dilated
from networks.layers.HybridTPro import DeTTA
from networks.layers.FrameDifferenceModule import FrameDifferenceModule
from networks.layers.background_align import BackgroundAlignmentModule
from networks.losses.Motion_loss import Motion_loss
from networks.layers.MultiScaleFeatureExtractor import MultiScaleFeatureExtractor
from networks.layers.HierarchicalFlowRefiner import HierarchicalFlowRefiner


# ==================== 改进的detector ====================
class detector(nn.Module):
    def __init__(self, num_classes, seqlen=10, out_len=10,
                 feature_channels=64, local_radius=2, attn_splits=1, prop_radius=-1):
        super(detector, self).__init__()
        self.out_len = out_len
        self.seqlen = seqlen
        self.feature_channels = feature_channels

        # ========== 改进的特征提取网络 ==========
        self.feature_extractor = MultiScaleFeatureExtractor(
            input_channels=1,
            base_channels=feature_channels,
            norm_fn='batch'
        )

        # -------- Hierarchical flow refiner (v0 and v1) --------
        channels_map = {'8x': feature_channels, '4x': feature_channels, '2x': feature_channels, '1x': feature_channels}
        self.flow_refiner_v0 = HierarchicalFlowRefiner(channels_map)
        self.flow_refiner_v1 = HierarchicalFlowRefiner(channels_map)
        # -------------------------------------------------------

        # ========== 背景对齐模块 ==========
        self.background_align = BackgroundAlignmentModule(
            feature_channels=feature_channels,
            local_radius=local_radius,
            attn_splits=attn_splits,
            prop_radius=prop_radius
        )

        # ========== 帧差分模块（每个分辨率一个）==========
        # 用于在差分后进行归一化和ReLU操作
        self.frame_diff_1x = FrameDifferenceModule(feature_channels)
        self.frame_diff_2x = FrameDifferenceModule(feature_channels)
        self.frame_diff_4x = FrameDifferenceModule(feature_channels)
        self.frame_diff_8x = FrameDifferenceModule(feature_channels)

        # 通道适配：将feature_channels降维到8，以便后续STD_Resblock处理
        self.channel_adapter1 = nn.Sequential(
            nn.Conv3d(feature_channels, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.channel_adapter2 = nn.Sequential(
            nn.Conv3d(feature_channels, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.channel_adapter3 = nn.Sequential(
            nn.Conv3d(feature_channels, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.channel_adapter4 = nn.Sequential(
            nn.Conv3d(feature_channels, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            TDRepConv3D_Dilated(32, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            TDRepConv3D_Dilated(32, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            TDRepConv3D_Dilated(32, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            TDRepConv3D_Dilated(32, 32),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # 各层的STD_Resblock
        self.layer1_block1 = DSCIM(32, 32)
        self.layer2_block1 = DSCIM(32, 32)
        self.layer3_block1 = DSCIM(32, 32)
        self.layer4_block1 = DSCIM(32, 32)

        # TPro时序建模
        self.TPro1 = DeTTA(d_model=32, num_head=4, seqlen=seqlen)
        self.conv_out1_1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TPro2 = DeTTA(d_model=32, num_head=4, seqlen=seqlen)
        self.conv_out1_2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TPro3 = DeTTA(d_model=32, num_head=4, seqlen=seqlen)
        self.conv_out1_3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TPro4 = DeTTA(d_model=32, num_head=4, seqlen=seqlen)
        self.conv_out1_4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True))

        self.conv_out2 = nn.Sequential(
            nn.Conv3d(in_channels=8 * 4, out_channels=8, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            # 改为4个分支
            nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.final = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                               padding=(0, 0, 0))
        # 辅助输出
        self.aux_head1 = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=1)
        self.aux_head2 = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=1)
        self.aux_head3 = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=1)
        self.aux_head4 = nn.Conv3d(in_channels=8, out_channels=num_classes, kernel_size=1)

    def _get_frame_indices(self, t):
        """获取三帧索引（辅助函数）"""
        if t == 0:
            return 0, 0, 0
        elif t == 1:
            return 0, 0, 1
        else:
            return t - 2, t - 1, t

    def _get_frame_indices_vectorized(self, T, device):
        """向量化三帧索引：返回 (idx0, idx1, idx2)，每个 shape (T,)，与逐帧 _get_frame_indices 严格等价。"""
        # t=0 -> (0,0,0), t=1 -> (0,0,1), t>=2 -> (t-2, t-1, t)
        idx2 = torch.arange(T, device=device)
        idx1 = torch.clamp(idx2 - 1, min=0)
        idx0 = torch.clamp(idx2 - 2, min=0)
        return idx0, idx1, idx2

    def _align_and_diff(self, feat0, feat1, feat2, v0_2, v1_2, diff_module, return_warped=False):
        """
        对齐特征并计算差分（辅助函数）

        Args:
            feat0, feat1, feat2: 输入特征图 [B, C, H, W]
            v0_2, v1_2: 运动向量 [B, 2, H, W]
            diff_module: FrameDifferenceModule，用于差分、归一化和ReLU
            return_warped: 是否返回对齐后的特征图（用于Motion Loss）

        Returns:
            如果return_warped=False: 返回差分特征 [B, C, H, W]（已归一化和ReLU）
            如果return_warped=True: 返回(差分特征, f0_warp, f1_warp)

        差分计算：使用FrameDifferenceModule实现类似TDifferenceConv的操作
        - 堆叠 f0_warp, f1_warp, feat2 成 [B, C, 3, H, W]
        - 应用差分卷积: 2 * feat2 - f0_warp - f1_warp
        - BatchNorm + ReLU
        """
        f0_warp, f1_warp = self.background_align.align_features(feat0, feat1, feat2, v0_2, v1_2)

        # 使用FrameDifferenceModule进行差分、归一化和ReLU
        diff_feat = diff_module(f0_warp, f1_warp, feat2)

        if return_warped:
            return diff_feat, f0_warp, f1_warp
        else:
            return diff_feat

    def _upsample_flow(self, flow, hidden_feat, extractor, upsample_module):
        """上采样运动向量（辅助函数）"""
        hidden = extractor(hidden_feat)
        return upsample_module(flow, hidden)

    def forward(self, seq_imgs):
        """
        输入: seq_imgs [B, 1, T, H, W] - T=10帧

        改进：三帧三帧处理，先拼接后分割（学习MFE-Net），只保存对齐差分后的结果
        """
        B, _, T, H, W = seq_imgs.size()
        # reshape 成 backbone 接受的格式
        imgs_2d = seq_imgs.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)

        feat_1x_all, feat_2x_all, feat_4x_all, feat_8x_all = self.feature_extractor(imgs_2d)

        # reshape 回时序形式
        feat_1x_all = feat_1x_all.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
        feat_2x_all = feat_2x_all.view(B, T, -1, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        feat_4x_all = feat_4x_all.view(B, T, -1, H // 4, W // 4).permute(0, 2, 1, 3, 4)
        feat_8x_all = feat_8x_all.view(B, T, -1, H // 8, W // 8).permute(0, 2, 1, 3, 4)

        # ========== 时间维向量化 + 重参数化为 batch 维（无显式 for 循环，数学等价）==========
        device = seq_imgs.device
        idx0, idx1, idx2 = self._get_frame_indices_vectorized(T, device)

        # 一次性构造所有时间步的三帧特征 [B, C, T, H, W]
        feat0_1x_all = feat_1x_all[:, :, idx0, :, :]
        feat1_1x_all = feat_1x_all[:, :, idx1, :, :]
        feat2_1x_all = feat_1x_all[:, :, idx2, :, :]
        feat0_2x_all = feat_2x_all[:, :, idx0, :, :]
        feat1_2x_all = feat_2x_all[:, :, idx1, :, :]
        feat2_2x_all = feat_2x_all[:, :, idx2, :, :]
        feat0_4x_all = feat_4x_all[:, :, idx0, :, :]
        feat1_4x_all = feat_4x_all[:, :, idx1, :, :]
        feat2_4x_all = feat_4x_all[:, :, idx2, :, :]
        feat0_8x_all = feat_8x_all[:, :, idx0, :, :]
        feat1_8x_all = feat_8x_all[:, :, idx1, :, :]
        feat2_8x_all = feat_8x_all[:, :, idx2, :, :]

        def merge_time_to_batch(x):
            """[B, C, T, H, W] -> [B*T, C, H, W]"""
            return x.permute(0, 2, 1, 3, 4).reshape(B * T, x.size(1), x.size(3), x.size(4))

        feat0_1x_bt = merge_time_to_batch(feat0_1x_all)
        feat1_1x_bt = merge_time_to_batch(feat1_1x_all)
        feat2_1x_bt = merge_time_to_batch(feat2_1x_all)
        feat0_2x_bt = merge_time_to_batch(feat0_2x_all)
        feat1_2x_bt = merge_time_to_batch(feat1_2x_all)
        feat2_2x_bt = merge_time_to_batch(feat2_2x_all)
        feat0_4x_bt = merge_time_to_batch(feat0_4x_all)
        feat1_4x_bt = merge_time_to_batch(feat1_4x_all)
        feat2_4x_bt = merge_time_to_batch(feat2_4x_all)
        feat0_8x_bt = merge_time_to_batch(feat0_8x_all)
        feat1_8x_bt = merge_time_to_batch(feat1_8x_all)
        feat2_8x_bt = merge_time_to_batch(feat2_8x_all)

        # 在 1/8 分辨率上批量计算运动向量 [B*T, 2, h8, w8]
        v0_2_8x_bt, v1_2_8x_bt = self.background_align.compute_refined_vector(
            feat0_8x_bt, feat1_8x_bt, feat2_8x_bt
        )

        # 分层 refinement（batch 维为 B*T，时间步间无依赖，数学等价于逐帧）
        feats_bt = {
            '8x': (feat0_8x_bt, feat1_8x_bt, feat2_8x_bt),
            '4x': (feat0_4x_bt, feat1_4x_bt, feat2_4x_bt),
            '2x': (feat0_2x_bt, feat1_2x_bt, feat2_2x_bt),
            '1x': (feat0_1x_bt, feat1_1x_bt, feat2_1x_bt),
        }
        v0_2_8x_bt, v0_2_4x_bt, v0_2_2x_bt, v0_2_1x_bt = self.flow_refiner_v0(
            v0_2_8x_bt, feats_bt, ref_idx=0
        )
        v1_2_8x_bt, v1_2_4x_bt, v1_2_2x_bt, v1_2_1x_bt = self.flow_refiner_v1(
            v1_2_8x_bt, feats_bt, ref_idx=1
        )

        # 批量对齐与差分（对齐、差分在时间维上独立，等价于逐帧）
        f0_warp_1x_bt, f1_warp_1x_bt = self.background_align.align_features(
            feat0_1x_bt, feat1_1x_bt, feat2_1x_bt, v0_2_1x_bt, v1_2_1x_bt
        )
        aligned_diff_1x_bt = self.frame_diff_1x(f0_warp_1x_bt, f1_warp_1x_bt, feat2_1x_bt)

        f0_warp_2x_bt, f1_warp_2x_bt = self.background_align.align_features(
            feat0_2x_bt, feat1_2x_bt, feat2_2x_bt, v0_2_2x_bt, v1_2_2x_bt
        )
        aligned_diff_2x_bt = self.frame_diff_2x(f0_warp_2x_bt, f1_warp_2x_bt, feat2_2x_bt)

        f0_warp_4x_bt, f1_warp_4x_bt = self.background_align.align_features(
            feat0_4x_bt, feat1_4x_bt, feat2_4x_bt, v0_2_4x_bt, v1_2_4x_bt
        )
        aligned_diff_4x_bt = self.frame_diff_4x(f0_warp_4x_bt, f1_warp_4x_bt, feat2_4x_bt)

        f0_warp_8x_bt, f1_warp_8x_bt = self.background_align.align_features(
            feat0_8x_bt, feat1_8x_bt, feat2_8x_bt, v0_2_8x_bt, v1_2_8x_bt
        )
        aligned_diff_8x_bt = self.frame_diff_8x(f0_warp_8x_bt, f1_warp_8x_bt, feat2_8x_bt)

        # 恢复为时序布局 [B, C, T, H, W]
        def unmerge_batch_to_time(x_bt, c):
            """[B*T, C, H, W] -> [B, C, T, H, W]"""
            return x_bt.view(B, T, c, x_bt.size(2), x_bt.size(3)).permute(0, 2, 1, 3, 4)

        C = feat_1x_all.size(1)
        seq_feats_list = [
            unmerge_batch_to_time(aligned_diff_1x_bt, C),
            unmerge_batch_to_time(aligned_diff_2x_bt, C),
            unmerge_batch_to_time(aligned_diff_4x_bt, C),
        ]
        seq_feats_8x = unmerge_batch_to_time(aligned_diff_8x_bt, C)

        # 仅训练时构造 Motion Loss 所需列表，推理阶段不再计算与保存
        if self.training:
            warp_img_list = []
            img_list = []
            img2_per_t = seq_imgs[:, :, idx2, :, :]  # [B, 1, T, H, W]
            for t in range(T):
                warp_img_list.append(f0_warp_1x_bt[B * t : B * (t + 1)])
                warp_img_list.append(f1_warp_1x_bt[B * t : B * (t + 1)])
                img2_t = img2_per_t[:, :, t, :, :]
                img_list.append(img2_t)
                img_list.append(img2_t)
        else:
            warp_img_list = []
            img_list = []

        # 通道适配
        seq_feats1 = self.conv1(self.channel_adapter1(seq_feats_list[0]))
        seq_feats2 = self.conv2(self.channel_adapter2(seq_feats_list[1]))
        seq_feats3 = self.conv3(self.channel_adapter3(seq_feats_list[2]))
        seq_feats4 = self.conv4(self.channel_adapter4(seq_feats_8x))

        seq_feats1 = self.layer1_block1(seq_feats1, seq_feats2)
        seq_feats2 = self.layer2_block1(seq_feats1, seq_feats2, seq_feats3)
        seq_feats3 = self.layer3_block1(seq_feats2, seq_feats3, seq_feats4)
        seq_feats4 = self.layer4_block1(seq_feats3, seq_feats4)

        # ========== TPro时序建模 ==========
        tpro_modules = [self.TPro1, self.TPro2, self.TPro3, self.TPro4]
        conv_out_modules = [self.conv_out1_1, self.conv_out1_2, self.conv_out1_3, self.conv_out1_4]
        seq_feats_list = [seq_feats1, seq_feats2, seq_feats3, seq_feats4]

        for i in range(4):
            # seq_feats_list[i]: [B, 32, T, H, W]
            seq_feats_original = seq_feats_list[i]

            seq_feats_original_perm = seq_feats_original.permute(0, 3, 4, 1, 2)  # [B, H, W, 32, T]
            tpro_output = tpro_modules[i](seq_feats_original_perm)  # [B, 32, T, H, W]
            seq_feats_list[i] = conv_out_modules[i](tpro_output)

        seq_feats1, seq_feats2, seq_feats3, seq_feats4 = seq_feats_list
        b, c, t, h1, w1 = seq_feats1.size()
        _, _, _, h2, w2 = seq_feats2.size()
        _, _, _, h3, w3 = seq_feats3.size()
        _, _, _, h4, w4 = seq_feats4.size()

        # 辅助输出（包含1x分支）
        aux_heads = [self.aux_head1, self.aux_head2, self.aux_head3, self.aux_head4]
        aux_outputs = [
            F.interpolate(aux_heads[i](seq_feats_list[i]), size=(t, H, W),
                          mode="trilinear", align_corners=True)
            for i in range(4)
        ]
        aux_out1, aux_out2, aux_out3, aux_out4 = aux_outputs

        # 融合四分支输出（包含1x分支）
        sizes = [(h1, w1), (h2, w2), (h3, w3), (h4, w4)]
        seq_feats_list = [
            F.interpolate(seq_feats_list[i].reshape(b, c * t, sizes[i][0], sizes[i][1]),
                          size=(H, W), mode="bilinear", align_corners=True
                          ).reshape(b, c, t, H, W)
            for i in range(4)
        ]

        seq_feats = self.conv_out2(torch.cat(seq_feats_list, dim=1))
        seq_midout = self.final(seq_feats)
        seq_midseg = seq_midout.squeeze(dim=1)

        # 将warp_img_list和img_list用于Motion Loss（与MFE-Net一致）
        # 注意：每个时间步有2个warp特征图（feat0_warp和feat1_warp），所以列表长度为2*T
        # MFE-Net方式：warp_img只取第一通道，img是原始输入图像（已经是单通道）
        # 直接传递列表给Motion Loss，Motion Loss内部会处理（只取第一通道）
        warp_img = warp_img_list  # [B, feature_channels, H, W] 的列表
        img = img_list  # [B, 1, H, W] 的列表（原始输入图像）

        return seq_feats, seq_midseg, aux_out1, aux_out2, aux_out3, aux_out4, warp_img, img
        # return _, seq_midseg, _, _, _, _, _, _


class SoftLoUloss(nn.Module):
    def __init__(self):
        super(SoftLoUloss, self).__init__()

    def forward(self, midpred, target):
        smooth = 0.00
        midpred = torch.sigmoid(midpred)
        intersection = midpred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(midpred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss_mid = (intersection_sum + smooth) / \
                   (pred_sum + target_sum - intersection_sum + smooth)
        loss_mid = 1 - torch.mean(loss_mid)
        return loss_mid


class Motionloss(nn.Module):
    """Motion Loss包装类，用于训练"""

    def __init__(self, l1=0.85, l2=0.15):
        super(Motionloss, self).__init__()
        self.motion_loss = Motion_loss(l1=l1, l2=l2)

    def forward(self, warp_img, img):
        """
        Args:
            warp_img: [B, T, H, W] 或 [B, 1, H, W]
            img: [B, T, H, W] 或 [B, 1, H, W]
        """
        return self.motion_loss(warp_img, img)

def _flops_correction_train_mode(model, seq_imgs):
    """
    训练态下 thop 会漏算：RepConv3D_SpatialTemporal_v3 的 local/global 分支（F.conv3d）、
    TDRepConv3D_Dilated 内 TDifferenceConv 的 F.conv3d。本函数通过一次 forward + hook 收集
    各模块输出规模，按与 thop 相同的公式补算 FLOPs，返回应加上的修正量（标量）。
    """
    from networks.layers.ReDC import RepConv3D_SpatialTemporal_v3
    from networks.layers.TD_RepConv import TDRepConv3D_Dilated

    correction = 0.0
    # 与 thop 一致：每个输出元素对应 (in_c/groups) * kernel_elements 次乘加，算 1 MAC ≈ 1 FLOP（thop 用 output_size * (in_c/g) * kT*kH*kW）
    def _conv3d_flops_per_output(in_c, g, kt, kh, kw):
        return (in_c // g) * kt * kh * kw

    collected_repconv = []
    collected_tdrepconv = []

    def hook_repconv(module, inp, out):
        if getattr(module, 'deploy', True):
            return
        x = inp[0]
        y = out
        # 漏算的是 local + global 两个分支
        n = y.numel()
        inc, g = module.in_channels, module.groups
        add_flops = n * _conv3d_flops_per_output(inc, g, 3, 3, 3) + n * _conv3d_flops_per_output(inc, g, 3, 5, 5)
        collected_repconv.append(add_flops)

    def hook_tdrepconv(module, inp, out):
        if getattr(module, 'deploy', True):
            return
        y = out
        n = y.numel()
        inc, g = module.in_channels, module.groups
        add_flops = n * _conv3d_flops_per_output(inc, g, 3, 1, 1) * 3  # 3 个 (3,1,1) 分支
        collected_tdrepconv.append(add_flops)

    hooks = []
    for m in model.modules():
        if isinstance(m, RepConv3D_SpatialTemporal_v3):
            hooks.append(m.register_forward_hook(hook_repconv))
        elif isinstance(m, TDRepConv3D_Dilated):
            hooks.append(m.register_forward_hook(hook_tdrepconv))
    with torch.no_grad():
        model(seq_imgs)
    for h in hooks:
        h.remove()
    return sum(collected_repconv) + sum(collected_tdrepconv)


if __name__ == '__main__':
    import time
    import torch
    from thop import profile

    seq_imgs = torch.randn(1, 1, 10, 512, 512)

    # 1) 训练模式（RepConv3D_SpatialTemporal_v3 / TDRepConv3D_Dilated 未重参数化）
    model = detector(num_classes=1, seqlen=10, out_len=10, feature_channels=64)
    model.eval()

    params_train = sum(p.numel() for p in model.parameters())
    # 先做校正统计（仅用本脚本 hook，不含 thop），再 profile，避免 thop 与二次 forward 冲突
    flops_train_correction = _flops_correction_train_mode(model, seq_imgs)
    flops_train_raw, _ = profile(model, inputs=(seq_imgs,), verbose=False)
    flops_train = flops_train_raw + flops_train_correction
    params_train_M = params_train / 1e6
    flops_train_G = flops_train / 1e9

    print("\n" + "=" * 60)
    print("Training Mode (deploy = False)")
    print("=" * 60)
    print(f"Params : {params_train_M:.3f} M")
    print(f"FLOPs  : {flops_train_G:.3f} G  (raw thop: {flops_train_raw/1e9:.3f} G + correction: {flops_train_correction/1e9:.3f} G)")

    # 2) 推理模式：对所有 RepConv3D_SpatialTemporal_v3 / TDRepConv3D_Dilated 调用 switch_to_deploy()
    from networks.layers.ReDC import RepConv3D_SpatialTemporal_v3
    from networks.layers.TD_RepConv import TDRepConv3D_Dilated

    for m in model.modules():
        if isinstance(m, RepConv3D_SpatialTemporal_v3):
            m.switch_to_deploy()
        if isinstance(m, TDRepConv3D_Dilated):
            m.switch_to_deploy()

    params_infer = sum(p.numel() for p in model.parameters())
    flops_infer, _ = profile(model, inputs=(seq_imgs,), verbose=False)
    params_infer_M = params_infer / 1e6
    flops_infer_G = flops_infer / 1e9

    print("\n" + "=" * 60)
    print("Inference Mode (after switch_to_deploy)")
    print("=" * 60)
    print(f"Params : {params_infer_M:.3f} M")
    print(f"FLOPs  : {flops_infer_G:.3f} G")

    print("\n" + "=" * 60)
    print("Summary (Train vs Infer, comparable)")
    print("=" * 60)
    print(f"Train Params : {params_train_M:.5f} M")
    print(f"Infer Params : {params_infer_M:.5f} M")
    print(f"Train FLOPs  : {flops_train_G:.5f} G  (with correction)")
    print(f"Infer FLOPs  : {flops_infer_G:.5f} G")
    print(f"Param ratio  : {params_infer / params_train:.4f}  (infer/train)")
    print(f"FLOPs ratio  : {flops_infer / flops_train:.4f}  (infer/train)")
    print("=" * 60)

    # 3) FPS 测试（推理模式）：清除 thop 的 hook 后计时
    for m in model.modules():
        m._forward_hooks.clear()
        m._forward_pre_hooks.clear()

    def measure_fps(model, inp, warmup=20, repeat=60):
        model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(inp)
            if inp.is_cuda:
                torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True) if inp.is_cuda else None
            end = torch.cuda.Event(enable_timing=True) if inp.is_cuda else None
            if inp.is_cuda:
                start.record()
            else:
                t0 = time.perf_counter()
            for _ in range(repeat):
                _ = model(inp)
            if inp.is_cuda:
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
            else:
                elapsed_ms = (time.perf_counter() - t0) * 1000
        avg_ms = elapsed_ms / repeat
        fps = 1000.0 / avg_ms
        return avg_ms, fps
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    seq_imgs_dev = seq_imgs.to(device)
    avg_ms, fps = measure_fps(model, seq_imgs_dev)
    # 公平对比：93 为 10 帧入 → 10 帧出，按「输出帧」统一与 MFE（3 帧入→1 帧出）比较
    out_frames_per_forward = seq_imgs.shape[2]  # T
    ms_per_out_frame = avg_ms / out_frames_per_forward
    out_fps = 1000.0 / ms_per_out_frame  # 每秒输出帧数
    print("\n" + "=" * 60)
    print("FPS (Inference Mode)")
    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Input shape : {tuple(seq_imgs.shape)}  (B, C, T, H, W)")
    print(f"Output      : {out_frames_per_forward} frames / forward")
    print(f"Avg time    : {avg_ms:.3f} ms / forward  |  {ms_per_out_frame:.3f} ms / output frame")
    print(f"Forward FPS : {fps:.2f}  (forward passes / s)")
    print(f"Output FPS  : {out_fps:.2f}  (output frames / s, 公平对比用)")
    print("=" * 60)
