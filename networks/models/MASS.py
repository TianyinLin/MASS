import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networks.layers.DSCIM import DSCIM
from networks.layers.SSVA import SSVA
from networks.layers.FrameDifferenceModule import FrameDifferenceModule
from networks.layers.background_align import BackgroundAlignmentModule
from networks.losses.Motion_loss import Motion_loss
from networks.layers.MultiScaleFeatureExtractor import MultiScaleFeatureExtractor
from networks.layers.HierarchicalFlowRefiner import HierarchicalFlowRefiner
from networks.layers.RepTDC import TDRepConv3D_Dilated


class detector(nn.Module):
    def __init__(self, num_classes, seqlen=10, out_len=10,
                 feature_channels=64, local_radius=2, attn_splits=1, prop_radius=-1):
        super(detector, self).__init__()
        self.out_len = out_len
        self.seqlen = seqlen
        self.feature_channels = feature_channels

        # ========== 特征提取网络 ==========
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
            local_radius=local_radius
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
        self.TPro1 = SSVA(d_model=32, num_head=4, seqlen=seqlen)
        self.conv_out1_1 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TPro2 = SSVA(d_model=32, num_head=4, seqlen=seqlen)
        self.conv_out1_2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TPro3 = SSVA(d_model=32, num_head=4, seqlen=seqlen)
        self.conv_out1_3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=8, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(8), nn.ReLU(inplace=True))
        self.TPro4 = SSVA(d_model=32, num_head=4, seqlen=seqlen)
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

    def forward(self, seq_imgs):
        """
        输入: seq_imgs [B, 1, T, H, W] - T=10帧
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

        # ========== 三帧三帧处理，只保存对齐差分后的结果 ==========
        # 存储对齐差分后的特征（保持4个分辨率）
        aligned_diff_1x_list = []
        aligned_diff_2x_list = []
        aligned_diff_4x_list = []
        aligned_diff_8x_list = []

        # 存储warp后的图像和原始图像（用于Motion Loss）
        warp_img_list = []
        img_list = []

        # 滑动窗口处理，每次3帧
        for t in range(T):
            # 获取三帧索引
            idx0, idx1, idx2 = self._get_frame_indices(t)

            feat0_1x = feat_1x_all[:, :, idx0]
            feat1_1x = feat_1x_all[:, :, idx1]
            feat2_1x = feat_1x_all[:, :, idx2]

            feat0_2x = feat_2x_all[:, :, idx0]
            feat1_2x = feat_2x_all[:, :, idx1]
            feat2_2x = feat_2x_all[:, :, idx2]

            feat0_4x = feat_4x_all[:, :, idx0]
            feat1_4x = feat_4x_all[:, :, idx1]
            feat2_4x = feat_4x_all[:, :, idx2]

            feat0_8x = feat_8x_all[:, :, idx0]
            feat1_8x = feat_8x_all[:, :, idx1]
            feat2_8x = feat_8x_all[:, :, idx2]

            # ========== 在1/8分辨率计算运动向量 ==========
            v0_2_8x, v1_2_8x = self.background_align.compute_refined_vector(feat0_8x, feat1_8x, feat2_8x)

            # ========== 分层 refinement ==========
            # 组装 feats dict（每个 value 是 (feat0, feat1, feat2)）
            feats_for_refine = {
                '8x': (feat0_8x, feat1_8x, feat2_8x),
                '4x': (feat0_4x, feat1_4x, feat2_4x),
                '2x': (feat0_2x, feat1_2x, feat2_2x),
                '1x': (feat0_1x, feat1_1x, feat2_1x)
            }
            # refine v0 (使用feat0作为reference)
            v0_2_8x, v0_2_4x, v0_2_2x, v0_2_1x = self.flow_refiner_v0(v0_2_8x, feats_for_refine, ref_idx=0)
            # refine v1 (使用feat1作为reference)
            v1_2_8x, v1_2_4x, v1_2_2x, v1_2_1x = self.flow_refiner_v1(v1_2_8x, feats_for_refine, ref_idx=1)

            # ========== 对齐与差分 ==========
            # 对齐与差分（1x分辨率）- 同时返回对齐后的特征图用于Motion Loss
            # 这样避免重复计算warp操作
            aligned_diff_1x, feat0_1x_warp, feat1_1x_warp = self._align_and_diff(
                feat0_1x, feat1_1x, feat2_1x, v0_2_1x, v1_2_1x,
                self.frame_diff_1x, return_warped=True
            )
            aligned_diff_1x_list.append(aligned_diff_1x)

            # 保存warp后的特征图和原始输入图像
            # 对两个warp后的特征图都计算Motion Loss，更全面地约束对齐质量
            img2_original = seq_imgs[:, :, idx2, :, :]  # [B, 1, H, W] - 原始输入图像
            warp_img_list.append(feat0_1x_warp)  # [B, feature_channels, H, W] - warp后的特征图
            warp_img_list.append(feat1_1x_warp)  # [B, feature_channels, H, W] - warp后的特征图
            img_list.append(img2_original)  # [B, 1, H, W] - 原始输入图像（参考）
            img_list.append(img2_original)  # [B, 1, H, W] - 原始输入图像（参考，与feat0_1x_warp对应）

            # 对齐与差分（其他分辨率）- 不需要返回warp特征图
            aligned_diff_2x_list.append(
                self._align_and_diff(feat0_2x, feat1_2x, feat2_2x, v0_2_2x, v1_2_2x, self.frame_diff_2x)
            )
            aligned_diff_4x_list.append(
                self._align_and_diff(feat0_4x, feat1_4x, feat2_4x, v0_2_4x, v1_2_4x, self.frame_diff_4x)
            )
            aligned_diff_8x_list.append(
                self._align_and_diff(feat0_8x, feat1_8x, feat2_8x, v0_2_8x, v1_2_8x, self.frame_diff_8x)
            )

        # ========== 组合成3D特征序列并处理 ==========
        seq_feats_list = [
            torch.stack(aligned_diff_1x_list, dim=2),  # [B, feature_channels (64), T, H, W]
            torch.stack(aligned_diff_2x_list, dim=2),  # [B, feature_channels (64), T, H/2, W/2]
            torch.stack(aligned_diff_4x_list, dim=2),  # [B, feature_channels (64), T, H/4, W/4]
        ]

        seq_feats_8x = torch.stack(aligned_diff_8x_list, dim=2)  # [B, feature_channels (64), T, H/8, W/8]

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

        # 统一处理TPro和输出层（包含1x分支）
        for i in range(4):
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

        warp_img = warp_img_list
        img = img_list

        return seq_feats, seq_midseg, aux_out1, aux_out2, aux_out3, aux_out4, warp_img, img


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
    from networks.layers.RepCDC import RepConv3D_SpatialTemporal_v3
    from networks.layers.RepTDC import TDRepConv3D_Dilated

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

    seq_imgs = torch.randn(1, 1, 10, 128, 128)

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
    from networks.layers.RepCDC import RepConv3D_SpatialTemporal_v3
    from networks.layers.RepTDC import TDRepConv3D_Dilated

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

    def measure_fps(model, inp, warmup=20, repeat=40):
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
