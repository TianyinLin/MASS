import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    """生成高斯窗口"""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """创建SSIM计算窗口"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """计算SSIM"""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    """SSIM计算函数"""
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class Motion_loss(nn.Module):
    """
    Motion Loss for background alignment
    用于约束配准后的图像与原始图像之间的差异
    """

    def __init__(self, l1=0.85, l2=0.15):
        super(Motion_loss, self).__init__()
        self.l1 = l1  # L1 loss权重
        self.l2 = l2  # SSIM loss权重

    def forward(self, warp_imgs, imgs):
        """
        计算Motion Loss（与MFE-Net一致）

        Args:
            warp_imgs: 配准后的特征图列表，每个元素为 [B, C, H, W]（多通道特征图）
            imgs: 原始图像列表，每个元素为 [B, 1, H, W] 或 [B, C, H, W]（原始输入图像）

        Returns:
            motion_loss: 标量损失值

        注意：与MFE-Net一致，只取第一通道进行计算
        """
        motion_L = 0

        # 如果输入是列表，遍历每个元素
        if isinstance(warp_imgs, list):
            for i in range(len(warp_imgs)):
                warp_img = warp_imgs[i]  # [B, C, H, W] 或 [B, 1, H, W]
                img = imgs[i]  # [B, 1, H, W] 或 [B, C, H, W]

                # MFE-Net方式：只取第一通道
                # warp_img: 如果是多通道特征图，只取第一通道
                if warp_img.dim() == 4:
                    if warp_img.size(1) > 1:
                        warp_img = warp_img[:, 0, :, :].unsqueeze(1)  # [B, 1, H, W]
                    elif warp_img.size(1) == 1:
                        pass  # 已经是单通道
                elif warp_img.dim() == 3:
                    warp_img = warp_img.unsqueeze(1)  # [B, 1, H, W]

                # img: 如果是多通道，只取第一通道
                if img.dim() == 4:
                    if img.size(1) > 1:
                        img = img[:, 0, :, :].unsqueeze(1)  # [B, 1, H, W]
                    elif img.size(1) == 1:
                        pass  # 已经是单通道
                elif img.dim() == 3:
                    img = img.unsqueeze(1)  # [B, 1, H, W]

                # 计算L1损失
                L1_loss = F.l1_loss(warp_img, img, reduction='mean')

                # 计算SSIM损失
                Loss_ssim = ssim(warp_img, img)

                # 组合损失
                motion_L += self.l1 * L1_loss + self.l2 * (1 - Loss_ssim)
        else:
            # 如果输入不是列表，直接处理
            warp_img = warp_imgs
            img = imgs

            # 处理维度（只取第一通道）
            if warp_img.dim() == 4:
                if warp_img.size(1) > 1:
                    warp_img = warp_img[:, 0, :, :].unsqueeze(1)
            elif warp_img.dim() == 3:
                warp_img = warp_img.unsqueeze(1)

            if img.dim() == 4:
                if img.size(1) > 1:
                    img = img[:, 0, :, :].unsqueeze(1)
            elif img.dim() == 3:
                img = img.unsqueeze(1)

            L1_loss = F.l1_loss(warp_img, img, reduction='mean')
            Loss_ssim = ssim(warp_img, img)
            motion_L = self.l1 * L1_loss + self.l2 * (1 - Loss_ssim)

        return motion_L





