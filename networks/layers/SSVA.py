import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from networks.layers.SPDC import RepConv3D_Spatial
from networks.layers.TemporalRepConv import RepConv3D

class LearnableWT(nn.Module):
    def __init__(self, d_model=32, num_head=4, T=10):
        super().__init__()
        assert d_model % num_head == 0

        self.d_model = d_model
        self.num_head = num_head
        self.hidden = d_model // num_head
        self.T = T

        self.WT = nn.Parameter(torch.randn(num_head, T, T)*0.01)
        self.norm = nn.BatchNorm3d(d_model)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv3d(d_model, d_model, (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(d_model),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        B, H, W, D, T = x.shape
        # reshape: → [BHW, T, D]
        x_flat = x.permute(0, 1, 2, 4, 3).reshape(B*H*W, T, D)
        # multi-head split → [BHW, T, num_head, hidden]
        x_heads = x_flat.view(B*H*W, T, self.num_head, self.hidden).permute(0, 2, 1, 3)

        out_heads = []
        for h in range(self.num_head):
            WT_h = self.WT[h]
            v_h = x_heads[:, h, :, :]

            out_h = torch.einsum("ij, bjc->bic", WT_h, v_h)
            out_heads.append(out_h.unsqueeze(1))

        out = torch.cat(out_heads, dim=1)
        out = out.permute(0, 2, 1, 3).reshape(B*H*W, T, D)

        out = out.view(B, H, W, T, D).permute(0, 4, 3, 1, 2)
        out = self.relu(self.norm(out))
        out = self.conv(out)
        out = out.permute(0, 4, 3, 1, 2)
        return out

class SSVA(nn.Module):

    def __init__(self, d_model=32, num_head=4, seqlen=10, ffn_ratio=2):
        super().__init__()

        assert d_model % num_head == 0

        self.d_model = d_model
        self.hidden_dim = d_model // num_head
        self.num_head = num_head
        self.seqlen = seqlen

        # ================= Q / K / V 构造 =================
        self.WT = LearnableWT(d_model=d_model, num_head=num_head, T=seqlen)

        self.spdc = RepConv3D_Spatial(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=5,
            stride=1,
            padding=2,
            use_fusion_conv=True
        )

        self.temporal_repconv = RepConv3D(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=5,
            stride=1,
            padding=2,
            use_fusion_conv=True
        )

        # ================= Attention 后投影 =================
        self.attn_proj = nn.Sequential(
            nn.Conv3d(d_model, d_model, 1),
            nn.BatchNorm3d(d_model),
            nn.ReLU(inplace=True)
        )

        # ================= FFN（Transformer 后半部分） =================
        self.ffn = nn.Sequential(
            nn.Conv3d(d_model, d_model * ffn_ratio, 1),
            nn.BatchNorm3d(d_model * ffn_ratio),
            nn.ReLU(inplace=True),
            nn.Conv3d(d_model * ffn_ratio, d_model, 1),
            nn.BatchNorm3d(d_model)
        )
        self.ReLU = nn.ReLU(inplace=True)

    def _mhsa(self, Q, K, V):
        """
        Q, K, V: [N, T, D]
        """
        N, T, D = Q.shape

        Q = Q.view(N, T, self.num_head, self.hidden_dim).transpose(1, 2)
        K = K.view(N, T, self.num_head, self.hidden_dim).transpose(1, 2)
        V = V.view(N, T, self.num_head, self.hidden_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(N, T, D)

        return out

    def forward(self, input):
        """
        input: [B, H, W, D, T]
        return: [B, D, T, H, W]
        """
        B, H, W, D, T = input.shape

        # ================= Q =================
        Q_3d = self.spdc(input.permute(0, 3, 4, 1, 2))   # [B, D, T, H, W]
        Q_3d = self.temporal_repconv(Q_3d)
        Q = Q_3d.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, D)

        # ================= K =================
        K_3d = self.WT(input)                           # [B, D, T, H, W]
        K = K_3d.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, D)

        # ================= V =================
        V = input.permute(0, 1, 2, 4, 3).reshape(B * H * W, T, D)

        # ================= MHSA =================
        attn_out = self._mhsa(Q, K, V)
        attn_out = attn_out.view(B, H, W, T, D).permute(0, 4, 3, 1, 2)

        # ================= Attention Residual =================
        x = input.permute(0, 3, 4, 1, 2)  # [B, D, T, H, W]
        x = x + self.attn_proj(attn_out)

        # ================= FFN + Residual =================
        out = x + self.ffn(x)

        return out