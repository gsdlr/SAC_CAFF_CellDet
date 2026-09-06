# -*- coding:utf-8 -*-
"""
Author：R
Date：25-06-2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================== Haar 小波变换 ========================

class HaarDWT2D(nn.Module):
    """正向 Haar 2D 小波变换（无可学习参数）"""
    def forward(self, x):
        x_ll = x[:, :, 0::2, 0::2]
        x_lr = x[:, :, 0::2, 1::2]
        x_rl = x[:, :, 1::2, 0::2]
        x_rr = x[:, :, 1::2, 1::2]
        ll = (x_ll + x_lr + x_rl + x_rr) / 2.0
        lh = (x_ll + x_lr - x_rl - x_rr) / 2.0
        hl = (x_ll - x_lr + x_rl - x_rr) / 2.0
        hh = (x_ll - x_lr - x_rl + x_rr) / 2.0
        return ll, lh, hl, hh


class HaarIDWT2D(nn.Module):
    """逆向 Haar 2D 小波变换（无可学习参数）"""
    def forward(self, ll, lh, hl, hh):
        B, C, H2, W2 = ll.shape
        out = ll.new_empty(B, C, H2 * 2, W2 * 2)
        out[:, :, 0::2, 0::2] = (ll + lh + hl + hh) / 2.0
        out[:, :, 0::2, 1::2] = (ll + lh - hl - hh) / 2.0
        out[:, :, 1::2, 0::2] = (ll - lh + hl - hh) / 2.0
        out[:, :, 1::2, 1::2] = (ll - lh - hl + hh) / 2.0
        return out


# ======================== Spatial Gate 模块 ========================
class SpatialGate(nn.Module):
    """Cross-guided spatial attention after IDWT"""
    def __init__(self):
        super().__init__()
        # 输入：skip 的 max/avg pooling (2通道) + dec 的 max/avg pooling (2通道)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, skip_refined, dec):
        """
        skip_refined: (B, C, H, W) IDWT重建后的特征
        dec:          (B, dec_ch, H, W)
        """
        s_max, _ = skip_refined.max(dim=1, keepdim=True)  # (B,1,H,W)
        s_avg = skip_refined.mean(dim=1, keepdim=True)     # (B,1,H,W)
        d_max, _ = dec.max(dim=1, keepdim=True)            # (B,1,H,W)
        d_avg = dec.mean(dim=1, keepdim=True)              # (B,1,H,W)

        mask = self.conv(torch.cat([s_avg, s_max, d_avg, d_max], dim=1))
        return skip_refined * mask


# ======================== CAFF 模块 ========================
class CAFF(nn.Module):
    """
    Cross-guided Adaptive Frequency Filtering
    用解码器语义特征引导 skip 特征的小波子带滤波
    """
    def __init__(self, skip_ch, dec_ch, reduction=8):
        super().__init__()
        self.dwt  = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        input_dim = 4 * skip_ch + dec_ch
        mid = max(input_dim // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 4 * skip_ch, bias=False),
            nn.Sigmoid(),
        )

        self.spatial_gate = SpatialGate()

    def forward(self, skip, dec):
        """
        skip : (B, skip_ch, H, W)  编码器的skip特征 (B, skip_ch, H, W)
        dec  : (B, dec_ch,  H, W)  解码器上采样后的特征 (B, dec_ch, H, W)，已对齐尺寸
        return: (B, skip_ch, H, W) 频率精炼后的 skip
        """
        assert skip.size(2) % 2 == 0 and skip.size(3) % 2 == 0, \
            f"CAFF requires even spatial dims, got {skip.shape}"
        # 小波分解
        ll, lh, hl, hh = self.dwt(skip)

        # skip 子带描述子
        # z_skip = torch.cat([
        #     ll.mean(dim=(-2, -1)),
        #     lh.mean(dim=(-2, -1)),
        #     hl.mean(dim=(-2, -1)),
        #     hh.mean(dim=(-2, -1)),
        # ], dim=1)                          # (B, 4*skip_ch)

        # 等价简洁写法
        subbands = torch.cat([ll, lh, hl, hh], dim=1)  # (B, 4C, H/2, W/2)
        z_skip = subbands.mean(dim=(-2, -1))  # (B, 4C)

        # 解码器语义描述子
        z_dec = dec.mean(dim=(-2, -1))     # (B, dec_ch)

        # 生成子带权重
        w = self.fc(torch.cat([z_skip, z_dec], dim=1))  # (B, 4*skip_ch)
        w_ll, w_lh, w_hl, w_hh = w.chunk(4, dim=1)

        # 子带加权
        ll = ll * w_ll[:, :, None, None]
        lh = lh * w_lh[:, :, None, None]
        hl = hl * w_hl[:, :, None, None]
        hh = hh * w_hh[:, :, None, None]

        # 逆变换还原
        out = self.idwt(ll, lh, hl, hh)

        # 空间级跨引导精炼
        out = self.spatial_gate(out, dec)

        # 逆变换还原
        return out


# ======================== 编码器块 ========================
class ResBlock(nn.Module):
    """
    结构示意:
    x
    │
    ▼
    [Conv3×3(s=2)-BN-ReLU]        ← 仅当 stride=2 时存在，否则跳过
    │
    ▼
    x_down ──────────────────┐
    │                        │
    ▼                        ▼
    Conv3×3-BN-ReLU     Shortcut(1×1-BN)
    │                        │
    ▼                        │
    Conv3×3-BN               │
    │                        │
    └─────── ⊕ ──────────────┘
             │
             ▼
            ReLU
             │
             ▼
        Conv3×3-BN-ReLU
             │
             ▼
            out
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # ---- 独立下采样（stride=2 时激活） ----
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3,
                          stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.downsample = nn.Identity()

        # ---- 残差主路径（全部 stride=1） ----
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # ---- shortcut（仅通道对齐，无空间变化） ----
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        # ---- 残差合并后的额外卷积 ----
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        # 1) 下采样（stride=1 时直接跳过）
        x = self.downsample(x)

        # 2) 残差学习
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)

        # 3) 额外卷积
        out = self.relu(self.bn3(self.conv3(out)))
        return out


# ======================== 空洞残差块（替代下采样） ========================

class DilResBlock(nn.Module):
    """
    与 ResBlock 结构一致，但用空洞卷积代替 stride 下采样
    采用 HDC (Hybrid Dilated Convolution) 模式：dilation = [d, 1]
    避免 gridding effect，同时扩大感受野

    结构示意:
    x ──────────────────────┐
    │                        │
    ▼                        ▼
    DilConv3×3(d)-BN-ReLU   Shortcut(1×1-BN)
    │                        │
    ▼                        │
    Conv3×3-BN               │
    │                        │
    └─────── ⊕ ──────────────┘
             │
             ▼
            ReLU
             │
             ▼
        Conv3×3-BN-ReLU
             │
             ▼
            out
    """
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # ---- 空洞卷积主路径 ----
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)

        # 第二层用 dilation=1，打破 gridding
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # ---- shortcut ----
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        # ---- 残差合并后的额外卷积 ----
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        out = self.relu(self.bn3(self.conv3(out)))
        return out


# ======================== 解码器块 ========================

class BasDec(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


# ======================== 上采样 + CAFF + 解码 ========================

class UpBlock(nn.Module):
    """转置卷积上采样 → CAFF 精炼 skip → 拼接 → BasDec"""
    def __init__(self, in_ch, skip_ch, out_ch, reduction=8):
        super().__init__()
        # 上采样：通道数从 in_ch 降到 skip_ch，空间尺寸 ×2
        self.up   = nn.ConvTranspose2d(in_ch, skip_ch,
                                       kernel_size=4, stride=2, padding=1)
        self.caff = CAFF(skip_ch=skip_ch, dec_ch=skip_ch,
                         reduction=reduction)
        # 拼接后通道数 = skip_ch + skip_ch = 2 * skip_ch
        self.dec  = BasDec(skip_ch * 2, out_ch)

    def forward(self, x, skip):
        dec = self.up(x)

        # 尺寸对齐（应对奇数像素差异）
        dh = skip.size(2) - dec.size(2)
        dw = skip.size(3) - dec.size(3)
        if dh != 0 or dw != 0:
            dec = F.pad(dec, (dw // 2, dw - dw // 2,
                              dh // 2, dh - dh // 2))

        skip_refined = self.caff(skip, dec)
        return self.dec(torch.cat([skip_refined, dec], dim=1))


# ======================== 同尺度融合 + CAFF + 解码 ========================

class FuseBlock(nn.Module):
    """
    同分辨率融合模块（替代 UpBlock 用于瓶颈与 skip 在同一尺度的情况）
    1×1 通道压缩 → CAFF 精炼 skip → 拼接 → BasDec
    """
    def __init__(self, in_ch, skip_ch, out_ch, reduction=8):
        super().__init__()
        # 通道压缩：从 in_ch 降到 skip_ch（不改变空间尺寸）
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, skip_ch, 1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(inplace=True),
        )
        self.caff = CAFF(skip_ch=skip_ch, dec_ch=skip_ch,
                         reduction=reduction)
        # 拼接后通道数 = skip_ch + skip_ch = 2 * skip_ch
        self.dec  = BasDec(skip_ch * 2, out_ch)

    def forward(self, x, skip):
        dec = self.reduce(x)
        skip_refined = self.caff(skip, dec)
        return self.dec(torch.cat([skip_refined, dec], dim=1))


# ======================== 完整 UNet（空洞卷积瓶颈版本） ========================
class CAFFNet(nn.Module):
    """
    编码器: stem + 3 层（每层 ResBlock，stride=2）→ 最深到 H/8
    瓶颈:   DilResBlock（dilation=2），不下采样，保持 H/8
    解码器: FuseBlock(同尺度) + 3 层 UpBlock（每层含 CAFF + BasDec）

    输入尺寸建议为 8 的倍数（仅经过 3 次 stride-2 下采样）
    相比 CAFFNet_4down，瓶颈分辨率从 H/16 提升到 H/8，
    保留更多空间细节，适合边缘/小目标敏感的任务。
    """
    def __init__(self, in_ch=3, out_ch=1, base=32, reduction=8):
        super().__init__()

        # ---------- Stem：保持原始分辨率 ----------
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # ---------- 编码器（3 次下采样） ----------
        self.enc1 = nn.Sequential(                       # H → H/2
            ResBlock(base,     base * 2,  stride=2),
        )
        self.enc2 = nn.Sequential(                       # H/2 → H/4
            ResBlock(base * 2, base * 4,  stride=2),
        )
        self.enc3 = nn.Sequential(                       # H/4 → H/8
            ResBlock(base * 4, base * 8,  stride=2),
        )

        # ---------- 空洞卷积瓶颈（不下采样，保持 H/8） ----------
        self.bottleneck = nn.Sequential(
            DilResBlock(base * 8, base * 16, dilation=2),
        )

        # ---------- 解码器（FuseBlock + 3 层 UpBlock） ----------
        # dec3: 同尺度融合（bottleneck H/8 + skip s3 H/8）
        self.dec3 = FuseBlock(base * 16, base * 8, base * 8, reduction)

        # dec2/1/0: 上采样 + CAFF 融合
        self.dec2 = UpBlock(base * 8,  base * 4, base * 4,  reduction)
        self.dec1 = UpBlock(base * 4,  base * 2, base * 2,  reduction)
        self.dec0 = UpBlock(base * 2,  base,     base,      reduction)

        # ---------- 输出头 ----------
        self.head = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.BatchNorm2d(base // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, out_ch, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        # 编码
        s0 = self.stem(x)          # (B, 32,   H,    W)
        s1 = self.enc1(s0)         # (B, 64,   H/2,  W/2)
        s2 = self.enc2(s1)         # (B, 128,  H/4,  W/4)
        s3 = self.enc3(s2)         # (B, 256,  H/8,  W/8)

        # 空洞卷积瓶颈（不下采样）
        feat = self.bottleneck(s3) # (B, 512,  H/8,  W/8)

        # 解码 + CAFF
        d3 = self.dec3(feat, s3)   # (B, 256,  H/8,  W/8)  同尺度融合
        d2 = self.dec2(d3,   s2)   # (B, 128,  H/4,  W/4)
        d1 = self.dec1(d2,   s1)   # (B, 64,   H/2,  W/2)
        d0 = self.dec0(d1,   s0)   # (B, 32,   H,    W)

        return self.head(d0), feat # (B, out_ch, H, W), (B, 512, H/8, W/8)


# ======================== 完整 UNet（4次下采样版本） ========================

class CAFFNet_4down(nn.Module):
    """
    编码器: stem + 3 层（每层 ResBlock，首个 stride=2）
    瓶颈:   ResBlock（stride=2）
    解码器: 4 层 UpBlock（每层含 CAFF + BasDec）

    输入尺寸建议为 16 的倍数（经过 4 次 stride-2 下采样）
    """
    def __init__(self, in_ch=3, out_ch=1, base=32, reduction=8):
        super().__init__()

        # ---------- Stem：保持原始分辨率 ----------
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # ---------- 编码器 ----------
        self.enc1 = nn.Sequential(
            ResBlock(base,     base * 2,  stride=2),
        )
        self.enc2 = nn.Sequential(
            ResBlock(base * 2, base * 4,  stride=2),
        )
        self.enc3 = nn.Sequential(
            ResBlock(base * 4, base * 8,  stride=2),
        )

        # ---------- 瓶颈 ----------
        self.bottleneck = nn.Sequential(
            ResBlock(base * 8,  base * 16, stride=2),
        )

        # ---------- 解码器（每层嵌入 CAFF） ----------
        self.dec3 = UpBlock(base * 16, base * 8, base * 8,  reduction)
        self.dec2 = UpBlock(base * 8,  base * 4, base * 4,  reduction)
        self.dec1 = UpBlock(base * 4,  base * 2, base * 2,  reduction)
        self.dec0 = UpBlock(base * 2,  base,     base,      reduction)

        # ---------- 输出头 ----------
        self.head = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.BatchNorm2d(base // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, out_ch, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        # 编码
        s0 = self.stem(x)          # (B, 32,   H,    W)
        s1 = self.enc1(s0)         # (B, 64,  H/2,  W/2)
        s2 = self.enc2(s1)         # (B, 128,  H/4,  W/4)
        s3 = self.enc3(s2)         # (B, 256,  H/8,  W/8)

        # 瓶颈
        feat = self.bottleneck(s3) # (B, 512, H/16, W/16)

        # 解码 + CAFF
        d3 = self.dec3(feat, s3)   # (B, 256,  H/8,  W/8)
        d2 = self.dec2(d3, s2)     # (B, 128,  H/4,  W/4)
        d1 = self.dec1(d2, s1)     # (B, 64,  H/2,  W/2)
        d0 = self.dec0(d1, s0)     # (B, 32,   H,    W)

        return self.head(d0), feat


# ======================== 测试 ========================

if __name__ == "__main__":
    model = CAFFNet(in_ch=3, out_ch=1, base=32)
    x = torch.randn(2, 3, 320, 320)
    out, feat = model(x)
    print(f"Input:      {x.shape}")
    print(f"Output:     {out.shape}")
    print(f"Bottleneck: {feat.shape}")

    # 参数量统计
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total / 1e6:.2f} M")