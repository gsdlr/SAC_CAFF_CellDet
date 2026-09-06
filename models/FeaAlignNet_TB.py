# -*- coding:utf-8 -*-
"""
Author：R
Date：25-06-2025
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List
# from inplace_abn import InPlaceABN, InPlaceABNSync
# BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')


class BasicEncoder(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, **kwargs):
        super(BasicEncoder, self).__init__()

        self.stride = stride
        self.conv_bn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.relu = nn.ReLU()

        self.conv_bn2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=self.stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        identity = x
        if self.stride != 1:
            identity = self.downsample(x)

        out = self.conv_bn1(x)
        out = self.relu(out)
        out = self.conv_bn2(out)

        out += identity
        out = self.relu(out)

        return out


class BasicDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, task='detection', config_s=False):
        super(BasicDecoder, self).__init__()

        self.task = task

        if config_s:
            self.conv_bn_relu = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            self.conv_bn_relu = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        if self.task == 'counting':
            self.upconv = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.conv_bn_relu(x)
        if self.task == 'counting':
            x = self.upconv(x)
        return x


# align module
class CAFAB(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(CAFAB, self).__init__()

        # adjust the channels of the high-level feature to be the same as that of the low-level
        self.adj = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=1, bias=False)

        # coordinate attention affer concat. The channel is 2*(channels of the low-level feature)
        self.coordatt = CoordAtt(2*outchannel, 2*outchannel)

        self.delta_gen1 = nn.Sequential(
                        nn.Conv2d(outchannel*2, outchannel, kernel_size=1, bias=False),
                        nn.BatchNorm2d(outchannel),
                        nn.LeakyReLU(0.01, inplace=True),
                        nn.Conv2d(outchannel, 2, kernel_size=3, padding=1, bias=False)
                        )

        self.delta_gen2 = nn.Sequential(
                        nn.Conv2d(outchannel*2, outchannel, kernel_size=1, bias=False),
                        nn.BatchNorm2d(outchannel),
                        nn.LeakyReLU(0.01, inplace=True),
                        nn.Conv2d(outchannel, 2, kernel_size=3, padding=1, bias=False)
                        )

        self.delta_gen1[3].weight.data.zero_()
        self.delta_gen2[3].weight.data.zero_()

    # https://github.com/speedinghzl/AlignSeg/issues/7
    # the normlization item is set to [w/s, h/s] rather than [h/s, w/s]
    # the function bilinear_interpolate_torch_gridsample2 is standard implementation, please use bilinear_interpolate_torch_gridsample2 for training.
    def bilinear_interpolate_torch_gridsample2(self, inputs, size, delta=0):
        out_h, out_w = size
        n, c, h, w = inputs.shape
        s = 2.0
        norm = torch.tensor([[[[(out_w-1)/s, (out_h-1)/s]]]]).type_as(inputs).to(inputs.device) # not [h/s, w/s]
        w_list = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h_list = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h_list.unsqueeze(2), w_list.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(inputs).to(inputs.device)
        grid = grid + delta.permute(0, 2, 3, 1) / norm

        outputs = F.grid_sample(inputs, grid, align_corners=True)
        return outputs

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = self.adj(high_stage)
        high_stage_up = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=True)

        concat = torch.cat((low_stage, high_stage_up), 1)
        # coordinate attention
        concat = self.coordatt(concat)

        delta1 = self.delta_gen1(concat)
        delta2 = self.delta_gen2(concat)
        high_stage = self.bilinear_interpolate_torch_gridsample2(high_stage, (h, w), delta1)
        low_stage = self.bilinear_interpolate_torch_gridsample2(low_stage, (h, w), delta2)

        high_stage += low_stage
        return high_stage


""" Coordinate Attention Module"""
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        # self.act = swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 在第三通道cocat, 也就是channel后面那个
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


""" Feature align-based ASPP module"""
class CAFA_ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256):
        super().__init__()

        # 低层特征适配器（共享）
        self.low_feat_adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 创建ASPP分支
        self.aspp_branches = nn.ModuleList()

        # 1. 1x1卷积分支
        self.aspp_branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        # 2. 空洞卷积分支
        for rate in atrous_rates:
            self.aspp_branches.append(ASPPConv(in_channels, out_channels, rate))

        # 3. 全局池化分支
        self.aspp_branches.append(ASPPPooling(in_channels, out_channels))

        # 为每个分支创建CAFAB对齐模块
        self.cafab_modules = nn.ModuleList([
            CAFAB(out_channels, out_channels) for _ in range(len(self.aspp_branches))
        ])

        # 特征融合层
        self.project = nn.Sequential(
            nn.Conv2d(len(self.aspp_branches) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 准备低层特征（原始输入特征的适配版本）
        low_feat = self.low_feat_adapter(x)

        branch_outputs = []
        for branch, cafab in zip(self.aspp_branches, self.cafab_modules):
            # 获取分支输出
            branch_feat = branch(x)

            # 使用CAFAB进行特征对齐和融合
            aligned_feat = cafab(low_feat, branch_feat)
            branch_outputs.append(aligned_feat)

        # 拼接所有分支输出
        concat_feats = torch.cat(branch_outputs, dim=1)

        # 最终投影融合
        return self.project(concat_feats)


class ASPPConv(nn.Sequential):
    """ASPP的单个空洞卷积分支"""
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super().__init__(*modules)

# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

# The original ASPP
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256) -> None:
        """
        in_channels: 输入通道数
        atrous_rates: dilation rate
        out_channels: 输出通道数，默认为 256
        """
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))

        # (B, C, H, W), dim = 1, 按通道拼接
        res = torch.cat(_res, dim=1)
        return self.project(res)

# down1到down4的层数为[1, 1, 1, 1], Two branches
class MyResUDetectorTB(nn.Module):
    def __init__(self):
        super(MyResUDetectorTB, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.down1 = nn.Sequential(
            BasicEncoder(in_channel=32, out_channel=32, stride=2),
            # BasicEncoder(in_channel=32, out_channel=32, stride=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.down2 = nn.Sequential(
            BasicEncoder(in_channel=32, out_channel=64, stride=2),
            # BasicEncoder(in_channel=64, out_channel=64, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.down3 = nn.Sequential(
            BasicEncoder(in_channel=64, out_channel=128, stride=2),
            # BasicEncoder(in_channel=128, out_channel=128, stride=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.down4 = nn.Sequential(
            BasicEncoder(in_channel=128, out_channel=256, stride=2),
            # BasicEncoder(in_channel=256, out_channel=256, stride=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # decoding for detection
        self.cafa_aspp1 = CAFA_ASPP(in_channels=256, atrous_rates=[2, 3, 5], out_channels=256)

        self.align1 = CAFAB(256, 128)
        self.decoder1 = BasicDecoder(256, 128)

        self.align2 = CAFAB(128, 64)
        self.decoder2 = BasicDecoder(128, 64)

        self.align3 = CAFAB(64, 32)
        self.decoder3 = BasicDecoder(64, 32)

        self.align4 = CAFAB(32, 32)

        self.out_layer_1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # decoding for counting
        self.cafa_aspp2 = CAFA_ASPP(in_channels=256, atrous_rates=[2, 3, 5], out_channels=256)
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),  # Added since Exp_013
            nn.ReLU()
        )

        self.up2 = BasicDecoder(256, 128, task='counting')
        self.up3 = BasicDecoder(128, 64, task='counting')
        self.up4 = BasicDecoder(64, 32, task='counting')

        self.out_layer_2 = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.in_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # the detection output
        x5_1 = self.cafa_aspp1(x5)

        align_feature1 = self.align1(x4, x5_1)
        x6 = torch.cat([x4, align_feature1], dim=1)
        x6 = self.decoder1(x6)

        align_feature2 = self.align2(x3, x6)
        x7 = torch.cat([x3, align_feature2], dim=1)
        x7 = self.decoder2(x7)

        align_feature3 = self.align3(x2, x7)
        x8 = torch.cat([x2, align_feature3], dim=1)
        x8 = self.decoder3(x8)

        align_feature4 = self.align4(x1, x8)
        x9 = torch.cat([x1, align_feature4], dim=1)
        x9 = self.out_layer_1(x9)

        # the counting output
        x5_2 = self.cafa_aspp2(x5)
        x10 = self.up1(x5_2)

        x10 = torch.cat([x4, x10], dim=1)
        x10 = self.up2(x10)
        x10 = torch.cat([x3, x10], dim=1)
        x10 = self.up3(x10)
        x10 = torch.cat([x2, x10], dim=1)
        x10 = self.up4(x10)
        x10 = torch.cat([x1, x10], dim=1)
        x10 = self.out_layer_2(x10)

        # x9: detection map; x10: count map
        return x9, x10


if __name__ == '__main__':
    inp = torch.randn(2, 3, 224, 224)

    model = MyResUDetectorTB()
    out_det, out_cnt = model(inp)
    print(out_det.shape)
    print(out_cnt.shape)
