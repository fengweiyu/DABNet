import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DABNet"]


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                             padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class CoordAtt(nn.Module):
    """
    Coordinate Attention (坐标注意力)

    将通道注意力分解为两个一维编码过程，沿水平和垂直方向聚合特征，
    能够在保留空间位置信息的同时建模通道间依赖关系，几乎不增加计算量。

    参考: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021

    Args:
        inp: 输入通道数
        oup: 输出通道数
        reduction: 中间通道的缩减比例，默认 32
    """

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.inp = inp
        self.oup = oup
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, 1)
        self.conv_w = nn.Conv2d(mip, oup, 1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 通道缩减
        y = self.act(self.bn1(self.conv1(x)))

        # 分别沿 X 和 Y 方向编码注意力权重
        x_h = self.conv_h(y)  # H 方向注意力
        x_w = self.conv_w(y)  # W 方向注意力

        # 空间-通道联合注意力
        attn = (x_h + x_w).sigmoid()

        # 当 inp != oup 时, 需要对齐维度（1×1 投影）
        if c != self.oup or identity.shape[1] != self.oup:
            identity = F.conv2d(identity, torch.eye(self.oup, c, device=x.device).view(self.oup, c, 1, 1))

        return identity * attn


class LightFPN(nn.Module):
    """
    轻量级多尺度特征金字塔融合模块 (Light Feature Pyramid Network)

    将编码器不同层级的特征通过侧边卷积统一后，自顶向下逐级融合上采样，
    最终输出融合了多尺度上下文和空间细节的特征图。

    特征层级（以 Cityscapes 512×1024 输入为例）：
        Level 0: H/2 × W/2   (浅层，空间细节丰富)
        Level 1: H/4 × W/4   (中层)
        Level 2: H/8 × W/8   (深层，语义信息丰富)

    数据流：
        f2(H/8) ──→ lateral_2 ──→ p2 ──→ upsample(×2) ──→ +
        f1(H/4) ──→ lateral_1 ────────────────────────→ + ──→ upsample(×2) ──→ +
        f0(H/2) ──→ lateral_0 ──────────────────────────────────────────────→ + ──→ fusion ──→ output(H/2)

    Args:
        in_channels_list: 各层级输入通道数列表 [C0, C1, C2]，从浅到深
        out_channels: 统一后的通道数，默认 128
    """

    def __init__(self, in_channels_list, out_channels=128):
        super().__init__()
        self.out_channels = out_channels

        # 侧边 1×1 卷积，将各层级特征统一到 out_channels 维度
        self.lateral_convs = nn.ModuleList()
        for in_c in in_channels_list:
            self.lateral_convs.append(
                Conv(in_c, out_channels, kSize=1, stride=1, padding=0, bn_acti=True)
            )

        # 融合后的特征平滑卷积
        self.fusion_conv = Conv(out_channels, out_channels, kSize=3, stride=1, padding=1, bn_acti=True)

    def forward(self, features):
        """
        Args:
            features: [f0, f1, f2] 从浅层到深层的特征列表
                      分辨率依次为 H/2, H/4, H/8
        Returns:
            融合后的特征图，分辨率 = H/2，通道数 = out_channels
        """
        f0, f1, f2 = features

        # 自顶向下融合
        p2 = self.lateral_convs[2](f2)  # H/8, out_channels
        p1 = self.lateral_convs[1](f1)  # H/4, out_channels
        p1 = p1 + F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)

        p0 = self.lateral_convs[0](f0)  # H/2, out_channels
        p0 = p0 + F.interpolate(p1, size=p0.shape[2:], mode='bilinear', align_corners=False)

        out = self.fusion_conv(p0)
        return out


class DABNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=6, fpn_channels=128):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 3)
        self.ca_1 = CoordAtt(32 + 3, 32 + 3)  # 浅层 Coordinate Attention

        # DAB Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + 3)
        self.ca_2 = CoordAtt(128 + 3, 128 + 3)  # 中层 Coordinate Attention

        # DAB Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)
        self.ca_3 = CoordAtt(256 + 3, 256 + 3)  # 深层 Coordinate Attention

        # LightFPN: 多尺度特征融合
        # 三个特征层级通道数: [output0_cat(35), output1_cat(131), output2_cat(259)]
        self.fpn = LightFPN(in_channels_list=[32 + 3, 128 + 3, 256 + 3],
                            out_channels=fpn_channels)

        # 分类器: 接收 FPN 融合后的特征 (H/2 分辨率)
        self.classifier = nn.Sequential(Conv(fpn_channels, classes, 1, 1, padding=0))

    def forward(self, input):

        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))
        output0_cat = self.ca_1(output0_cat)  # Coordinate Attention 增强浅层特征

        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))
        output1_cat = self.ca_2(output1_cat)  # Coordinate Attention 增强中层特征

        # DAB Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))
        output2_cat = self.ca_3(output2_cat)  # Coordinate Attention 增强深层特征

        # 多尺度特征融合: 将三个层级的特征送入 LightFPN
        fpn_out = self.fpn([output0_cat, output1_cat, output2_cat])

        # 分类并上采样到原始分辨率
        out = self.classifier(fpn_out)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        return out
