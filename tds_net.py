import torch
import math
from torch import nn, cat
from torchsummary import summary
from typing import Dict
from einops import rearrange
from model_utils import SE_Block, PP_Model, BasicBlock, CED, SobelEdgeDetector, Bag, SGEEM


class DSConv_pro(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            kernel_size: int = 9,
            extend_scope: float = 1.0,
            morph: int = 0,
            if_offset: bool = True,
            device: torch.device = "cuda",
    ):
        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        offset = self.offset_conv(input)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)

        output = self.gn(output)
        output = self.relu(output)

        return output


import torch
import einops


def get_coordinate_map_2D(
        offset: torch.Tensor,
        morph: int,
        extend_scope: float = 1.0,
        device: torch.device = "cuda",
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    # 确保 offset 在指定设备上
    offset = offset.to(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                    y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                    y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        # 确保所有操作的张量都在同一设备上
        y_new_ = y_new_.to(device)
        y_offset_new_ = y_offset_new_.to(device)

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                    x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                    x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        # 确保所有操作的张量都在同一设备上
        x_new_ = x_new_.to(device)
        x_offset_new_ = x_offset_new_.to(device)

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
        input_feature: torch.Tensor,
        y_coordinate_map: torch.Tensor,
        x_coordinate_map: torch.Tensor,
        interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    # 获取输入特征的设备
    device = input_feature.device

    # 确保坐标映射张量在同一设备上
    y_coordinate_map = y_coordinate_map.to(device)
    x_coordinate_map = x_coordinate_map.to(device)

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
        coordinate_map: torch.Tensor,
        origin: list,
        target: list = [-1, 1],
):
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Ups(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.bn = nn.BatchNorm2d(in_channels)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(self.bn(x))
        x = self.relu(self.conv(x))
        return x


# 语义概览信息分支的通道注意力
class ChannelAttention(nn.Module):
    """通道注意力模块，通过全局平均池化和最大值池化提取并拼接注意力分数"""

    def __init__(self, in_channels=128, reduction_ratio=16, out_channels=1024):
        super(ChannelAttention, self).__init__()
        assert out_channels % 2 == 0, "out_channels must be even"

        branch_channels = out_channels // 2

        # 全局平均池化分支
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, branch_channels, bias=False)
        )

        # 全局最大值池化分支
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, branch_channels, bias=False)
        )

        # 可选的激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化分支
        avg_out = self.avg_pool(x)
        avg_out = torch.flatten(avg_out, 1)
        avg_out = self.avg_fc(avg_out)

        # 最大值池化分支
        max_out = self.max_pool(x)
        max_out = torch.flatten(max_out, 1)
        max_out = self.max_fc(max_out)

        # 拼接两个分支的输出
        out = torch.cat([avg_out, max_out], dim=1)

        # 重塑为[B, C, 1, 1]的注意力分数
        out = out.unsqueeze(-1).unsqueeze(-1)

        # 应用sigmoid激活（可选）
        # out = self.sigmoid(out)

        return out


# 语义概览信息分支的通道注意力增强
class AttentionEnhancementModule(nn.Module):
    def __init__(self, feature_channels=1024, attn_channels=128, reduction_ratio=16):
        super(AttentionEnhancementModule, self).__init__()

        # 通道注意力模块，从分支B提取注意力分数
        self.channel_attention = ChannelAttention(
            in_channels=attn_channels,
            reduction_ratio=reduction_ratio,
            out_channels=feature_channels  # 确保输出通道数与主干分支A相同
        )

        # 可选的特征转换层，用于调整通道数（如果分支B的通道数不等于1024）
        self.feature_adaptation = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature_A, feature_B):
        """
        feature_A: 主干分支A的特征，形状为 [B, 1024, 32, 32]
        feature_B: 注意力分支B的特征，形状为 [B, 128, 256, 256]
        """
        # 从分支B提取通道注意力分数
        attention = self.channel_attention(feature_B)  # [B, 1024, 1, 1]

        # 应用注意力：特征A乘以注意力分数
        enhanced_feature = feature_A * attention  # [B, 1024, 32, 32]

        # 特征融合：先相乘后相加 (类似SE模块的结构)
        output = feature_A + enhanced_feature  # [B, 1024, 32, 32]

        # 可选的特征转换
        output = self.feature_adaptation(output)

        return output


"""U-Net with DSConv_pro"""


class EncoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TDSNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            n_classes: int = 1,
            kernel_size: int = 9,
            extend_scope: float = 1.0,
            if_offset: bool = True,
            device: str = "cuda",
            number: int = 64,
    ):
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number

        # Encoder
        self.conv1_1 = EncoderConv(in_channels, self.number)
        self.dsc_conv1_x = DSConv_pro(
            in_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.dsc_conv1_y = DSConv_pro(
            in_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv1_3 = EncoderConv(3 * self.number, self.number)

        self.conv2_1 = EncoderConv(self.number, 2 * self.number)
        self.dsc_conv2_x = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.dsc_conv2_y = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv2_3 = EncoderConv(6 * self.number, 2 * self.number)

        self.conv3_1 = EncoderConv(2 * self.number, 4 * self.number)
        self.dsc_conv3_x = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.dsc_conv3_y = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv3_3 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv4_1 = EncoderConv(4 * self.number, 8 * self.number)
        self.dsc_conv4_x = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.dsc_conv4_y = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv4_3 = EncoderConv(24 * self.number, 8 * self.number)

        self.conv5_1 = EncoderConv(8 * self.number, 16 * self.number)
        self.dsc_conv5_x = DSConv_pro(
            8 * self.number,
            16 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.dsc_conv5_y = DSConv_pro(
            8 * self.number,
            16 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv5_3 = EncoderConv(48 * self.number, 16 * self.number)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv6_1 = DecoderConv(24 * self.number, 8 * self.number)
        self.conv6_2 = DecoderConv(8 * self.number, 8 * self.number)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv7_1 = DecoderConv(12 * self.number, 4 * self.number)
        self.conv7_2 = DecoderConv(4 * self.number, 4 * self.number)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv8_1 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv8_2 = DecoderConv(2 * self.number, 2 * self.number)

        # 最后一级上采样模块
        self.convup1 = nn.Conv2d(192, 128, 1)

        # --upsample模块128x256x256-64x512x512
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.up1_ = nn.Conv2d(128, 64, 1)

        self.out_conv = OutConv(self.number, n_classes)
        # ----------------------

        # self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

        # 语义分支的旁支
        # stage1,128x256x256-128x256x256,考虑此处是否为空间注意力
        self.stage1_att = SE_Block(128)

        # down2的旁支256x128x128-128x256x256
        self.down2_1up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.smooth2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # down3的旁支512x64x64-256x128x128-128x256x256
        self.down3_1up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.smooth3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # down4的旁支1025x32x32-512x64x64-256x128x128-128x256x256
        self.down4_1up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.GELU(),
            self.down3_1up
        )
        self.smooth4_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # 语义分支的注意力模块
        self.attention1 = SE_Block(128)
        self.attention2 = SE_Block(128)

        self.attention3 = SE_Block(128)
        # 输出时上采样后的smooth
        self.smooth_out = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.attention4 = SE_Block(128)
        self.attention_out = SE_Block(64)

        # 语义分支输出通道注意力指导特征分支提取
        self.enhancement = AttentionEnhancementModule(feature_channels=1024, attn_channels=128, reduction_ratio=16)

        # ----边缘流
        self.edge1 = SobelEdgeDetector(device)  # shuru3x512x512-1x512x512
        # 初始卷积
        self.initial_conv = nn.Conv2d(3, 21, kernel_size=1, stride=1)  # 3x512x512-21x512x512
        # down conv
        self.edge_conv0 = nn.Conv2d(22, 64, kernel_size=2, stride=2)  # 22x512x512-64x256x256
        # 语义边界第一次融合
        self.sgeem1 = SGEEM(64, 128)
        # ---edge_convX2,basicblock为2xConv
        self.edge_conv1 = BasicBlock(64, 64)  # 64x256x256-64x256x256
        # 语义边界第2次融合
        self.sgeem2 = SGEEM(64, 128)
        # 边缘网络在中间部分进行融合

        # ---CONVx2
        self.edge_conv2 = BasicBlock(64, 64)  # 边缘为64x256x256-64x256x256
        # 语义边界第3次融合
        self.sgeem3 = SGEEM(64, 128)

        # ---边缘分割head
        self.edge_head = nn.Sequential(
            Ups(64, 32),
            nn.Conv2d(32, 1, kernel_size=1, stride=1))  # 32x512x512-1x512x512

    def forward(self, x):
        # Encoder block 1
        # x:3x512x512
        x1_1 = self.conv1_1(x)
        x1_x = self.dsc_conv1_x(x)
        x1_y = self.dsc_conv1_y(x)
        stage0 = self.conv1_3(cat([x1_1, x1_x, x1_y], dim=1))  # stage0:64x512x512
        x1 = self.maxpooling(stage0)

        # Encoder block 2
        x2_1 = self.conv2_1(x1)
        x2_x = self.dsc_conv2_x(x1)
        x2_y = self.dsc_conv2_y(x1)
        stage1 = self.conv2_3(cat([x2_1, x2_x, x2_y], dim=1))  # stage1:128x256x256
        x2 = self.maxpooling(stage1)

        # Encoder block 3
        x3_1 = self.conv3_1(x2)
        x3_x = self.dsc_conv3_x(x2)
        x3_y = self.dsc_conv3_y(x2)
        stage2 = self.conv3_3(cat([x3_1, x3_x, x3_y], dim=1))  # stage2:256x128x128
        x3 = self.maxpooling(stage2)

        # Encoder block 4
        x4_1 = self.conv4_1(x3)
        x4_x = self.dsc_conv4_x(x3)
        x4_y = self.dsc_conv4_y(x3)
        stage3 = self.conv4_3(cat([x4_1, x4_x, x4_y], dim=1))  # stage3:512x64x64
        x4 = self.maxpooling(stage3)

        # Encoder block 5
        x5_1 = self.conv5_1(x4)
        x5_x = self.dsc_conv5_x(x4)
        x5_y = self.dsc_conv5_y(x4)
        stage4 = self.conv5_3(cat([x5_1, x5_x, x5_y], dim=1))  # stage4:1024x32x32

        # 语义分支的旁支
        yuyi1_att = self.stage1_att(stage1)  # 128x256x256
        yuyi_d20 = self.down2_1up(stage2)  # 256x128x128-128x256x256
        yuyi_d2 = self.smooth2_1(yuyi_d20)  # 128x256x256-128x256x256
        yuyi_d30 = self.down3_1up(stage3)  # 512x64x64-256x128x128-128x256x256
        yuyi_d3 = self.smooth3_1(yuyi_d30)  # 128x256x256-128x256x256
        yuyi_d40 = self.down4_1up(stage4)  # 1024x32x32-512x64x64-256x128x128-128x256x256
        yuyi_d4 = self.smooth4_1(yuyi_d40)  # 128x256x256-128x256x256

        # 各不同维度特征进行add
        yuyi = yuyi1_att + yuyi_d2 + yuyi_d3 + yuyi_d4  # 128x256x256-128x256x256
        # 考虑语义后面加一层空间注意力模块，筛选有用特征
        yuyi1 = self.attention1(yuyi)  # 128x256x256-128x256x256
        yuyi2 = self.attention2(yuyi)  # 128x256x256-128x256x256
        yuyi3 = self.attention3(yuyi)  # 128x256x256-128x256x256
        # 上面输出的特征为128x256x256，作为边缘流特征融合的contact

        # 语义概览旁支通道注意力增强
        stage4_enhancment = self.enhancement(stage4, yuyi)

        # 解码阶段

        # Decoder block 0
        up4 = self.up4(stage4_enhancment)  # 1024x32x32-512x64x64
        up4 = cat([up4, stage3], dim=1)  # 1024x64x64
        up4 = self.conv6_1(up4)
        up4 = self.conv6_2(up4)  # decoder0:512x64x64

        # Decoder block 1
        up3 = self.up3(up4)
        up3 = cat([up3, stage2], dim=1)
        up3 = self.conv7_1(up3)
        up3 = self.conv7_2(up3)  # decoder1:256x128x128

        # Decoder block 2
        up2 = self.up2(up3)
        up2 = cat([up2, stage1], dim=1)
        up2 = self.conv8_1(up2)
        up2 = self.conv8_2(up2)  # decoder2:128x256x256

        # 边界分支
        xe = self.edge1.forward(x)  # 1x512x512
        xc = self.initial_conv(x)  # 21x512x512
        xe = torch.cat((xe, xc), dim=1)  # 22x512x512
        xe = self.edge_conv0(xe)  # 22x512x512-64x256x256
        # 语义与边界分支第一次融合
        xe = self.sgeem1(xe, yuyi1)  # 64x256x256-64x256x256
        xe = self.edge_conv1(xe)  # 64x256x256-64x256x256
        # 语义与边界分支第二次融合
        xe = self.sgeem2(xe, yuyi2)  # 64x256x256-64x256x256
        xe = self.edge_conv2(xe)  # 64x256x256-64x256x256
        # 语义与边界分支第3次融合
        xe = self.sgeem3(xe, yuyi3)  # 64x256x256-64x256x256

        # 边界分支与unet分支融合，contact
        # 融合时进行通道融合contact，然后再残差连接
        xeup = torch.cat([xe, up2], dim=1)  # 192x256x256
        # 调整通道数192-128
        xeup = self.convup1(xeup)  # 192x256x256-128x256x256
        xeup = self.attention4(xeup)  # 128x256x256-128x256x256
        up1 = self.up1(xeup)  # 128x256x256-64x512x512
        up1 = self.smooth_out(up1)  # 64x512x512

        up1 = torch.cat([up1, stage0], dim=1)

        up1 = self.up1_(up1)  # 128x512x512-64x512x512
        up1 = self.attention_out(up1)  # 64x512x512-64x512x512

        out_unet = self.out_conv(up1)  # 64x512x512-2x512x512

        # 边缘分支out
        out_edge = self.edge_head(xe)  # 64x256x256-1x512x512

        return out_unet, out_edge


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型
    model = TDSNet(
        in_channels=3,
        n_classes=1,
        kernel_size=9,
        extend_scope=1.0,
        if_offset=True,
        device=device,
        number=64,
    ).to(device)

    input_tensor = torch.randn(1, 3, 512, 512).to(device)
    output, out_edge = model(input_tensor)
    print(output.shape)
