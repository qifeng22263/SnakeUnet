import torch
from torch import nn,cat
from einops import rearrange


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


class DoubleDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, extend_scope=1.0, device="cuda"):
        super().__init__()
        self.conv1_x = DSConv_pro(in_channels, out_channels, kernel_size, extend_scope, 0, True, device)
        self.conv1_y = DSConv_pro(in_channels, out_channels, kernel_size, extend_scope, 1, True, device)
        self.conv2_x = DSConv_pro(out_channels, out_channels, kernel_size, extend_scope, 0, True, device)
        self.conv2_y = DSConv_pro(out_channels, out_channels, kernel_size, extend_scope, 1, True, device)

    def forward(self, x):
        x = self.conv1_x(x) + self.conv1_y(x)
        x = self.conv2_x(x) + self.conv2_y(x)
        return x


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


class SnakeUNet(nn.Module):
    def __init__(
        self,
        n_channels:int=3,
        n_classes:int=1,
        kernel_size:int=9,
        extend_scope:float=1.0,
        if_offset:bool=True,
        device:str="cuda",
        number:int=64,
    ):
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number

        # Encoder
        self.conv1_1 = EncoderConv(n_channels, self.number)
        self.dsc_conv1_x = DSConv_pro(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.dsc_conv1_y = DSConv_pro(
            n_channels,
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

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv5_1 = DecoderConv(12 * self.number, 4 * self.number)
        self.conv5_2 = DecoderConv(4 * self.number, 4 * self.number)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv6_1 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv6_2 = DecoderConv(2 * self.number, 2 * self.number)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv7_1 = DecoderConv(3 * self.number, self.number)
        self.conv7_2 = DecoderConv(self.number, self.number)

        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        self.maxpooling = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder block 1
        x1_1 = self.conv1_1(x)
        x1_x = self.dsc_conv1_x(x)
        x1_y = self.dsc_conv1_y(x)
        x1 = self.conv1_3(cat([x1_1, x1_x, x1_y], dim=1))
        x = self.maxpooling(x1)

        # Encoder block 2
        x2_1 = self.conv2_1(x)
        x2_x = self.dsc_conv2_x(x)
        x2_y = self.dsc_conv2_y(x)
        x2 = self.conv2_3(cat([x2_1, x2_x, x2_y], dim=1))
        x = self.maxpooling(x2)

        # Encoder block 3
        x3_1 = self.conv3_1(x)
        x3_x = self.dsc_conv3_x(x)
        x3_y = self.dsc_conv3_y(x)
        x3 = self.conv3_3(cat([x3_1, x3_x, x3_y], dim=1))
        x = self.maxpooling(x3)

        # Encoder block 4
        x4_1 = self.conv4_1(x)
        x4_x = self.dsc_conv4_x(x)
        x4_y = self.dsc_conv4_y(x)
        x4 = self.conv4_3(cat([x4_1, x4_x, x4_y], dim=1))

        # Decoder block 1
        x = self.up1(x4)
        x = cat([x, x3], dim=1)
        x = self.conv5_1(x)
        x = self.conv5_2(x)

        # Decoder block 2
        x = self.up2(x)
        x = cat([x, x2], dim=1)
        x = self.conv6_1(x)
        x = self.conv6_2(x)

        # Decoder block 3
        x = self.up3(x)
        x = cat([x, x1], dim=1)
        x = self.conv7_1(x)
        x = self.conv7_2(x)

        out = self.out_conv(x)
        out = self.sigmoid(out)

        return out



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型
    model = SnakeUNet(
        n_channels=3,
        n_classes=1,
        kernel_size=9,
        extend_scope=1.0,
        if_offset=True,
        device=device,
        number=64,
    ).to(device)

    input_tensor = torch.randn(1, 3, 512, 512).to(device)
    output = model(input_tensor)
    print(output.shape)
