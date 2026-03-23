
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

bn_mom = 0.1

#bianyuan detector
class SobelEdgeDetector:
    def __init__(self, device):
        self.device = device

    def preprocess(self, tensor):
        """
        预处理输入张量，确保其为灰度图。
        :param tensor: 输入的PyTorch张量，形状为 (B, C, H, W)
        """
        if tensor.size(1) == 3:
            # 将RGB图像转换为灰度图
            gray_tensor = torch.mean(tensor, dim=1, keepdim=True)
        elif tensor.size(1) == 1:
            gray_tensor = tensor
        else:
            raise ValueError("Input tensor must be a 3-channel RGB image or a single-channel grayscale image.")

        # 将通道从通道在前转换为通道在后
        gray_tensor = gray_tensor.permute(0, 2, 3, 1).to(self.device)
        return gray_tensor

    def sobel(self, tensor):
        """
        使用Sobel算子检测边缘。
        :param tensor: 输入的PyTorch张量，形状为 (B, H, W, C)
        """
        tensor = tensor.cpu().detach().numpy()

        # 逐个处理批次中的每个图像
        edges = []
        for i in range(tensor.shape[0]):
            image = tensor[i]
            if len(image.shape) == 3 and image.shape[2] == 1:
                image = image.squeeze(2)  # 移除单通道维度
            gray_image = image

            # 使用OpenCV的Sobel函数来提取边缘,此处参数可以调整，将ksize可以调整为1、3、5等，对于裂缝，越大引入噪声越多，所以本次训练试一下1和3的效果
            sobel_x = cv2.Sobel(gray_image, -1, 1, 0, ksize=1)
            sobel_y = cv2.Sobel(gray_image, -1, 0, 1, ksize=1)
            #print(sobel_x.shape)
            edge = cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0)

            # 将边缘图转换回PyTorch张量
            edge_tensor = torch.from_numpy(edge).unsqueeze(0).unsqueeze(0).to(self.device)
            edges.append(edge_tensor)

        # 将边缘图列表合并为一个张量
        return torch.cat(edges, dim=0)

    def forward(self, tensor):
        """
        处理输入张量，执行边缘检测。
        :param tensor: 输入的PyTorch张量，形状为 (B, C, H, W)
        """
        # 预处理
        tensor = self.preprocess(tensor)
        # Sobel边缘检测
        edges = self.sobel(tensor)
        return edges
#--------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)




#-------------一、SENet模块,纯通道注意力-----------------------------
# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)

#------金字塔池化模块
class PP_Model(nn.Module):
    def __init__(self,in_channels,K):
        super(PP_Model, self).__init__()
        self.in_channels=in_channels
        #self.out_channels=out_channels
        # 对输入n，H,W特征infeture进行4种尺度的平均池化
        # 得到4种尺度的池化结果p1、p2、p3、p4-1、2、4、8
        # 对每个尺度池化结果进行1x1卷积处理，调整n为n/4，得到pc1、pc2、pc3、pc4
        # 对pc1、pc2、pc3、pc4进行上采样为特征原尺寸H,W的特征pco1、pco2、pco3、pco4
        # contact连接infeture和特征pco1、pco2、pco3、pco4
        # 得到模块输出2n,H,W

        #以下是自适应平均值池化
        '''
        self.pypool1=nn.AdaptiveAvgPool2d(1)
        self.pypool2=nn.AdaptiveAvgPool2d(2)
        self.pypool3=nn.AdaptiveAvgPool2d(4)
        self.pypool4=nn.AdaptiveAvgPool2d(8)
        '''

        # 以下自适应池化选择为最大值池化，突出前景
        self.pypool1=nn.AdaptiveMaxPool2d(1)
        self.pypool2=nn.AdaptiveMaxPool2d(2)
        self.pypool3=nn.AdaptiveMaxPool2d(4)
        self.pypool4=nn.AdaptiveMaxPool2d(8)



        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels//4,kernel_size=1,stride=1,padding=0,bias=True)
        self.up1=nn.Upsample(scale_factor=K)
        self.up2=nn.Upsample(scale_factor=K//2)
        self.up3 = nn.Upsample(scale_factor=K//4)
        self.up4 = nn.Upsample(scale_factor=K//8)

    def forward(self,x):
        p1=self.pypool1(x)   #64x512-64x1
        p2=self.pypool2(x)    #64x512-64x2
        p3=self.pypool3(x)    #64x512-64x4
        p4=self.pypool4(x)    #64x512-64x8
        pc1=self.conv1(p1)    #64x1-16x1
        pc2=self.conv1(p2)    #64x2-16x2
        pc3=self.conv1(p3)    #64x4-16x4
        pc4=self.conv1(p4)     #64x8-16x8
        pco1=self.up1(pc1)    #16x1-16x512
        pco2=self.up2(pc2)    #16x2-16x512
        pco3=self.up3(pc3)    #16x4-16x512
        pco4=self.up4(pc4)    #16x8-16x512
        out1=torch.cat((pco1,pco2),dim=1)  #32x512
        out2=torch.cat((out1,pco3),dim=1)   #48x512
        out3=torch.cat((out2,pco4),dim=1)   #64x512
        out4=torch.cat((out3,x),dim=1)      #128x512
        return out4
#------------------
#----语义边缘注意力模块
#----CED model
#
class CED(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,dilation=1, groups=1, bias=False):
        super(CED, self).__init__()

        self.attention = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),

            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=kernel_size),

            nn.Sigmoid()
        )
        self.outconv = nn.Conv2d(out_channels,out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x,y):
        #语义分支的特征与边缘分支contact
        #xy=torch.cat((x,y),dim=1)
        #语义分支的特征与边缘分支相加add
        xy = x+y
        #print(xy.shape)
        attention_mask=self.attention(xy)
        #print(attention_mask.shape)
        out=(x*(attention_mask+1))
        #print(out.shape)
        out=self.outconv(out)
        return out


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d):
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
            BatchNorm(in_channels),
            nn.Conv2d(in_channels, out_channels,kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, bone,edge):
        '''
        #细节分支作为注意力
        edge_att = torch.sigmoid(edge)
        return self.conv(bone * (1 + edge_att))
        '''
        #尝试边缘直接加在语义分支上面

        return self.conv(bone+edge)


class SGEEM(nn.Module):
    """语义引导边缘增强模块（单层级）"""

    def __init__(self, channels_edge,channels_yuyi):
        super().__init__()
        # 语义特征压缩（1x1卷积降维）
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(channels_yuyi, channels_yuyi // 4, 1),
            nn.BatchNorm2d(channels_yuyi // 4),
            nn.ReLU()
        )

        # 空间注意力生成
        self.attention = nn.Sequential(
            nn.Conv2d(channels_yuyi // 4, 1, 1),
            nn.Sigmoid()  # 生成0-1的空间权重
        )

        # 边缘特征增强
        self.edge_conv = nn.Conv2d(channels_edge, channels_edge, 3, padding=1)

    def forward(self, edge_feat, semantic_feat):
        # 1. 压缩语义特征
        compressed_semantic = self.semantic_conv(semantic_feat)    #128x256x256-32x256x256

        # 2. 生成注意力图
        attention_map = self.attention(compressed_semantic)       #32x256x256-1x256x256

        # 3. 应用注意力到边缘特征
        guided_edge = edge_feat * attention_map    #

        # 4. 增强处理
        return self.edge_conv(guided_edge) + edge_feat  # 残差连接

class NOSGEEM(nn.Module):
    """语义引导边缘增强模块（单层级）"""

    def __init__(self, channels_edge):
        super().__init__()



        # 边缘特征增强
        self.edge_conv = nn.Conv2d(channels_edge, channels_edge, 3, padding=1)

    def forward(self, edge_feat):

        # 4. 增强处理
        return self.edge_conv(edge_feat)  # 残差连接




#--------深层语义信息融合注意力

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CrossNetworkFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # 通道调整与对齐
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()
        # 使用反卷积进行上采样
        self.upsample = nn.ConvTranspose2d(c1, c2, kernel_size=2, stride=2)

        # 空间注意力权重生成
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c2 * 2, c2 // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2 // 4),
            nn.ReLU(),
            nn.Conv2d(c2 // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 通道注意力权重生成
        self.channel_attention = ChannelAttention(c2)
        # 残差卷积
        self.res_conv = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU()
        )

    def forward(self, feat_a, feat_b):
        # 对齐与通道适配
        #feat_b = self.upsample(feat_b)
        #feat_b = self.conv1(feat_b)

        # 空间注意力权重
        combined = torch.cat([feat_a, feat_b], dim=1)
        spatial_weight = self.spatial_attention(combined)
        # 通道注意力权重
        channel_weight = self.channel_attention(feat_a + feat_b)

        # 加权求和
        spatial_fused = spatial_weight * feat_a + (1 - spatial_weight) * feat_b
        channel_fused = channel_weight * spatial_fused

        # 残差增强
        return channel_fused + self.res_conv(channel_fused)

'''
# 使用示例
# 创建 CrossNetworkFusion 模块的实例
fusion_module = CrossNetworkFusion(c1=1024, c2=1024)
# 生成随机的特征图 feat_a 和 feat_b
feat_a = torch.randn(2, 1024, 32, 32)  # 网络 A 的深层特征
feat_b = torch.randn(2, 1024, 32, 32)  # 网络 B 的深层特征
# 调用融合模块进行特征融合
fused_feat = fusion_module(feat_a, feat_b)

'''
#------------------------



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    '''
    x= torch.randn(2,128, 256, 256).cuda()
    y= torch.randn(2,128, 256, 256).cuda()
    ced=CED(256,128).cuda()
    out=ced(x,y)
    print(out.shape)
    '''
    detector = SobelEdgeDetector(device=device)
    img_tensor=torch.randn(2,3,512,512).cuda()
    edges_tensor = detector.forward(img_tensor)


    print(edges_tensor.shape)


