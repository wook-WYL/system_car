import torch
import torch.nn as nn
import encoding
from scipy.stats import truncnorm
import math

# 假设 BasicBlock 已经被定义，或者你从 torchvision 导入
from torchvision.models.resnet import BasicBlock

from utils_nn import *

# 定义 STANDARD 等对象
STANDARD = "standard"  # 这里只是一个示例，使用合适的对象或数据

__all__ = [
    'STANDARD4',
    'GUIDE4',
    'GFL4',
    'DKPF4'
]


def STANDARD4():
    return Baseline(norm_layer=encoding.nn.SyncBatchNorm, guide=STANDARD, weight_ks=3, bc=4)


def GUIDE4():
    return Baseline(norm_layer=encoding.nn.SyncBatchNorm, guide=GUIDE, weight_ks=1, bc=4)


def GFL4():
    return Baseline(norm_layer=encoding.nn.SyncBatchNorm, guide=GFL, weight_ks=1, bc=4)


def DKPF4():
    return Baseline(norm_layer=encoding.nn.SyncBatchNorm, guide=DKPF, weight_ks=1, bc=4)


# Baseline 类的实现
class Baseline(nn.Module):

    def __init__(self, guide, block=BasicBlock, bc=16, \
                 img_layers=[2, 2, 2, 2, 2], depth_layers=[2, 2, 2, 2, 2],
                 norm_layer=nn.BatchNorm2d, weight_ks=3, in_channels_rgb=3, in_channels_depth=1,kernel_sizes=[3,5,7]):
        super().__init__()

        # rgb feature
        self._norm_layer = norm_layer
        in_channels = bc * 2  # H W C
        self.layer1_0 = Basic2d(in_channels_rgb, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)  # 2c,1
        self.inplanes = in_channels
        self.layer1_1 = self._make_layer(block, in_channels * 2, img_layers[0], stride=2)  # 4c 1/2
        self.inplanes = in_channels * 2 * block.expansion
        self.layer1_2 = self._make_layer(block, in_channels * 4, img_layers[1], stride=2)  # 8c 1/4
        self.inplanes = in_channels * 4 * block.expansion
        self.layer1_3 = self._make_layer(block, in_channels * 8, img_layers[2], stride=2)  # 16c 1/8
        self.inplanes = in_channels * 8 * block.expansion
        self.layer1_4 = self._make_layer(block, in_channels * 8, img_layers[3], stride=2)  # 16c 1/16
        self.inplanes = in_channels * 8 * block.expansion
        self.layer1_5 = self._make_layer(block, in_channels * 8, img_layers[4], stride=2)  # 16c 1/32

        self.layer2_0 = Basic2d(in_channels_depth, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)  # 2c,1
        self.inplanes = in_channels
        self.layer2_1 = self._make_layer(block, in_channels * 2, img_layers[0], stride=2)  # 4c 1/2
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_2 = self._make_layer(block, in_channels * 4, img_layers[1], stride=2)  # 8c 1/4
        self.inplanes = in_channels * 4 * block.expansion
        self.layer2_3 = self._make_layer(block, in_channels * 4, img_layers[2], stride=2)  # 16c 1/8
        self.inplanes = in_channels * 4 * block.expansion
        self.layer2_4 = self._make_layer(block, in_channels * 4, img_layers[3], stride=2)  # 16c 1/16

        self.dlayer2_4 = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)  # 16c 1/8
        self.dlayer2_3 = Basic2dTrans(in_channels * 16, in_channels * 8, norm_layer)  # 8c 1/4
        self.dlayer2_2 = Basic2dTrans(in_channels * 16, in_channels * 4, norm_layer)  # 4c 1/2
        self.dlayer2_1 = Basic2dTrans(in_channels * 8, in_channels * 2, norm_layer)  # 2c 1/2
        self.dlayer2_0 = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)

        self.dlayer3_4 = Basic2dTrans(in_channels * 4, in_channels * 4, norm_layer)  # 16c 1/8
        self.dlayer3_3 = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)  # 8c 1/4
        self.dlayer3_2 = Basic2dTrans(in_channels * 8, in_channels * 2, norm_layer)  # 4c 1/2
        self.dlayer3_1 = Basic2dTrans(in_channels * 4, in_channels * 1, norm_layer)  # 2c 1/2
        self.dlayer3_0 = Basic2dTrans(in_channels * 2, in_channels * 1, norm_layer)

        # Define CBAM after some key layers
        self.cbam_rgb = CBAM(512, reduction_ratio=16)

        self.conv1x1_0 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.upsample0 = nn.Upsample(size=(45, 80), mode='bilinear', align_corners=False)
        self.conv1x1_1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.upsample2 = nn.Upsample(size=(90, 160), mode='bilinear', align_corners=False)

        # Output layer to match the target output size
        self.layer2_output = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1,
                                       padding=1)  # 3 channels for RGB output

        self.guide1 = DKPF(256, 256, norm_layer, kernel_sizes)
        self.guide2 = DKPF(256, 256, norm_layer, kernel_sizes)
        self.guide3 = DKPF(128 , 128, norm_layer, kernel_sizes)
        # self.guide3 = DKPF(in_channels * 8, in_channels * 8, norm_layer, kernel_sizes)
        # self.guide4 = DKPF(in_channels * 8, in_channels * 8, norm_layer, kernel_sizes)

        self.layer5_1 = block(in_channels * 4, in_channels * 4, norm_layer=norm_layer, act=False)
        self.upsample3 = nn.Upsample(size=(180, 360), mode='bilinear', align_corners=False)
        self.layer5_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels * 4, in_channels * 2, 1))
        self.upsample4 = nn.Upsample(size=(360, 640), mode='bilinear', align_corners=False)

    def forward(self, rgb, depth, check=True):
        # RGB feature processing
        x1_0 = self.layer1_0(rgb)  # 2c
        x1_1 = self.layer1_1(x1_0)  # 4c 1/2
        x1_2 = self.layer1_2(x1_1)  # 8c 1/4
        x1_3 = self.layer1_3(x1_2)  # 16c 1/8
        x1_4 = self.layer1_4(x1_3)  # 32c 1/16
        x1_5 = self.layer1_5(x1_4)  # 32c 1/32

        if check:
            print("                           ")
            print("x1_0        :", x1_0.size())
            print("x1_1        :", x1_1.size())
            print("x1_2        :", x1_2.size())
            print("x1_3        :", x1_3.size())
            print("x1_4        :", x1_4.size())
            print("x1_5        :", x1_5.size())
        dx1_4 = self.dlayer2_4(x1_5)
        dx1_4 = dx1_4[:, :, :x1_4.size(2), :x1_4.size(3)]
        dx1_4_cat = torch.cat([dx1_4, x1_4], 1)

        dx1_3 = self.dlayer2_3(dx1_4_cat)
        dx1_3 = dx1_3[:, :, :x1_3.size(2), :x1_3.size(3)]
        dx1_3_cat = torch.cat([dx1_3, x1_3], 1)

        dx1_2 = self.dlayer2_2(dx1_3_cat)
        dx1_2 = dx1_2[:, :, :x1_2.size(2), :x1_2.size(3)]
        dx1_2_cat = torch.cat([dx1_2, x1_2], 1)

        dx1_1 = self.dlayer2_1(dx1_2_cat)
        dx1_1 = dx1_1[:, :, :x1_1.size(2), :x1_1.size(3)]
        dx1_1_cat = torch.cat([dx1_1, x1_1], 1)

        dx1_0 = self.dlayer2_0(dx1_1_cat)
        output_rgb = dx1_0
        if check:
            print("                           ")
            print("dx1_4       :", dx1_4.size())
            print("dx1_4_cat   :", dx1_4_cat.size())
            print("dx1_3       :", dx1_3.size())
            print("dx1_3_cat   :", dx1_3_cat.size())
            print("dx1_2       :", dx1_2.size())
            print("dx1_2_cat   :", dx1_2_cat.size())
            print("dx1_1       :", dx1_1.size())
            print("dx1_1_cat   :", dx1_1_cat.size())
            print("dx1_0       :", dx1_0.size())
        # Surface Normal processing (depth feature)
        x2_0 = self.layer2_0(depth)  # 2c
        x2_1 = self.layer2_1(x2_0)  # 4c 1/2
        x2_2 = self.layer2_2(x2_1)  # 8c 1/4
        x2_3 = self.layer2_3(x2_2)  # 16c 1/8
        x2_4 = self.layer2_4(x2_3)  # 32c 1/16
        if check:
            print("Depth Branch:")
            print("x2_0        :", x2_0.size())
            print("x2_1        :", x2_1.size())
            print("x2_2        :", x2_2.size())
            print("x2_3        :", x2_3.size())
            print("x2_4        :", x2_4.size())

        dx2_4 = self.dlayer3_4(x2_4)
        dx2_4 = dx2_4[:, :, :x2_3.size(2), :x2_3.size(3)]
        dx2_4_cat = torch.cat([dx2_4, x2_3], 1)
        dx2_3 = self.dlayer3_3(dx2_4_cat)
        dx2_3 = dx2_3[:, :, :x2_2.size(2), :x2_2.size(3)]
        dx2_3_cat = torch.cat([dx2_3, x2_2], 1)
        dx2_2 = self.dlayer3_2(dx2_3_cat)
        dx2_2 = dx2_2[:, :, :x2_1.size(2), :x2_1.size(3)]
        dx2_2_cat = torch.cat([dx2_2, x2_1], 1)
        dx2_1 = self.dlayer3_1(dx2_2_cat)
        dx2_1 = dx2_1[:, :, :x2_0.size(2), :x2_0.size(3)]

        output_depth = self.layer2_output(dx2_1)  # Final RGB output for depth
        if check:
            print("                           ")
            print("dx2_4       :", dx2_4.size())
            print("dx2_4_cat   :", dx2_4_cat.size())
            print("dx2_3       :", dx2_3.size())
            print("dx2_3_cat   :", dx2_3_cat.size())
            print("dx2_2       :", dx2_2.size())
            print("dx2_2_cat   :", dx2_2_cat.size())
            print("dx2_1       :", dx2_1.size())

            # print("dx2_1_cat   :", dx2_1_cat.size())
            # print("dx2_0       :", dx2_0.size())

        x2_4 = self.conv1x1_1(x2_4)
        x1_5_up = Func.interpolate(x1_5, size=(x2_4.size(2), x2_4.size(3)), mode='bilinear', align_corners=False)
        x_combined = torch.cat([x1_5_up, x2_4], dim=1)
        x_cbam = self.conv1x1_0(self.cbam_rgb(x_combined))

        print('x1_5_up',x1_5_up.size())# 256 12 20
        print('x_cbam',x_cbam.size()) #


        dx2_4_up = Func.interpolate(dx2_4_cat, size=(dx1_4.size(2), dx1_4.size(3)), mode='bilinear', align_corners=False)
        f1 = self.guide1(dx1_4,dx2_4_up) #####################
        print('f1.size()',f1.size())# 256 23 40

        f1_cat = self.upsample0(self.conv1x1_0(torch.cat([f1, x_cbam], dim=1)))# 512 23 40 -> 256 23 40 -> 256 45 80
        print('dx2_4_up_cat', f1_cat.size())


        dx2_3_up = Func.interpolate(dx2_3_cat, size=(dx1_3.size(2), dx1_3.size(3)), mode='bilinear', align_corners=False)
        print('dx2_3_up', dx2_3_up.size())  # 128 90 60
        print('dx1_3', dx1_3.size())# torch.Size([1, 256, 45, 80])
        f2 = self.guide2(dx1_3, dx2_3_up)# torch.Size([1, 256, 45, 80]) ###############
        dx2_3_up_cat =  f2 + f1_cat
        print('f2.size()',f2.size())
        print('dx2_3_up_cat.size()',dx2_3_up_cat.size())
        f2_reshape = self.upsample2(self.conv1x1_2(dx2_3_up_cat))# 256 45 80 -> 256 90 160


        dx2_2_up = Func.interpolate(dx2_2_cat, size=(dx1_2.size(2), dx1_2.size(3)), mode='bilinear',
                                    align_corners=False)
        print('dx2_2_up', dx2_2_up.size())  # 128 90 160
        print('dx1_2', dx1_2.size())  # torch.Size([1, 128. 90, 160])
        f3 = self.guide3(dx1_2, dx2_2_up)
        print('f3.size()', f3.size())

        dx2_2_up_cat = f3 + f2_reshape # 残差连接 torch.Size([1, 128, 90, 160])

        print('dx2_2_up_cat.size()',dx2_2_up_cat.size()) # torch.Size([1, 128, 90, 160])

        fusion_layer_output = self.upsample3(self.layer5_1(dx2_2_up_cat))
        fusion_layer_output = self.upsample4(self.layer5_2(fusion_layer_output))
        print('fusion_layer_output.size()',fusion_layer_output.size()) # torch.Size([1, 3, 90, 160])

        fusion_layer_output_with_rgb = fusion_layer_output + output_rgb

        return output_rgb, output_depth, fusion_layer_output_with_rgb

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)


def main():
    device = torch.device('cpu')

    rgb_input = torch.randn(1, 3, 360, 640).to(device)  # 随机初始化一个张量，形状为 [batch_size, channels, height, width]
    depth_input = torch.randn(1, 1, 360, 640).to(device)  # 随机初始化一个张量，形状为 [batch_size, channels, height, width]
    # 假设你已经定义了 STANDARD， GUIDE， GFL 或 DKPF 等
    guide = STANDARD  # 这里可以选择传入 STANDARD、GUIDE、GFL 或 DKPF

    # 创建Baseline模型，并传入 guide 参数
    model = Baseline(guide=guide, block=BasicBlock)  # 使用BasicBlock替代Identity
    model = model.to(device)

    # 前向传播
    output_rgb, output_depth,fusion_layer_output_with_rgb = model(rgb_input, depth_input)

    # 打印输出的尺寸
    print(f"Input shape: {rgb_input.shape}")
    print(f"Output RGB shape: {output_rgb.shape}")
    print(f"Output Depth shape: {output_depth.shape}")
    print(f"fusion_layer_output_with_rgbshape: {fusion_layer_output_with_rgb.shape}")


if __name__ == "__main__":
    main()
