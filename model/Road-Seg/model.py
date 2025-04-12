import torch
import torch.nn as nn
from torch.nn import init
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F

### network ###
class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        return output

class upsample_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upsample_layer, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        output = self.activation(x)
        return output


class RoadSeg(nn.Module):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """
    def __init__(self, num_labels, use_sne):
        super(RoadSeg, self).__init__()

        self.num_resnet_layers = 18

        if self.num_resnet_layers == 18:
            resnet_raw_model1 = torchvision.models.resnet18(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet18(pretrained=True)
            filters = [64, 64, 128, 256, 512]
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = torchvision.models.resnet34(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet34(pretrained=True)
            filters = [64, 64, 128, 256, 512]
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = torchvision.models.resnet50(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet50(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = torchvision.models.resnet101(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet101(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = torchvision.models.resnet152(pretrained=True)
            resnet_raw_model2 = torchvision.models.resnet152(pretrained=True)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError('num_resnet_layers should be 18, 34, 50, 101 or 152')

        ### encoder for another image ###
        if use_sne:
            self.encoder_another_conv1 = resnet_raw_model1.conv1
        else:
            # if another image is depth, initialize the weights of the first layer
            self.encoder_another_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.encoder_another_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1), dim=1)

        self.encoder_another_bn1 = resnet_raw_model1.bn1
        self.encoder_another_relu = resnet_raw_model1.relu
        self.encoder_another_maxpool = resnet_raw_model1.maxpool
        self.encoder_another_layer1 = resnet_raw_model1.layer1
        self.encoder_another_layer2 = resnet_raw_model1.layer2
        self.encoder_another_layer3 = resnet_raw_model1.layer3
        self.encoder_another_layer4 = resnet_raw_model1.layer4

        ###  encoder for rgb image  ###
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        ###  decoder  ###
        self.conv1_1 = conv_block_nested(filters[0]*2, filters[0], filters[0])
        self.conv2_1 = conv_block_nested(filters[1]*2, filters[1], filters[1])
        self.conv3_1 = conv_block_nested(filters[2]*2, filters[2], filters[2])
        self.conv4_1 = conv_block_nested(filters[3]*2, filters[3], filters[3])

        self.conv1_2 = conv_block_nested(filters[0]*3, filters[0], filters[0])
        self.conv2_2 = conv_block_nested(filters[1]*3, filters[1], filters[1])
        self.conv3_2 = conv_block_nested(filters[2]*3, filters[2], filters[2])

        self.conv1_3 = conv_block_nested(filters[0]*4, filters[0], filters[0])
        self.conv2_3 = conv_block_nested(filters[1]*4, filters[1], filters[1])

        self.conv1_4 = conv_block_nested(filters[0]*5, filters[0], filters[0])

        self.up2_0 = upsample_layer(filters[1], filters[0])
        self.up2_1 = upsample_layer(filters[1], filters[0])
        self.up2_2 = upsample_layer(filters[1], filters[0])
        self.up2_3 = upsample_layer(filters[1], filters[0])

        self.up3_0 = upsample_layer(filters[2], filters[1])
        self.up3_1 = upsample_layer(filters[2], filters[1])
        self.up3_2 = upsample_layer(filters[2], filters[1])

        self.up4_0 = upsample_layer(filters[3], filters[2])
        self.up4_1 = upsample_layer(filters[3], filters[2])

        self.up5_0 = upsample_layer(filters[4], filters[3])

        self.final = upsample_layer(filters[0], num_labels)


    def forward(self, rgb, another):
        # encoder
        rgb = self.encoder_rgb_conv1(rgb)
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)
        another = self.encoder_another_conv1(another)
        another = self.encoder_another_bn1(another)
        another = self.encoder_another_relu(another)
        rgb = rgb + another
        x1_0 = rgb
        print("x1_0",x1_0.size())

        rgb = self.encoder_rgb_maxpool(rgb)
        another = self.encoder_another_maxpool(another)
        rgb = self.encoder_rgb_layer1(rgb)
        another = self.encoder_another_layer1(another)
        rgb = rgb + another
        x2_0 = rgb
        print("x2_0",x2_0.size())

        rgb = self.encoder_rgb_layer2(rgb)
        another = self.encoder_another_layer2(another)
        rgb = rgb + another
        x3_0 = rgb
        print("x3_0",x3_0.size())
        rgb = self.encoder_rgb_layer3(rgb)
        another = self.encoder_another_layer3(another)
        rgb = rgb + another
        x4_0 = rgb
        print("x4_0",x4_0.size())
        rgb = self.encoder_rgb_layer4(rgb)
        another = self.encoder_another_layer4(another)
        x5_0 = rgb + another

        # decoder
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        print("self.up4_0(x4_0)",((self.up4_0(x4_0)).narrow(2, 0, 45)).size())
        print("x3_0",x3_0.size())

        x3_1 = self.conv3_1(torch.cat([x3_0, (self.up4_0(x4_0)).narrow(2, 0, 45)], dim=1))
        # print("x4_0",x4_0.size())
        # print("self.up5_0(x5_0)",self.up5_0(x5_0).size())
        x4_1 = self.conv4_1(torch.cat([x4_0, (self.up5_0(x5_0)).narrow(2, 0, 23)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))

        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, (self.up4_0(x4_0)).narrow(2, 0, 45)], dim=1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up3_2(x3_2)], dim=1))

        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up2_3(x2_3)], dim=1))
        out = self.final(x1_4)
        return out


# 测试代码
def main():
    # 实例化模型
    model = RoadSeg(num_labels=2, use_sne=False)  # 假设我们用的是2个标签进行分类

    # 创建输入张量
    rgb = torch.randn(1, 3, 360, 640)  # RGB图像
    another = torch.randn(1, 1, 360, 640)  # 另一个图像（如深度图或法线图）

    # 将模型设置为评估模式
    model.eval()

    # 前向传播
    with torch.no_grad():  # 关闭梯度计算
        output = model(rgb, another)

    # 打印输出尺寸
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
