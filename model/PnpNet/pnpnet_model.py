import torch
from torch import nn
from torch.nn import functional as F
import extractors
import torch
from torch import nn
from torch.nn import functional as F


# PSP Module
class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):  # 修改输出特征数为 512
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)  # 改为 512 输出
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


# PSP Upsample Module
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=False)
        return self.conv(p)


# PSPNet Model
class PSPNet(nn.Module):
    def __init__(self, n_classes=2, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=1024, backend='resnet34',
                 pretrained=True):
        super().__init__()
        # 从extractors中获取backbone模型，并修改conv1使其支持4通道输入
        self.feats = getattr(extractors, backend)(pretrained)
        self.feats.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改conv1为接受4通道输入

        self.psp = PSPModule(psp_size, 512, sizes)  # 修改PSPModule中的通道数为 512
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(512, 256)  # 修改上采样的输入输出通道数
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax(dim=1)  # 对每个类别进行log softmax
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        final = self.final(p)
        return final


# 测试代码
def main():
    # 创建一个输入张量，形状为 (1, 4, 360, 640)
    input_tensor = torch.randn(1, 4, 360, 640)

    # 设置 PSPNet 参数
    n_classes = 2  # 假设你有2个类别
    psp_size = 512
    deep_features_size = 1024
    backend = 'resnet34'  # 使用ResNet34作为基础网络
    pretrained = False  # 是否使用预训练模型

    # 实例化 PSPNet 模型
    model = PSPNet(n_classes=n_classes,
                   sizes=(1, 2, 3, 6),
                   psp_size=psp_size,
                   deep_features_size=deep_features_size,
                   backend=backend,
                   pretrained=pretrained)
    # 将模型设置为评估模式（推理模式）
    model.eval()
    # 将输入张量传入模型并进行前向传播
    with torch.no_grad():  # 关闭梯度计算，节省内存
        output = model(input_tensor)

    print("input_tensor:", input_tensor.size())
    print("Output shape:", output.shape)  # 输出分割结果的形状

if __name__ == "__main__":
    main()
