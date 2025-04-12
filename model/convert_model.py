import torch
from models import Baseline  # 修改为你的实际模型路径和类名
import torch.nn as nn
import torch.nn.functional as Func
from torchvision.models.resnet import BasicBlock

class DKPF(nn.Module):
    def __init__(self, rgb_dim, depth_dim, norm_layer=None, kernel_sizes=[3, 5, 7]):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_dim, depth_dim // 4, 3, padding=1),
            norm_layer(depth_dim // 4),
            nn.ReLU()
        )

        # 动态卷积核生成（基于Depth几何特征）
        self.kernel_gen = nn.Conv2d(depth_dim // 4, sum([k * k for k in kernel_sizes]), 1)

        # 多尺度卷积分支（不同kernel size）
        self.conv_branches = nn.ModuleList([
            nn.Conv2d(rgb_dim, rgb_dim, k, padding=k // 2, groups=rgb_dim)
            for k in kernel_sizes
        ])

        # 多尺度门控权重生成
        self.gate_net = nn.Sequential(
            nn.Conv2d(rgb_dim + depth_dim, len(kernel_sizes), 3, padding=1),
            nn.Softmax(dim=1)
        )
        self.res_conv = nn.Conv2d(rgb_dim, rgb_dim, 1)

    def forward(self, rgb_feat, depth_feat):
        """
        Input:
            rgb_feat:   [B, C, H, W] 上采样后的RGB特征
            depth_feat: [B, D, H, W] 上采样后的Depth特征（或Surface Normal）
        Output:
            fused_feat: [B, C, H, W] 融合后的特征
        """

        B, C, H, W = rgb_feat.shape

        # Step 1: 基于Depth生成动态卷积核 -------------------------------------------------
        # 基于 Depth 特征生成动态卷积核
        depth_enc = self.depth_encoder(depth_feat)  # [B, D/4, H, W]
        kernel_weights = self.kernel_gen(depth_enc)  # [B, sum(k^experiment), H, W]

        # 拆分不同尺度的核参数
        kernels = []
        ptr = 0
        for k in self.conv_branches:
            k_size = k.kernel_size[0]  # 获取当前卷积分支的核大小
            kernels.append(kernel_weights[:, ptr:ptr + k_size * k_size, :, :])
            ptr += k_size * k_size

        # Step experiment: 多尺度动态卷积 --------------------------------------------------------
        multi_scale_feats = []
        for conv, kernel in zip(self.conv_branches, kernels):
            # 动态卷积实现（Depth引导的卷积核）
            feat = conv(rgb_feat)  # 普通卷积
            feat = self._dynamic_conv(feat, kernel, k=conv.kernel_size[0])  # dynamic_conv
            multi_scale_feats.append(feat)

        # Step 3: 自适应多尺度融合 -----------------------------------------------------
        gate_input = torch.cat([rgb_feat, depth_feat], dim=1)
        gates = self.gate_net(gate_input)  # [B, num_scales, H, W]

        fused = 0
        for i in range(len(multi_scale_feats)):
            fused += multi_scale_feats[i] * gates[:, i:i + 1, :, :]  # 空间自适应加权

        # Step 4: 残差连接 -----------------------------------------------------------
        return fused + self.res_conv(rgb_feat)  # 保留原始特征

    def _dynamic_conv(self, feat, kernel, k):
        """
        动态卷积实现（核参数来自 Depth 分支）
        feat:   [B, C, H, W]
        kernel: [B, 1, k*k, H, W]
        """
        B, C, H, W = feat.shape

        # 确保 kernel 的形状为 [B, 1, k*k, H, W]
        if kernel.dim() == 4:
            kernel = kernel.unsqueeze(1)  # 添加缺失的通道维度

        # 展开输入特征，获得 [B, C*k*k, H*W]
        feat_unfold = Func.unfold(feat, kernel_size=k, padding=k // 2)  # [B, C*k*k, H*W]
        feat_unfold = feat_unfold.view(B, C, k * k, H, W)  # [B, C, k*k, H, W]

        # 扩展 kernel 到 [B, C, k*k, H, W]
        kernel = kernel.expand(-1, C, -1, -1, -1)

        # 动态卷积
        dynamic_feat = torch.einsum('bckhw,bckhw->bchw', feat_unfold, kernel)  # [B, C, H, W]
        return dynamic_feat


# 实例化模型结构
model = Baseline(guide = DKPF)
model.load_state_dict(torch.load("ai/ckpts/RELLIS_3D-22-02-25-03:06:40/model_best.pth"))  # 加载参数
model.eval().to('cuda')  # 设置为评估模式，并转到 GPU

# 创建 dummy inputs
rgb_input = torch.randn(1, 3, 360, 640, device='cuda')
depth_input = torch.randn(1, 1, 360, 640, device='cuda')

# 导出 ONNX
torch.onnx.export(
    model,
    (rgb_input, depth_input),
    "DualBranchModel.onnx",
    input_names=["rgb_input", "depth_input"],
    output_names=["output_1"],
    opset_version=11,
    verbose=True
)
