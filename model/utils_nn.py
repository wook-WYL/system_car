import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn import functional as Func
import GuideConv
import torch.nn.functional as F

def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Conv2dLocal_F(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = GuideConv.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = GuideConv.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight


class Conv2dLocal(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, input, weight):
        output = Conv2dLocal_F.apply(input, weight)
        return output


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Basic2dLocal(nn.Module):
    def __init__(self, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = Conv2dLocal()
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, weight):
        out = self.conv(input, weight)
        out = self.bn(out)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out

class RW_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, shrink_factor):
        super(RW_Module, self).__init__()
        self.chanel_in = in_dim
        self.shrink_factor = shrink_factor

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def own_softmax1(self, x):
    
        maxes1 = torch.max(x, 1, keepdim=True)[0]
        maxes2 = torch.max(x, 2, keepdim=True)[0]
        x_exp = torch.exp(x-0.5*maxes1-0.5*maxes2)
        x_exp_sum_sqrt = torch.sqrt(torch.sum(x_exp, 2, keepdim=True))

        return (x_exp/x_exp_sum_sqrt)/torch.transpose(x_exp_sum_sqrt, 1, 2)
    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        x_shrink = x
        m_batchsize, C, height, width = x.size()
        if self.shrink_factor != 1:
            height = (height - 1) // self.shrink_factor + 1
            width = (width - 1) // self.shrink_factor + 1
            x_shrink = Func.interpolate(x_shrink, size=(height, width), mode='bilinear', align_corners=True)            
        
            proj_query = self.query_conv(x_shrink).view(m_batchsize, -1, width*height).permute(0, 2, 1) # (B,H*W,C) Q
            proj_key = self.key_conv(x_shrink).view(m_batchsize, -1, width*height)# (B,C,H*W) K
        
            energy = torch.bmm(proj_query, proj_key)#(B,H*W,H*W) Q*K

            attention = self.softmax(energy) # A = softmax(F^T F)

        proj_value = self.value_conv(x_shrink).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # AF
        out = out.view(m_batchsize, C, height, width)
        
        if self.shrink_factor != 1:
            height = (height - 1) * self.shrink_factor + 1
            width = (width - 1) * self.shrink_factor + 1
            out = Func.interpolate(out, size=(height, width), mode='bilinear', align_corners=True)

        out = self.gamma*out + x # F' = αAF + F
        return out,energy

class STANDARD(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Basic2d(input_planes + weight_planes, input_planes + weight_planes, norm_layer, kernel_size=weight_ks)
        self.conv2 = Basic2d(input_planes + weight_planes, input_planes + weight_planes, norm_layer, kernel_size=weight_ks)
        self.conv3 = Basic2d(input_planes + weight_planes, input_planes + weight_planes, norm_layer, kernel_size=weight_ks)
        self.conv4 = Basic2d(input_planes + weight_planes, input_planes + weight_planes, norm_layer, kernel_size=1, padding=0)
        self.conv5 = Basic2d(input_planes + weight_planes, input_planes, norm_layer)

    def forward(self, input, weight):
        
        x = torch.cat([input, weight], 1)        
        x1 = self.conv1(x)        
        x2 = self.conv2(x1)        
        x3 = self.conv3(x2)         
        x4 = self.conv4(x3)       
        x5 = self.conv5(x4)

        return x5

class GUIDE(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.local = Basic2dLocal(input_planes, norm_layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv11 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv12 = nn.Conv2d(input_planes, input_planes * 9, kernel_size=weight_ks, padding=weight_ks // 2)
        self.conv21 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv22 = nn.Conv2d(input_planes, input_planes * input_planes, kernel_size=1, padding=0)
        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic2d(input_planes, input_planes, norm_layer)

    def forward(self, input, weight):
        
        B, Ci, H, W = input.shape
        weight = torch.cat([input, weight], 1)        
        weight11 = self.conv11(weight)        
        weight12 = self.conv12(weight11)        
        weight21 = self.conv21(weight)        
        weight21_ = self.pool(weight21)        
        weight22_ = self.conv22(weight21_)
        weight22 = weight22_.view(B, -1, Ci)

        out1_ = self.local(input, weight12)
        out1 = out1_.view(B, Ci, -1)             
        out2_ = torch.bmm(weight22, out1)
        out2 = out2_.view(B, Ci, H, W)        
        out3 = self.br(out2)        
        out4 = self.conv3(out3)

        return out4

class GFL(nn.Module):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=3, return_ks=False):
        super().__init__()
        self.return_ks = return_ks
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.local = Basic2dLocal(input_planes, norm_layer)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv11 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv12 = nn.Conv2d(input_planes, input_planes * 9, kernel_size=weight_ks, padding=weight_ks // 2)
        self.conv21 = Basic2d(input_planes + weight_planes, input_planes, None)
        self.conv22 = nn.Conv2d(input_planes, input_planes * input_planes, kernel_size=1, padding=0)
        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic2d(input_planes, input_planes, norm_layer)
        self.conv00 = Basic2d(input_planes + weight_planes, 2, None)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, weight):
        
        B, Ci, H, W = input.shape
        input_weight = torch.cat([input, weight], 1) 
        mask = self.conv00(input_weight)  
        mask = self.softmax(mask)

        input = input * torch.unsqueeze(mask[:, 0, :, :], 1)
        weight = weight * torch.unsqueeze(mask[:, 1, :, :], 1)
        input_weight2 = torch.cat([input, weight], 1) 

        weight11 = self.conv11(input_weight2)        
        weight12 = self.conv12(weight11)        
        weight21 = self.conv21(input_weight2)        
        weight21_ = self.pool(weight21)     # 压缩成一个标量   
        weight22_ = self.conv22(weight21_)
        weight22 = weight22_.view(B, -1, Ci)

        out1_ = self.local(input, weight12)
        out1 = out1_.view(B, Ci, -1)             
        out2_ = torch.bmm(weight22, out1)
        out2 = out2_.view(B, Ci, H, W)        
        out3 = self.br(out2)        
        out4 = self.conv3(out3)

        if self.return_ks:
            return out4, weight12
        
        return out4

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

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


class DKPFGate(nn.Module):
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

        # 通道门控 (Gc)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(rgb_dim, rgb_dim // 4, kernel_size=1, bias=False),  # 降维
            nn.ReLU(),
            nn.Conv2d(rgb_dim // 4, rgb_dim, kernel_size=1, bias=False),  # 恢复通道维度
            nn.Sigmoid()
        )

        # 空间门控 (Gs)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(rgb_dim + depth_dim, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 跨模态残差路径
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
            feat = conv(rgb_feat)  # 普通卷积获取基础特征
            feat = self._dynamic_conv(feat, kernel, k=conv.kernel_size[0])  # 动态核调制
            multi_scale_feats.append(feat)

        # Step 3: 通道-空间联合门控 -----------------------------------------------------
        # 通道门控 (Gc)
        Gc = self.channel_gate(rgb_feat)  # [B, C, 1, 1]

        # 空间门控 (Gs)
        gate_input = torch.cat([rgb_feat, depth_feat], dim=1)  # 通道拼接
        Gs = self.spatial_gate(gate_input)  # [B, 1, H, W]

        # 计算联合门控 G_fused
        G_fused = Gc * Gs  # [B, C, H, W]

        # 进行门控加权
        fused = 0
        for i in range(len(multi_scale_feats)):
            fused += multi_scale_feats[i] * G_fused  # 逐元素加权

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


# attention2d模块
class attention2d(nn.Module):
    def __init__(self, in_planes, K):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，生成全局特征
        self.fc1 = nn.Conv2d(in_planes, K, 1)  # 第一个卷积层，降低维度
        self.fc2 = nn.Conv2d(K, K, 1)  # 第二个卷积层，生成通道注意力

    def forward(self, x):
        x = self.avgpool(x)  # 对输入进行池化，获取全局信息
        x = self.fc1(x)  # 通过卷积降低通道数
        x = Func.relu(x)  # 激活函数
        x = self.fc2(x)  # 输出通道注意力
        return Func.softmax(x, 1)  # Softmax 归一化，输出每个通道的权重


class DKPF_Attention(nn.Module):
    def __init__(self, rgb_dim, depth_dim, norm_layer=None, kernel_sizes=[3, 5, 7], attention_dim=None):
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

        # 动态卷积核选择加权机制
        # 如果没有指定attention_dim，默认为len(self.conv_branches)
        attention_dim = attention_dim or len(self.conv_branches)
        self.attention = attention2d(rgb_dim + depth_dim, attention_dim)  # 使用新的注意力机制
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
            feat = conv(rgb_feat)  # 普通卷积获取基础特征
            feat = self._dynamic_conv(feat, kernel, k=conv.kernel_size[0])  # 动态核调制
            multi_scale_feats.append(feat)

        # Step 3: 自适应多尺度融合 -----------------------------------------------------
        gate_input = torch.cat([rgb_feat, depth_feat], dim=1)
        gates = self.gate_net(gate_input)  # [B, num_scales, H, W]

        # 引入通道注意力机制
        attention_input = torch.cat([rgb_feat, depth_feat], dim=1)
        attention_weights = self.attention(attention_input)  # [B, K, 1, 1]
        attention_weights = attention_weights.view(B, -1, 1, 1).expand(-1, len(self.conv_branches), H,
                                                                       W)  # [B, num_scales, H, W]

        fused = 0
        for i in range(len(multi_scale_feats)):
            fused += multi_scale_feats[i] * gates[:, i:i + 1, :, :] * attention_weights[:, i:i + 1, :, :]

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
