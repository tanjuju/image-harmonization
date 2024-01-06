import math

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable, Function

class ECABlock(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)#加入一个 最大通道注意力

        # self.conv = nn.Conv1d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)#增加的
        ###########自适应卷积核数量############
        k_size=int(abs((math.log(channel,2)+1)/2))
        k_size=k_size if k_size%2 else k_size+1
        #######################
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)#原来的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)

        # y1=self.max_pool(x)#增加一个

        y=y.squeeze(-1).transpose(-1, -2)
        # y1=y1.squeeze(-1).transpose(-1, -2)#增加一个
        # y=torch.cat([y, y1], 1)

        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha #[B,C,1,1]
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
            # [B,1,1,1],公式中的根号C在mean中体现
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / \
                (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)
        # 这里的1+tanh就相当于乘加操作
        return x * gate



def min_max_norm(in_):
    """
        normalization
    :param in_:
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)

class SpatialGate(nn.Module):
    def __init__(self, in_dim=2, mask_mode='mask'):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.mask_mode = mask_mode
        self.spatial = nn.Sequential(*[
            BasicConv(in_dim, in_dim, 3, 1, 1),
            BasicConv(in_dim, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2,  relu=False)
        ])
        self.act = nn.Sigmoid()
    def forward(self, x):
        x_compress = x
        x_out = self.spatial(x_compress)
        attention = self.act(x_out) 
        x = x * attention
        return x

    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, insn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.ins_norm = nn.InstanceNorm2d(out_planes, affine=False, track_running_stats=False) if insn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.ins_norm is not None:
            x = self.ins_norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class DualAttention(nn.Module):
    def __init__(self, gate_channels, mask_mode='mask'):
        super(DualAttention, self).__init__()
        self.ChannelGate = ECABlock(gate_channels)
        # self.ChannelGate = GCT(gate_channels)

        self.SpatialGate = SpatialGate(gate_channels, mask_mode=mask_mode)
        self.mask_mode = mask_mode
    def forward(self, x):
        x_ca = self.ChannelGate(x)#通道注意力
        # x_ca=x_ca*mask+x*(1-mask)           #加入的哦
        ###################
        # x_ca=x
        ################
        x_out = self.SpatialGate(x_ca)#空间注意力

        return x_out + x_ca
        # return x_out