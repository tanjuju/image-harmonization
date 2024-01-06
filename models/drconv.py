import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable, Function
from models.dfconv import DeformConv2d

def xcorr_slow(x, kernel, kwargs):
    """for loop to calculate cross correlation
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk,  **kwargs)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel, kwargs):
    """group conv2d to calculate cross correlation
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk,  **kwargs, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po

class Corr(Function):
    @staticmethod
    def symbolic(g, x, kernel, groups):
        return g.op("Corr", x, kernel, groups_i=groups)

    @staticmethod
    def forward(self, x, kernel, groups, kwargs):
        """group conv2d to calculate cross correlation
        """
        batch = x.size(0)
        channel = x.size(1)
        x = x.view(1, -1, x.size(2), x.size(3))
        kernel = kernel.view(-1, channel // groups, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, **kwargs, groups=groups * batch)
        out = out.view(batch, -1, out.size(2), out.size(3))
        return out

class Correlation(nn.Module):
    use_slow = True

    def __init__(self, use_slow=None):
        super(Correlation, self).__init__()
        if use_slow is not None:
            self.use_slow = use_slow
        else:
            self.use_slow = Correlation.use_slow

    def extra_repr(self):
        if self.use_slow: return "xcorr_slow"
        return "xcorr_fast"

    def forward(self, x, kernel, **kwargs):
        if self.training:
            if self.use_slow:
                return xcorr_slow(x, kernel, kwargs)
            else:
                return xcorr_fast(x, kernel, kwargs)
        else:
            return Corr.apply(x, kernel, 1, kwargs)

class DRConv2d(nn.Module):
    #in_channels=out_channels kernel_size=1 region_num=2
    def __init__(self, in_channels, out_channels, kernel_size, region_num=2, **kwargs):
        super(DRConv2d, self).__init__()
        self.region_num = 2
        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1, groups=region_num)
        )
        self.conv_guide = nn.Conv2d(in_channels, region_num, kernel_size=kernel_size, **kwargs)
        
        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.act = nn.Sigmoid()
    def forward(self, input, mask):
        kernel = self.conv_kernel(input)# B region_num * in_channels * out_channels w h
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3)) # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs) # B x (r*out) x W x H
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)) # B x r x out x W x H

        mask = F.interpolate(mask.detach(), size=input.size()[2:], mode='nearest')
        mask = mask.unsqueeze(1) 
        inv_msak = 1 - mask
        guide_mask = torch.cat((mask, inv_msak), 1)

        output = torch.sum(output * guide_mask, dim=1)#将前景和背景分离再合并
        return output


class myDRC(nn.Module):
    # in_channels=out_channels kernel_size=1 region_num=2
    #区域动态分离
    def __init__(self, in_channels, out_channels, kernel_size, region_num=2, **kwargs):
        super(myDRC, self).__init__()
        self.region_num = 2
        self.conv_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
            nn.Conv2d(in_channels, region_num * region_num, kernel_size=1),
            nn.Sigmoid(),
            nn.Conv2d(region_num * region_num, region_num * in_channels * out_channels, kernel_size=1,
                      groups=region_num)
        )

        self.corr = Correlation(use_slow=False)
        self.kwargs = kwargs
        self.def_conv1 = nn.Sequential(nn.ReLU(True), DeformConv2d(out_channels, out_channels, 3, 1, 1))
        self.def_conv2 = nn.Sequential(nn.ReLU(True), DeformConv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, input, mask):
        kernel = self.conv_kernel(input)  # B region_num * in_channels * out_channels w h
        kernel = kernel.view(kernel.size(0), -1, kernel.size(2), kernel.size(3))  # B x (r*in*out) x W X H
        output = self.corr(input, kernel, **self.kwargs)  # B x (r*out) x W x H
        output = output.view(output.size(0), self.region_num, -1, output.size(2), output.size(3)).contiguous()  # B x r x out x W x H

        mask = F.interpolate(mask.detach(), size=input.size()[2:], mode='nearest')
        output1,output2=torch.chunk(output,2,dim=1)
        print(output1.size())
        output1=self.def_conv1(output1.squeeze(1))
        output2=self.def_conv2(output2.squeeze(1))

        # mask = mask.unsqueeze(1)#在第一维度加维度
        inv_msak = 1 - mask
        # guide_mask = torch.cat((mask, inv_msak), 1)

        # output = torch.sum(output * guide_mask, dim=1)  # 将前景和背景分离再合并
        output=output1*mask+output2*inv_msak
        return output