import torch
import torch.nn as nn

from mmcv.ops import ModulatedDeformConv2d, DeformConv2d
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, activation=nn.LeakyReLU(0.1, inplace=True),
                 norm_layer=nn.InstanceNorm2d):
        super(ResBlock, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
                                   activation,
                                   nn.Conv2d(out_channels, in_channels, kernel_size, 1, 1))

    def forward(self, x):
        x = x + self.model(x)

        return x


class hallucination_module(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, norm_layer=nn.InstanceNorm2d):
        super(hallucination_module, self).__init__()

        self.dilation = dilation

        if self.dilation != 0:

            self.hallucination_conv = DeformConv(out_channels, out_channels, modulation=True, dilation=self.dilation)

        else:

            self.m_conv = nn.Conv2d(in_channels, 3 * 3, kernel_size=3, stride=1, bias=True)

            self.m_conv.weight.data.zero_()
            self.m_conv.bias.data.zero_()

            self.dconv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):

        if self.dilation != 0:

            hallucination_output, hallucination_map = self.hallucination_conv(x)

        else:
            hallucination_map = 0

            mask = torch.sigmoid(self.m_conv(x))

            offset = torch.zeros_like(mask.repeat(1, 2, 1, 1))

            hallucination_output = self.dconv(x, offset, mask)

        return hallucination_output, hallucination_map


class hallucination_res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=(0, 1, 2, 4), norm_layer=nn.InstanceNorm2d):
        super(hallucination_res_block, self).__init__()

        self.dilations = dilations

        self.res_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
                                      nn.LeakyReLU(0.1, inplace=True))

        self.hallucination_d0 = hallucination_module(in_channels, out_channels, dilations[0])
        self.hallucination_d1 = hallucination_module(in_channels, out_channels, dilations[1])
        self.hallucination_d2 = hallucination_module(in_channels, out_channels, dilations[2])
        self.hallucination_d3 = hallucination_module(in_channels, out_channels, dilations[3])

        self.mask_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       ResBlock(out_channels, out_channels,
                                                norm_layer=norm_layer),
                                       ResBlock(out_channels, out_channels,
                                                norm_layer=norm_layer),
                                       nn.Conv2d(out_channels, 4, 1, 1))

        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        res = self.res_conv(x)

        d0_out, _ = self.hallucination_d0(res)
        d1_out, map1 = self.hallucination_d1(res)
        d2_out, map2 = self.hallucination_d2(res)
        d3_out, map3 = self.hallucination_d3(res)

        mask = self.mask_conv(x)
        mask = torch.softmax(mask, 1)

        sum_out = d0_out * mask[:, 0:1, :, :] + d1_out * mask[:, 1:2, :, :] + \
                  d2_out * mask[:, 2:3, :, :] + d3_out * mask[:, 3:4, :, :]

        res = self.fusion_conv(sum_out) + x

        map = torch.cat([map1, map2, map3], 1)

        return res, map


class DeformConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=True, modulation=True, dilation=1):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(1)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, stride=stride, dilation=dilation,
                                padding=dilation, bias=bias)

        self.p_conv.weight.data.zero_()
        if bias:
            self.p_conv.bias.data.zero_()

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, stride=stride, dilation=dilation,
                                    padding=dilation, bias=bias)

            self.m_conv.weight.data.zero_()
            if bias:
                self.m_conv.bias.data.zero_()

            self.dconv = ModulatedDeformConv2d(inc, outc, kernel_size, padding=padding)
        else:
            self.dconv = DeformConv2d(inc, outc, kernel_size, padding=padding)

    def forward(self, x):
        offset = self.p_conv(x)

        if self.modulation:
            mask = torch.sigmoid(self.m_conv(x))
            x_offset_conv = self.dconv(x, offset, mask)
        else:
            x_offset_conv = self.dconv(x, offset)

        return x_offset_conv, offset


class Dynamic_conv(nn.Module):
    def __init__(self, kernel_size):
        super(Dynamic_conv, self).__init__()

        self.reflect_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size)

    def forward(self, x, kernel):
        b, c, h, w = x.size()
        x = self.reflect_pad(x)

        kernel = F.softmax(kernel, dim=1)

        unfolded_x = self.unfold(x)
        unfolded_x = unfolded_x.view(b, c, -1, h, w)

        out = torch.einsum('bkhw,bckhw->bchw', [kernel, unfolded_x])

        return out
