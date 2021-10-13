import torch
import torch.nn as nn
import torch.nn.functional as F
from Zhang_github_N.base_color import *
import math
import numpy as np


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, padding=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, padding=0):
        super().__init__()
        self.down_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, padding=padding)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, upsampling_size=None):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(size=upsampling_size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, padding=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if not x2==None:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class ConvLayer(BaseColor):
    def __init__(self):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                   Down(32, 64, padding=1))
        self.conv2 = Down(64, 128, padding=0)
        self.conv3 = Down(128, 256, padding=0)
        self.conv4 = Down(256, 512, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        conv1 = self.conv1(self.normalize_l(x))
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv1, conv2, conv3, conv4

class Reconstruction(BaseColor):
    def __init__(self):
        super(Reconstruction, self).__init__()
        bilinear = True
        self.reconstruction_layers_up1 = Up(512, 512, bilinear=bilinear, upsampling_size=(44, 44))
        self.skipq1 = nn.Conv2d(512,313, kernel_size=1, stride=1, padding=0, bias=False)
        self.skipab1 = nn.Conv2d(313,2, kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction_layers_up2 = Up(256 + 512, 256, bilinear=bilinear, upsampling_size=(48, 48))
        self.skipq2 = nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=False)
        self.skipab2 = nn.Conv2d(313, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction_layers_up3 = Up(128 + 256, 128, bilinear=bilinear, upsampling_size=(52, 52))
        self.skipq3 = nn.Conv2d(128, 313, kernel_size=1, stride=1, padding=0, bias=False)
        self.skipab3 = nn.Conv2d(313, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.reconstruction_layers_up4 = Up(64 + 128, 64, bilinear=bilinear, upsampling_size=(56, 56))
        self.skipq4 = nn.Conv2d(64, 313, kernel_size=1, stride=1, padding=0, bias=False)
        self.skipab4 = nn.Conv2d(313, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.q = nn.Conv2d(64, 313, kernel_size=1, stride=1, padding=0, bias=False)
        # self.ab = nn.Conv2d(64, 2, kernel_size=1,stride=1,padding=0,bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, conv1, conv2, conv3, conv4,depth='Q'):
        # Go up..
        if depth == '1' or depth == '2' or depth == '3' or depth == '4' or depth == 'Q':
            # Go up..
            x = self.reconstruction_layers_up1(conv4)
            if depth == '1':
                q = self.skipq1(x)
                ab = self.skipab1(q)
            if depth == '2' or depth == '3' or depth == '4' or depth == 'Q':
                x = self.reconstruction_layers_up2(x, conv3)
                if depth == '2':
                    q = self.skipq2(x)
                    ab = self.skipab2(q)
                if depth == '3' or depth == '4' or depth == 'Q':
                    x = self.reconstruction_layers_up3(x, conv2)
                    if depth == '3':
                        q = self.skipq3(x)
                        ab = self.skipab3(q)
                    if depth == '4' or depth == 'Q':
                        x = self.reconstruction_layers_up4(x, conv1)
                        if depth == '4':
                            q = self.skipq4(x)
                            ab = self.skipab4(q)
                        if depth == 'Q':
                            # x = self.conv_layer_up(x)  # <----with zhang
                            q = self.softmax(self.q(x))
                            x = self.model_out(q)
                            ab = self.unnormalize_ab(self.upsample4(x))

        return ab, q


class CapsNet_MR(nn.Module):
    def __init__(self):
        super(CapsNet_MR, self).__init__()

        self.conv_layer = ConvLayer()
        self.reconstruction = Reconstruction()

        self.mse_loss = nn.MSELoss()

    def forward(self, data, depth = 'Q'):
        conv1, conv2, conv3, conv4 = self.conv_layer(data)
        reconstructionsAB, reconstructionsQ = self.reconstruction( conv1, conv2, conv3, conv4,depth=depth)
        return reconstructionsAB, reconstructionsQ

    def CE_loss(self, data, preds):
        batch_size = data.size(0)
        loss = -torch.mean(torch.sum(data * torch.log(preds), dim=1))
        return loss

    def loss(self, data, x, target, reconstructions):  # <--------------------------------------ML+REC
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def loss_togheter(self, data, reconstructions):
        loss_AB = self.mse_loss(reconstructions.view(reconstructions.size(0), -1),
                                data.view(reconstructions.size(0), -1))
        return loss_AB * 0.001

    def reconstruction_loss(self, data, reconstructions, plus=False):
        reconstructions_A = reconstructions[:, 0, :, :]
        data_A = data[:, 0, :, :]
        reconstructions_B = reconstructions[:, 1, :, :]
        data_B = data[:, 1, :, :]
        loss_A = self.mse_loss(reconstructions_A.view(reconstructions.size(0), -1),
                               data_A.view(reconstructions.size(0), -1))
        loss_B = self.mse_loss(reconstructions_B.view(reconstructions.size(0), -1),
                               data_B.view(reconstructions.size(0), -1))

        if not plus:
            loss = loss_A + loss_B
        else:
            loss_AB = self.loss_togheter(data, reconstructions)
            loss = loss_AB + loss_A + loss_B

        return loss * 0.001

#
#
# import torch
# a = torch.randn(2,1,224,224)
# m = CapsNet_MR()
# n,m = m(a,'4')
# print(m.size())