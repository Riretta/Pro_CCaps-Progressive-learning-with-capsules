import torch.nn.functional as F
from Zhang_github_N.base_color import *
import math
import numpy as np

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1, padding=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

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
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2)
            self.conv = DoubleConv(in_channels//2, out_channels,  padding=1)

    def forward(self, x1, x2):
        if self.bilinear: x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        if not self.bilinear: x = self.up(x)
        return self.conv(x)

class Residual(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.ConvBlock = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, padding=0),
                                        nn.BatchNorm2d(out_channel))
        self.Relu = nn.ReLU(inplace=True)

    def forward(self,x,y):
        if not x.size()[1] == self.in_channel:
            print("Residual function: dimension error x")
            exit()
        if not y.size()[1] == self.out_channel:
            print("Residual function: dimension error y")
            exit()
        if not self.in_channel == self.out_channel: x = self.ConvBlock(x)
        addiction = torch.sum(torch.stack([x,y]),dim=0)
        return self.Relu(addiction)

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=16,num_routes=(32,9,9)):
        super(PrimaryCaps, self).__init__()
        self.routes_dim = np.prod(num_routes)
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=3, stride=2, padding=2)
            for _ in range(num_capsules)])  # 16 num capsules

    def forward(self, x):
        u = [capsule(x).view(x.shape[0], self.routes_dim,1) for capsule in self.capsules]
        u = torch.cat(u, dim=-1)  # => batch size, num_capsules, num feat map per capsule, H feature map, W feature map
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)  # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor

import U_CapsNets.helpers as helpers # to get transpose softmax function

# dynamic routing
def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    '''Performs dynamic routing between two capsule layers.
       param b_ij: initial log probabilities that capsule i should be coupled to capsule j
       param u_hat: input, weighted capsule vectors, W u
       param squash: given, normalizing squash function
       param routing_iterations: number of times to update coupling coefficients
       return: v_j, output capsule vectors
       '''
    # update b_ij, c_ij for number of routing iterations
    for iteration in range(routing_iterations):
        # softmax calculation of coupling coefficients, c_ij
        c_ij = helpers.softmax(b_ij, dim=2)

        # calculating total capsule inputs, s_j = sum(c_ij*u_hat)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)

        # squashing to get a normalized vector output, v_j
        v_j = squash(s_j)

        # if not on the last iteration, calculate agreement and new b_ij
        if iteration < routing_iterations - 1:
            # agreement
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)

            # new b_ij
            b_ij = b_ij + a_ij
        else:
            u_j = (c_ij * u_hat)

    return v_j, u_j  # return latest v_j

class DigitCaps(nn.Module):
    def __init__(self, logits_num=32, num_routes=(32, 6, 6), num_capsules=16):
        super(DigitCaps, self).__init__()

        self.in_channels = logits_num
        self.num_routes = np.prod(num_routes)
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, self.num_routes,
                                          num_capsules,logits_num))
    def forward(self, x):
        '''Defines the feedforward behavior.
                   param u: the input; vectors from the previous PrimaryCaps layer
                   return: a set of normalized, capsule output vectors
                   '''

        # adding batch_size dims and stacking all u vectors
        u = x[None, :, :, None, :]
        # 4D weight matrix
        W = self.W[:, None, :, :, :]

        # calculating u_hat = W*u
        u_hat = torch.matmul(u, W)

        # getting the correct size of b_ij
        # setting them all to 0, initially
        b_ij = torch.zeros(*u_hat.size())

        # moving b_ij to GPU, if available
        b_ij = b_ij.to(x.device)

        # update coupling coefficients and calculate v_j
        v_j, u_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        return v_j, u_j  # return final vector outputs

    def squash(self, input_tensor):
        '''Squashes an input Tensor so it has a magnitude between 0-1.
           param input_tensor: a stack of capsule inputs, s_j
           return: a stack of normalized, capsule output vectors, v_j
           '''
        # same squash function as before
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)  # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor

class TPrimaryCaps(nn.Module):
    def __init__(self,num_routes,num_capsules,logits_num):
        super(TPrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.W = nn.Parameter(torch.randn(1, np.prod(num_routes), num_capsules,logits_num))
        self.reconstruction_capsules = nn.ModuleList([nn.ConvTranspose2d(in_channels=num_routes[0],
                                                                         out_channels=int(512 / num_capsules),
                                                                         kernel_size=3, stride=2, padding=2) for _ in
                                                      range(num_capsules)])
    def forward(self, a):
        (y,x) = a
        x = x.squeeze()
        if len(x.size())<3: x = x.unsqueeze(0)
        batch_size = x.size(0)

        W = torch.cat([self.W] * batch_size, dim=0)
        x = x[:, :, :, None]
        uhat = torch.matmul(W, x)
        uhat = uhat.permute(0, 2, 1, 3)
        uhat = uhat.view(uhat.size(0), uhat.size(1), *self.num_routes)

        # Recombine capsules into a feature map matrix
        # A reconstrution capsule sees as input the output of a previous capsule...
        u_rec = [capsule(uhat[:, ii, :, :, :]) for ii, capsule in enumerate(self.reconstruction_capsules)]
        u_rec = torch.cat(u_rec, dim=1)

        return u_rec

class Reconstruction(BaseColor):
    def __init__(self, logits_num=32, num_capsules=16, num_routes=(32, 9, 9), AM=False, dim=None, nmap=None):
        super(Reconstruction, self).__init__()

        if nmap is None:
            nmap = [512, 256, 128, 64]
        if dim is None:
            dim = [(16, 16), (20, 20), (24, 24), (28, 28)]
        self.AM = AM

        self.color_channels = 2
        self.num_routes = num_routes

        self.W = nn.Parameter(torch.randn(1, np.prod(num_routes), num_capsules,logits_num))

        self.reconstruction_capsules = nn.ModuleList([nn.ConvTranspose2d(in_channels=num_routes[0],
                                                                         out_channels=int(512 / num_capsules),
                                                                         kernel_size=3, stride=2, padding=2) for _ in
                                                      range(num_capsules)])
        bilinear = True
        self.reconstruction_layers_up1 = Up(nmap[0] + 512, 512, bilinear=bilinear, upsampling_size=dim[0])
        self.reconstruction_layers_up2 = Up(nmap[1] + 512, 256, bilinear=bilinear, upsampling_size=dim[1])
        self.reconstruction_layers_up3 = Up(nmap[2] + 256, 128, bilinear=bilinear, upsampling_size=dim[2])
        self.reconstruction_layers_up4 = Up(nmap[3] + 128, 64, bilinear=bilinear, upsampling_size=dim[3])
        self.conv_layer_up = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1,dilation=1),
                                           nn.BatchNorm2d(32),
                                           nn.ReLU(inplace=True))
        self.q = nn.Conv2d(nmap[4], 313, kernel_size=1, stride=1, padding=0, bias=False)
        self.residual = Residual(32, nmap[4])
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x, conv1, conv2, conv3, conv4, conv_layer_d):
        batch_size = x.size(0)

        W = torch.cat([self.W] * batch_size, dim=0)
        x = x[:, :, :, None]

        uhat = torch.matmul(W, x)
        uhat = uhat.permute(0, 2, 1, 3)
        uhat = uhat.view(uhat.size(0), uhat.size(1), *self.num_routes)

        # Recombine capsules into a feature map matrix
        # A reconstrution capsule sees as input the output of a previous capsule...
        u_rec = [capsule(uhat[:, ii, :, :, :]) for ii, capsule in enumerate(self.reconstruction_capsules)]
        u_rec = torch.cat(u_rec, dim=1)
        a = 0
        # Go up..
        x = self.reconstruction_layers_up1(u_rec, conv4)
        x = self.reconstruction_layers_up2(x, conv3)
        x = self.reconstruction_layers_up3(x, conv2)
        x = self.reconstruction_layers_up4(x, conv1)
        x = self.conv_layer_up(x)
        x = self.residual(x, conv_layer_d)
        q = self.q(x)
        x = self.model_out(self.softmax(q))

        return self.unnormalize_ab(self.upsample4(x)), self.softmax(q)  # da provare prima era solo q

class CapsNet_MR(nn.Module):
    def __init__(self, logits_num, backbone,  AM=False, num_capsules=16, num_routes=(32, 9, 9)):
        super(CapsNet_MR, self).__init__()
        print('__UCapsNet__tiny__')
        self.conv_layer = backbone
        dim_up, n_map = backbone.MapDimension(torch.randn([1,1,224,224]))
        self.primary_capsules = PrimaryCaps(num_capsules=num_capsules,num_routes=num_routes)
        self.digit_capsules = DigitCaps(logits_num=logits_num, num_routes=num_routes, num_capsules=num_capsules)
        self.reconstruction = Reconstruction(logits_num=logits_num, AM=AM, num_routes=num_routes,
                                             num_capsules=num_capsules, dim=dim_up,nmap= n_map)

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        conv1, conv2, conv3, conv4, conv_layer_d = self.conv_layer(data)
        primary_caps_output = self.primary_capsules(conv4)
        output, u_hat = self.digit_capsules(primary_caps_output)
        reconstructionsAB, reconstructionsQ = self.reconstruction(u_hat.squeeze(), conv1, conv2, conv3, conv4,conv_layer_d)
        return reconstructionsAB, reconstructionsQ

    def caps_loss(self, data, x, target, reconstructions):  # <--------------------------------------ML+REC
        return (self.margin_loss(x, target)/data.size(0) + self.reconstruction_loss(data, reconstructions))

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

        return (loss * 0.0005)/data.size(0)



