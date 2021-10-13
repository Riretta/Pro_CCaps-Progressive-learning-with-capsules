from Zhang_github_N.training_layers_PL import PriorBoostLayer, NNEncLayer, NonGrayMaskLayer
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
keys = ['primary','1','2','3','4','Q']
layers_dim_224 = {keys[0]:(15,15),keys[1]:(16,16),keys[2]:(20,20),keys[3]:(24,24),keys[4]:(28,28),keys[5]:(56,56)}
layers_dim_256 = {keys[0]:(19,19),keys[1]:(20,20),keys[2]:(24,24),keys[3]:(28,28),keys[4]:(32,32),keys[5]:(64,64)}
layers_dim_ResNet = {keys[0]:(7,7),keys[1]:(7,7),keys[2]:(14,14),keys[3]:(28,28),keys[4]:(56,56),keys[5]:(56,56)}
layers_dim_Zhang = {keys[0]:(27,27),keys[1]:(28,28),keys[2]:(28,28),keys[3]:(28,28),keys[4]:(28,28),keys[5]:(56,56)}
layers_dim_UNET = {keys[1]:(44,44),keys[2]:(48,48),keys[3]:(52,52),keys[4]:(56,56),keys[5]:(56,56)}


class Quantization_module(nn.Module):
    def __init__(self):
        super(Quantization_module, self).__init__()

        self.encode_layer = NNEncLayer()
        self.boost_layer = PriorBoostLayer()
        self.nongray_mask = NonGrayMaskLayer()

    def forward(self,x,depth='Q',mode=224, ResNet_mode = False, Zhang_mode = False, UNET_mode = False):
        if not ResNet_mode and not Zhang_mode and not UNET_mode:
            if mode == 224: layers_dim = layers_dim_224
            elif mode == 256: layers_dim = layers_dim_256
        elif ResNet_mode:
            layers_dim = layers_dim_ResNet
        elif Zhang_mode:
            layers_dim = layers_dim_Zhang
        elif UNET_mode:
            layers_dim = layers_dim_UNET

        xx = F.interpolate(x, size=layers_dim[depth])
        encode, max_encode = self.encode_layer.forward(xx)

        targets = Tensor(max_encode).long()
        boost = Tensor(self.boost_layer.forward(encode)).float()
        mask = Tensor(self.nongray_mask.forward(x)).float()
        boost_nongray = boost*mask

        if depth == keys[5]: xx = x

        return targets, boost_nongray, xx

    def depth_identifier(self,epoch):
        if (epoch > 0 or epoch == 0) and (epoch < 15 or epoch == 15):
            return keys[0]
        if epoch > 15 and (epoch < 25 or epoch == 25):
            return keys[1]
        if epoch > 25 and (epoch < 35 or epoch == 35):
            return keys[2]
        if epoch > 35 and (epoch < 45 or epoch == 45):
            return keys[3]
        if epoch > 45 and (epoch < 55 or epoch == 55):
            return keys[4]
        if epoch > 55:
            return keys[5]