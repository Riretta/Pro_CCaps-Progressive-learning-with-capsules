from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from U_CapsNets.fastai_unet import DynamicUnet
import torch
import torch.nn.functional as F

class ResNet_BB(torch.nn.Module):
    def __init__(self,n_input=1):
        super(ResNet_BB, self).__init__()
        body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
        # print(body)
        self.net_G = DynamicUnet(body, 313, (224, 224),16,(32,5,5),128)
        self.softmax = torch.nn.Softmax(dim=1)
        self.model_out = torch.nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)

    def forward(self, x):
        q = self.net_G(x)
        ab = self.model_out(self.softmax(q))
        return ab, self.softmax(q)

# from torchsummary import summary
# net = ResNet_BB()
# b = net(torch.randn(1,1,224,224))
