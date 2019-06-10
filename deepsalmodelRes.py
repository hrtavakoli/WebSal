'''
Shallow Saliency Model

@author: Hamed R. Tavakoli
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.device(device)


class ScaleUp(nn.Module):

    def __init__(self, in_size, out_size):
        super(ScaleUp, self).__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_size),
            nn.LeakyReLU(inplace=True))
        self.conv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
        )

        self.__init_weights__()

    def __init_weights__(self):

        nn.init.kaiming_normal_(self.conv[0].weight)
        nn.init.constant_(self.conv[0].bias, 0.0)
        nn.init.kaiming_normal_(self.conv[3].weight)
        nn.init.constant_(self.conv[3].bias, 0.0)

        nn.init.kaiming_normal_(self.up[0].weight)
        nn.init.constant_(self.up[0].bias, 0.0)

    def forward(self, inputs):
        output = self.up(inputs)
        output = self.conv(output)
        return output


class DeepSal(nn.Module):

    def __init__(self, ):
        super(DeepSal, self).__init__()
        self.encode_image = resnet50(pretrained=True)
        modules = list(self.encode_image.children())[:-2]
        self.encode_image = nn.Sequential(*modules)
        self.decoder1 = ScaleUp(2048, 1024)
        self.decoder2 = ScaleUp(1024, 512)
        self.decoder3 = ScaleUp(512, 256)
        self.saliency = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        self.__init_weights__()

    def __init_weights__(self):

        nn.init.kaiming_normal_(self.saliency.weight)
        nn.init.constant_(self.saliency.bias, 0.0)

    def forward(self, x):
        x1 = self.encode_image(x)
        x1 = self.decoder1(x1)
        x1 = self.decoder2(x1)
        x1 = self.decoder3(x1)
        sal = self.saliency(x1)
        sal = F.relu(sal, inplace=True)
        return sal


if __name__ == "__main__":
    sample_input = torch.ones(1, 3, 256, 320).to(device)
    model = DeepSal().to(device)
    print(model(sample_input).shape)
