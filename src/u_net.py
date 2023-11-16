import torch
import torchvision.transforms.functional
from torch import nn

class DoubleConvulation(nn.Module):
    def __init__ (self, channels_in, channels_out):
        super().__init()

        self.first_convolution = nn.Conv2d(channels_in, channels_out, kernel_size = 3, padding = 1)
        self.first_activation = nn.ReLU()

        self.second_convolution = nn.Conv2d(channels_out, channels_out, kernel_size = 3, padding = 1)
        self.second_activation = nn.ReLU()

    def forward(self, x):

        x_a = self.first_convolution(x)
        x_b = self.first_activation(x_a)
        x_c = self.second_activation(x_b)
        x_d = self.second_activation(x_c)

        return x_d

class MaxPool2x2(nn.Module):
    def __init__(self):
        super().__init()

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)

class UpConv2x2(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        self.up = nn.ConvTranspose2d(channels_in, channels_out, kernel_size = 2, stride = 2)

    def forward(self, x):
        return self.up(x)

class CopyAndCrop(nn.Module):
    def forward(self, x, contracting_x):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim = 1)
        return x

class UNet(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init()

        self.down_convulation = nn.ModuleList([DoubleConvulation(i, j)
                                               for i, j in [(channels_in, 64), (64, 128), (128, 256), (256, 512)]])

        self.max_pool = nn.ModuleList([MaxPool2x2() for _ in range(4)])

        self.middle_convulation = DoubleConvulation(512, 1024)

        self.up_sample = nn.ModuleList([UpConv2x2(i, j)
                                      for i, j in [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.up_convulation = nn.ModuleList([DoubleConvulation(i, j)
                                             for i, j in [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.concat = nn.ModuleList([CopyAndCrop() for _ in range(4)])

        self.last_convulation = nn.Conv2d(64, channels_out, kernel_size = 1)

    def forward(self, x):
        passed = []

        for i in range(len(self.down_convulation)):
            x = self.down_convulation[i](x)
            passed.append(x)
            x = self.max_pool[i](x)

        x = self.middle_convulation(x)

        for i in range(len(self.up_convulation)):

            x = self.up_sample[i](x)
            x = self.concat[i](x, passed.pop())
            x = self.up_convulation[i](x)

        x = self.last_convulation(x)

        return x