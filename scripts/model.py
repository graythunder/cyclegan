import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from layer import *


class Generator(nn.Module):
    def __init__(self, image_size=256):
        super(Generator, self).__init__()

        if image_size >= 256:
            n_res = 9
        else:
            n_res = 6

        layers = []
        layers.append(InitialLayer(3, 64))
        layers.append(DownsampleLayer(64, 128))
        layers.append(DownsampleLayer(128, 256))
        for i in range(n_res):
            layers.append(ResidualBlock(256))
        layers.append(UpsampleLayer(256, 128))
        layers.append(UpsampleLayer(128, 64))
        layers.append(InitialLayer(64, 3))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        y = self.model(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, input_size=256):
        super(Discriminator, self).__init__()

        outdim = input_size // 2**3 - 4
        self.out_shape = (1, outdim, outdim)

        self.model = nn.Sequential(
            DiscriminatorLayer(3, 64, norm=False),
            DiscriminatorLayer(64, 128),
            DiscriminatorLayer(128, 256),
            DiscriminatorLayer(256, 512, stride=1),
            nn.Conv2d(512, 1, 4)
        )

    def forward(self, x):
        y = self.model(x)
        return y
