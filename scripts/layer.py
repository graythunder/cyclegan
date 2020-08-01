import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

class InitialLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialLayer, self).__init__()
        self.refpad = nn.ReflectionPad2d(3)
        self.conv = nn.Conv2d(in_channels, out_channels, 7)
        self.norm = nn.InstanceNorm2d(out_channels)       
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.refpad(x)
        h = self.conv(h)
        h = self.norm(h)
        y = self.relu(h) 
        return y

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleLayer, self).__init__()
        self.refpad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.refpad(x)
        h = self.conv(h)
        h = self.norm(h)
        y = self.relu(h) 
        return y

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
    
        self.refpad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3)
        self.norm1 = nn.InstanceNorm2d(n_channels)       
        self.relu1 = nn.ReLU()
        self.refpad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3)
        self.norm2 = nn.InstanceNorm2d(n_channels)       
        self.relu2 = nn.ReLU()

    def forward(self, x):
        h = self.refpad1(x)
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.relu1(h)
        h = self.refpad2(x)
        h = self.conv2(h)
        h = self.norm2(h)
        y = self.relu2(x + h)
        return y

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleLayer, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.convt(x)
        h = self.norm(h)
        y = self.relu(h) 
        return y

class DiscriminatorLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, norm=True):
        super(DiscriminatorLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        h = self.conv(x)
        h = self.norm(h)
        y = self.lrelu(h)
        return y