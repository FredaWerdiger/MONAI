import torch
from torch.nn import Conv3d, ReLU, Sigmoid
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 32, 3, 1, 1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv3d(32, 64, 3, 1, 1),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU(inplace=True))
        self.linear_layer = nn.Sequential(
            nn.Linear(64, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.linear_layer(x)
        return x

