import torch
from torch.nn import Conv3d, ReLU, Sigmoid
import torch.nn as nn

# convolutional layers, max pooling, and full connected layer
# the output of the convolutional layers has to be flattened in order to

class SimpleClassificationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleClassificationModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 32, 3, 1, 1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv3d(32, 64, 3, 1, 1),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2))
        self.fcl = nn.Sequential(
            nn.Linear(2097152, 128),
            nn.Linear(128, 2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fcl(x)
        return x

class SimpleSegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleSegmentationModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, 32, 3, 1, 1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv3d(32, 64, 3, 1, 1),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool3d(kernel_size=2, stride=2))
        self.finalconv = nn.Conv3d(64, 2, 1)
        self.upsample = nn.Upsample(mode="trilinear", scale_factor=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.finalconv(x)
        x = self.upsample(x)
        return x

# model = SimpleClassificationModel(2,2)
model = SimpleSegmentationModel(2,2)
input = torch.randn(1, 2, 128, 128, 128)
output = model(input)
output.shape
