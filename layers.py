import torch
import torch.nn as nn


class ResidualLayer(nn.Module):
    """
    this is use for creating a residual layer
    """

    def __init__(self, input_channels, output_channels, downsampling=False):
        super(ResidualLayer, self).__init__()
        self.downsampling = downsampling
        stride = 1
        if downsampling:
            stride = 2
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample_layer = nn.Conv2d(
            input_channels, output_channels, kernel_size=1, stride=2, bias=False
        )
        self.bn3 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        residual = x.clone()
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv1(x))
        if self.downsampling:
            residual = self.bn3(self.downsample_layer(residual))
        result = self.relu(out + residual)
        return result
