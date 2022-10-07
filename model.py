import torch
import torch.nn as nn

from .layers import ResidualLayer


class ResNet(nn.Module):
    """
    this is used to create resnet layer
    """

    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.res_layer1 = ResidualLayer(input_channels=32, output_channels=32)
        self.res_layer2 = ResidualLayer(input_channels=32, output_channels=32)
        self.res_layer3 = ResidualLayer(
            input_channels=32, output_channels=64, downsampling=True
        )
        self.res_layer4 = ResidualLayer(input_channels=64, output_channels=64)
        self.res_layer5 = ResidualLayer(
            input_channels=64, output_channels=128, downsampling=True
        )
        self.res_layer6 = ResidualLayer(
            input_channels=128,
            output_channels=128,
        )
        self.avg_pool = nn.AvgPool2d(8)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)
        x = self.res_layer5(x)
        x = self.res_layer6(x)
        x = self.avg_pool(x)
        x = x.view(-1, 128)
        x = self.dropout(x)
        x = self.fc(x)
        return x
