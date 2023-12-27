import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import zipfile

"""Define Resnet18 Pytorch Class"""

# Define block


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(Block, self).__init__()

        # controls if we need to downsample during skip connection so that matrices are same size
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        # check if we need to downsample skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

# Define resnet 18 architecture


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(Resnet18, self).__init__()

        self.relu = nn.ReLU()

        # Block 1 input = 224x224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64,
                               kernel_size=7, stride=2, padding=3)  # output = 112x112x64
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)  # output = 56x56x64

        # Block 2 output = 56x56x64
        self.block2 = self.create_layer(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Block 3 output = 28x28x128
        self.block3 = self.create_layer(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        # Block 4 output = 14x14x256
        self.block4 = self.create_layer(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        # Block 5 output = 7x7x512
        self.block5 = self.create_layer(
            in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)

        # average pooling layer
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        # fully connected layer
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def create_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        # check if layer has downsampling
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        layer = nn.Sequential(
            Block(in_channels, out_channels, kernel_size,
                  stride, padding, downsample),
            Block(out_channels, out_channels, kernel_size, stride=1, padding=padding,
                  downsample=None)  # Downsample only in skip connection layer
        )
        return layer

    def forward(self, x, feature_extract=False):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if feature_extract:
            return x

        x = self.fc(x)

        return x
