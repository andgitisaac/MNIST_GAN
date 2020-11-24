import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, AuxClassifier, zDim, numClasses=10):
        super(Generator, self).__init__()
        if AuxClassifier:
            zDim += numClasses
        self.conv1 = nn.ConvTranspose2d(zDim, 64, 7, 1, 0, bias=False)
        self.conv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = torch.tanh(x)

        return x


