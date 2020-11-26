import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, GANType, numClasses=10):
        super(Discriminator, self).__init__()
        self.GANType = GANType

        if self.GANType == "CGAN":
            self.conv1 = nn.Conv2d(1, 32, 4, 2, 1, bias=False)
            self.conv1_c = nn.Conv2d(numClasses, 32, 4, 2, 1, bias=False)
            self.conv2 = nn.Conv2d(32 * 2, 64, 4, 2, 1, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, 32, 4, 2, 1, bias=False)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
            
        self.conv3 = nn.Conv2d(64, 1, 7, 1, 0, bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        if self.GANType == "ACGAN":
            self.fc1 = nn.Linear(3136, 128)
            self.fc2 = nn.Linear(128, 10)

    def forward(self, x, c=None):
        w, h = x.size()[-2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        if self.GANType == "CGAN":
            c = self.conv1_c(c)
            x = torch.cat([x, c], dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        out = x

        out = self.conv3(out)
        out = torch.sigmoid(out)
        out = out.view(-1, 1)

        if self.GANType == "ACGAN":
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)

            x = self.fc2(x)

            return out, x
        
        return out