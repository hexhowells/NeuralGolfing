import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(8, 5, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(5, 4, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(36, 10),
            nn.LogSoftmax(dim=1)
        )
            
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x