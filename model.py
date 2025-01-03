import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, conv1_out=8, conv2_out=5, conv3_out=4, kernel_size1=3, kernel_size2=3, kernel_size3=3):
        super(Network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_out, kernel_size=kernel_size1, stride=1),
            nn.PReLU(num_parameters=conv1_out),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(conv1_out, conv2_out, kernel_size=kernel_size2, stride=1),
            nn.PReLU(num_parameters=conv2_out),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(conv2_out, conv3_out, kernel_size=kernel_size3, stride=1),
            nn.PReLU(num_parameters=conv3_out),
        )

        # Dynamically calculate the size of the flattened feature map
        self._initialize_linear_in_size(kernel_size1, kernel_size2, kernel_size3, conv1_out, conv2_out, conv3_out)

        # Define classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.linear_in, 10),
        )

    def _initialize_linear_in_size(self, kernel_size1, kernel_size2, kernel_size3, conv1_out, conv2_out, conv3_out):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)  # MNIST image size
            features_out = self.features(dummy_input)
            self.linear_in = features_out.view(-1).shape[0]  # Flattened size of the feature map

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
