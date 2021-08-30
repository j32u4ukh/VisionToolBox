import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PtClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2d: input channels, output channels, filter shape
        # conv1.weight.shape = [6, 3, 5, 5]), conv1.bias.shape = [6]
        self.conv1 = nn.Conv2d(3, 6, (5, 5), padding=(2, 2))
        # layer1.shape = [n_batch, 6, 24, 24]
        self.layer1 = None
        self.pool = nn.MaxPool2d(2, 2)
        self.layer2 = None
        self.conv2 = nn.Conv2d(6, 16, (3, 3), padding=(1, 1))
        self.layer3 = None
        self.layer4 = None
        self.layer5 = None
        # self.fd1 = nn.Linear(16 * 5 * 5, 120)
        self.layer6 = None
        self.fd2 = nn.Linear(120, 84)
        self.layer7 = None
        self.fd3 = nn.Linear(84, 10)
        self.layer8 = None

    def forward(self, x):
        self.layer1 = F.relu(self.conv1(x))
        self.layer2 = self.pool(self.layer1)
        self.layer3 = F.relu(self.conv2(self.layer2))
        self.layer4 = self.pool(self.layer3)
        l5 = 1
        for dim in self.layer4.shape[1:]:
            l5 *= dim
        self.layer5 = self.layer4.view(-1, l5)
        self.fd1 = nn.Linear(self.layer5.shape[-1], 60)
        self.layer6_1 = F.relu(self.fd1(self.layer5))
        self.layer6_2 = F.relu(self.fd1(self.layer5))
        self.layer6 = torch.cat((self.layer6_1, self.layer6_2), dim=1)
        self.layer7 = F.relu(self.fd2(self.layer6))
        self.layer8 = self.fd3(self.layer7)
        return self.layer8


model = PtClassifier()
input_data = np.random.random((10, 28, 28, 3))
input_data = np.transpose(input_data, (0, 3, 1, 2))
pt_input = torch.from_numpy(input_data).float()
pt_output = model(pt_input)

