import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=z_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=784)
        self.BN1 = nn.BatchNorm1d(256)
        self.BN2 = nn.BatchNorm1d(512)
        self.BN3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = self.BN1(x)
        x = nn.LeakyReLU(0.2)(self.fc2(x))
        x = self.BN2(x)
        x = nn.LeakyReLU(0.2)(self.fc3(x))
        x = self.BN3(x)
        x = self.fc4(x)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = nn.LeakyReLU(0.2)(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
