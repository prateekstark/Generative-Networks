import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=300)
        self.fc12 = nn.Linear(in_features=300, out_features=z_dim)
        self.fc11 = nn.Linear(in_features=300, out_features=z_dim)
        
    def forward(self, x):
        hidden_variable = F.relu(self.fc1(x))
        return self.fc11(hidden_variable), self.fc12(hidden_variable)
    
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=z_dim, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=784)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
    
    
class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
    
    def forward(self, x):
        mu, logvariance = self.encoder(x)
        sigma = torch.exp(0.5 * logvariance)
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma
        return self.decoder(z), mu, logvariance
        
  