import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=300)
        self.fc22 = nn.Linear(in_features=300, out_features=z_dim)
        self.fc21 = nn.Linear(in_features=300, out_features=z_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        hidden_variable = F.relu(self.fc2(x))
        mu = self.fc21(hidden_variable)
        sigma = torch.exp(0.5 * self.fc22(hidden_variable))
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma
        return z
 
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=z_dim, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=784)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    
class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=z_dim, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=300)
        self.fc3 = nn.Linear(in_features=300, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(x)


# class VAE(nn.Module):
#     def __init__(self, z_dim):
#         super().__init__()
#         self.encoder = Encoder(z_dim)
#         self.decoder = Decoder(z_dim)
    
#     def forward(self, x):
#         mu, logvariance = self.encoder(x)
#         sigma = torch.exp(0.5 * logvariance)
#         epsilon = torch.randn_like(sigma)
#         z = mu + epsilon * sigma
#         return self.decoder(z), mu, logvariance
        
#   