import torch
from torch import nn
from torch.nn import functional as F


class MNISTVAE(nn.Module):
    def __init__(self, z_dim: int, x_dim=784, h_dim1=512, h_dim2=256):
        super(MNISTVAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var


class VAELatentClassifier(nn.Module):

    def __init__(self, z_dim: int, num_classes: int = 10) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            # nn.Linear(z_dim, z_dim),
            # nn.ReLU(),
            nn.Linear(z_dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
