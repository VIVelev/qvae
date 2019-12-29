# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'VAE',
]


class VAE(nn.Module):
    ''' beta-Variational AutoEncoder

    Implementation of the ideas presented here:

    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        - https://arxiv.org/abs/1312.6114

    β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR, 2017
        - https://openreview.net/forum?id=Sy2fzU9gl

    Parameters:
    -----------
    latent_dim : int, dimension of the hidden representation vector
    beta : int, KLD weighting coefficent (see β-VAE)
    
    '''
    
    def __init__(self, latent_dim=20, beta=5):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, sigma):
        return mu + sigma * torch.randn_like(sigma)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        reconstruction = self.decode(z)

        return mu, sigma, z, reconstruction

    def loss_function(self, x, recon_x, mu, sigma):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2)

        return BCE + self.beta * KLD
