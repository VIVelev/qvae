import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'AutoEncoder',
]


class AutoEncoder(nn.Module):
    '''Convolutional AutoEncoder
    '''

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent_vector = self.encoder(x)
        reconstruction = self.decoder(latent_vector)

        return latent_vector, reconstruction

    def loss_function(self, x, recon_x):
        return F.binary_cross_entropy(recon_x, x)
