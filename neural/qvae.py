
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convolutional import ResDecoder, ResEncoder
from .quantizer import Quantizer

__all__ = [
    'QVAE',
]

class QVAE(nn.Module):
    '''Quantized Variational AutoEncoder
    
    ...
    
    Parameters:
    -----------
    in_channels : int,
    num_hiddens : int,
    num_res_hiddens : int,
    num_res_layers : int,
    rgb_out : bool,
    embedding_dim : int,
    min_cluster_size : int,
    beta : float,
    lmbd : float,
    laplace : float,

    '''

    def __init__(self, in_channels=3, num_hiddens=128, num_res_hiddens=32, num_res_layers=2, rgb_out=True,
                    embedding_dim=8, min_cluster_size=5, beta=0.99, lmbd=0.1, laplace=1e-5):

        super(QVAE, self).__init__()

        self.encoder = ResEncoder(in_channels, num_hiddens, num_res_hiddens, num_res_layers)

        self.pre_qunatization_conv = nn.Conv2d(num_hiddens, embedding_dim, 
                kernel_size=1, stride=1)

        self.quantizer = Quantizer(embedding_dim, min_cluster_size, beta, lmbd, laplace)

        self.decoder = ResDecoder(embedding_dim, num_hiddens, num_res_hiddens, num_res_layers, rgb_out=rgb_out)

    def forward(self, x):
        encoded = self.pre_qunatization_conv(
            self.encoder(x)
        )

        _, quantized = self.quantizer(encoded)
        quantized_skip_grad = encoded + (quantized - encoded).detach()

        recon_x = self.decoder(quantized_skip_grad)

        return encoded, quantized, recon_x

    def loss_function(self, x, recon_x, encoded, quantized):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
        q_loss = self.quantizer.loss_function(encoded, quantized)

        return recon_loss + q_loss
