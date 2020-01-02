# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convolutional import ResDecoder, ResEncoder

__all__ = [
    'VectorQuantizer',
    'VectorQuantizerEMA',
    'VQVAE',
]


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, beta):
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, x):
        # Convert inputs `x` from BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        input_shape = x.shape

        # Flatten input
        x_flatten = x.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = torch.norm(x_flatten[:, None, :] - self.embeddings.weight.data[None, :, :], dim=2)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = encodings @ self.embeddings.weight.data
        quantized_reshaped = quantized.reshape(input_shape).permute(0, 3, 1, 2)
        
        return encodings, quantized_reshaped

    def loss_function(self, x, quantized):
        dictionary_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(x, quantized.detach())
        
        return dictionary_loss + self.beta * commitment_loss


class VectorQuantizerEMA(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, beta, decay):
        super(VectorQuantizerEMA, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.requires_grad_(False)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

        self.clusters_size = nn.Parameter(torch.zeros(num_embeddings), requires_grad=False)
        self.ema_weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim), requires_grad=False)

    def forward(self, x):
        # Convert inputs `x` from BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        input_shape = x.shape

        # Flatten input
        x_flatten = x.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = torch.norm(x_flatten[:, None, :] - self.embeddings.weight.data[None, :, :], dim=2)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = encodings @ self.embeddings.weight.data
        quantized_reshaped = quantized.reshape(input_shape).permute(0, 3, 1, 2)

        # Use EMA to update the embedding vectors
        if self.training:
            self.clusters_size.data = self.decay * self.clusters_size.data + (1 - self.decay) * encodings.sum(0)
            # Laplace smoothing of the cluster size
            n = self.clusters_size.data.sum()
            self.clusters_size.data = (self.clusters_size.data + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            
            self.ema_weight.data = self.decay * self.ema_weight.data + (1 - self.decay) * (encodings.t() @ x_flatten)
            
            self.embeddings.weight.data = self.ema_weight.data / self.clusters_size.unsqueeze(1)

        return encodings, quantized_reshaped

    def loss_function(self, x, quantized):
        commitment_loss = F.mse_loss(x, quantized.detach())
        return self.beta * commitment_loss


class VQVAE(nn.Module):

    def __init__(self, in_channels=3, num_hiddens=128, num_res_hiddens=32, num_res_layers=2,
                    num_embeddings=16, embedding_dim=8, beta=0.25, decay=.0, rgb_out=True):

        super(VQVAE, self).__init__()

        self.encoder = ResEncoder(in_channels, num_hiddens, num_res_hiddens, num_res_layers)

        self.pre_qunatization_conv = nn.Conv2d(num_hiddens, embedding_dim, 
                kernel_size=1, stride=1)

        if decay == 0:
            self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, beta)
        else:
            self.quantizer = VectorQuantizerEMA(num_embeddings, embedding_dim, beta, decay)

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
