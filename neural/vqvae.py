# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'VectorQuantizer',
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
        distances = torch.sqrt(torch.sum(x_flatten**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * x_flatten @ self.embeddings.weight.T)
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = encodings @ self.embeddings.weight
        quantized_reshaped = quantized.reshape(input_shape).permute(0, 3, 1, 2)
        
        return encodings, quantized_reshaped

    def loss_function(self, x, quantized):
        dictionary_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(x, quantized.detach())
        
        return dictionary_loss + self.beta * commitment_loss
    
class VQVAE(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, beta):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.pre_vq_conv = nn.Sequential(
            nn.Conv2d(8, embedding_dim, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.vq = VectorQuantizer(num_embeddings, embedding_dim, beta)

        self.post_vq_conv = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 8, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.pre_vq_conv(
            self.encoder(x)
        )

        _, quantized = self.vq(encoded)
        quantized_skip_grad = encoded + (quantized - encoded).detach()

        recon_x = self.decoder(
            self.post_vq_conv(quantized_skip_grad)
        )

        return encoded, quantized, recon_x

    def loss_function(self, x, recon_x, encoded, quantized):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        vq_loss = self.vq.loss_function(encoded, quantized)

        return recon_loss + vq_loss
