# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.cluster import DBSCAN

from .convolutional import ResDecoder, ResEncoder

__all__ = [
    'Quantizer',
    'QVAE',
]


class Quantizer(nn.Module):

    def __init__(self, embedding_dim=8, beta=0.99, lmbd=0.01, eps=0.5, min_samples=10, laplace_coeff=1e-5):
        super(Quantizer, self).__init__()

        self.embedding_dim = embedding_dim
        self.beta = beta
        self.lmbd = lmbd
        self.eps = eps
        self.min_samples = min_samples
        self.laplace_coeff = laplace_coeff

        #####################

        self.num_embeddings = 1

        ### NN Parameters ###

        self.embeddings = nn.Embedding(self.num_embeddings, embedding_dim)
        self.embeddings.weight.requires_grad_(False)
        self.embeddings.weight.data.normal_()

        self.ema_normalizer = nn.Parameter(torch.zeros(self.num_embeddings, 1), requires_grad=False)
        self.ema_weight = nn.Parameter(torch.randn(self.num_embeddings, embedding_dim), requires_grad=False)

        #####################

    def forward(self, x):
        # Convert inputs `x` from BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        input_shape = x.shape

        # Flatten input
        x_flatten = x.reshape(-1, self.embedding_dim)

        with torch.no_grad():

             # Use DBSCAN & EMA to update the embedding vectors
            if self.train:

                samples = torch.cat((
                    x_flatten,
                    self.embeddings.weight.data,
                ))

                sample_weight = [1] * x_flatten.shape[0] + [self.min_samples] * self.num_embeddings

                # Run DBSCAN
                db = DBSCAN(eps=self.eps, min_samples=self.min_samples,
                            n_jobs=-1).fit(samples.detach().numpy(), sample_weight=sample_weight)

                # Ignoring noise
                non_noise_labels = db.labels_[db.labels_ != -1]
                num_clusters = len(set(non_noise_labels))
                label2count = dict(zip(*np.unique(non_noise_labels, return_counts=True)))

                # Compute batch embeddings
                batch_embeddings = torch.zeros(num_clusters, self.embedding_dim, device=x.device)
                for i in db.core_sample_indices_:
                    batch_embeddings[db.labels_[i]] += samples[i] * (1/label2count[db.labels_[i]])

                # Calculate distances (Euclidean distance)
                distances = torch.norm(x_flatten[:, None, :] - batch_embeddings[None, :, :], dim=2)

                # Encoding
                encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
                encodings = torch.zeros(encoding_indices.shape[0], num_clusters, device=x.device)
                encodings.scatter_(1, encoding_indices, 1)

                ################################################################################
                ###               Adding new / Removing embeddings if necessary              ###

                labels = torch.from_numpy(
                    db.labels_[x_flatten.shape[0]:]
                ).long().to(x.device).unsqueeze(0)

                transformer = torch.zeros(num_clusters, self.num_embeddings, device=x.device)
                transformer.scatter_(0, labels, 1)
                transformer *= 1 / (transformer.sum(1, keepdim=True) + 1e-9)

                self.ema_normalizer.data = transformer @ self.ema_normalizer.data
                self.ema_weight.data = transformer @ self.ema_weight.data

                ################################################################################

                ### Use EMA to update the embedding vectors ###

                self.ema_normalizer.data = self.beta * self.ema_normalizer.data + (1 - self.beta) * encodings.sum(0, keepdim=True).T
                # Laplace smoothing of the ema_normalizer
                n = self.ema_normalizer.data.sum()
                self.ema_normalizer.data = (self.ema_normalizer.data + self.laplace_coeff) / (n + num_clusters * self.laplace_coeff) * n

                self.ema_weight.data = self.beta * self.ema_weight.data + (1 - self.beta) * (encodings.T @ x_flatten)

                self.embeddings.weight.data = self.ema_weight.data / self.ema_normalizer.data
                self.num_embeddings = num_clusters


            # Calculate distances (Euclidean distance)
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
        return self.lmbd * F.mse_loss(x, quantized.detach())


class QVAE(nn.Module):

    def __init__(self, in_channels=3, num_hiddens=128, num_res_hiddens=32, num_res_layers=2, rgb_out=True,
                    embedding_dim=8, beta=0.99, lmbd=0.01, eps=0.5, min_samples=10, laplace_coeff=1e-5):

        super(QVAE, self).__init__()

        self.encoder = ResEncoder(in_channels, num_hiddens, num_res_hiddens, num_res_layers)

        self.pre_qunatization_conv = nn.Conv2d(num_hiddens, embedding_dim, 
                kernel_size=1, stride=1)

        self.quantizer = Quantizer(embedding_dim, beta, lmbd, eps, min_samples, laplace_coeff)

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
