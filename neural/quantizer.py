# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdbscan import HDBSCAN

__all__ = [
    'Quantizer',
]


class Quantizer(nn.Module):
    '''HDBSCAN Quantizer trained by EMA
    
    ...
    
    Parameters:
    -----------
    embedding_dim : int,
    min_cluster_size : int,
    beta : float,
    lmbd : float,
    laplace : float,

    '''

    def __init__(self, embedding_dim=8, min_cluster_size=5, beta=0.99, lmbd=0.1, laplace=1e-5):
        super(Quantizer, self).__init__()

        ### NN Hyperparameters ###

        self.embedding_dim = embedding_dim
        self.min_cluster_size = min_cluster_size
        self.beta = beta
        self.lmbd = lmbd
        self.laplace = laplace

        ##############################

        self.num_embeddings = 1

        ### NN Parameters ###

        self.embeddings = nn.Embedding(self.num_embeddings, embedding_dim)
        self.embeddings.weight.requires_grad_(False)
        self.embeddings.weight.data.normal_()

        self.ema_normalizer = nn.Parameter(torch.zeros(self.num_embeddings, 1), requires_grad=False)
        self.ema_weight = nn.Parameter(torch.randn(self.num_embeddings, embedding_dim), requires_grad=False)

        ##############################

    def forward(self, x):
        # Convert inputs `x` from BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        input_shape = x.shape

        # Flatten input
        x_flatten = x.reshape(-1, self.embedding_dim)

        with torch.no_grad():

            # Use HDBSCAN & EMA to update the embedding vectors
            if self.training:

                samples = torch.cat((
                    x_flatten,
                    self.embeddings.weight.data,
                ))

                # Run HDBSCAN
                hdb = HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    core_dist_n_jobs=-1,
                ).fit(samples.detach().cpu().numpy())

                # Drop noise
                non_noise_labels = hdb.labels_[hdb.labels_ != -1]
                num_clusters = len(set(non_noise_labels))

                if num_clusters > 0:

                    non_noise_labels = torch.from_numpy(non_noise_labels).long().to(x.device).unsqueeze(0)
                    non_noise_samples = samples[hdb.labels_ != -1]

                    # Compute batch embeddings
                    transformer = torch.zeros(num_clusters, non_noise_samples.shape[0], device=x.device)
                    transformer.scatter_(0, non_noise_labels, 1)
                    transformer *= 1 / (transformer.sum(1, keepdim=True) + 1e-9)

                    batch_embeddings = transformer @ non_noise_samples

                    # Calculate distances (Euclidean distance)
                    distances = torch.norm(x_flatten[:, None, :] - batch_embeddings[None, :, :], dim=2)

                    # Encoding
                    encoding_indices = torch.argmin(distances, dim=1, keepdim=True)
                    encodings = torch.zeros(encoding_indices.shape[0], num_clusters, device=x.device)
                    encodings.scatter_(1, encoding_indices, 1)

                    ################################################################################
                    ###               Adding new / Removing embeddings if necessary              ###

                    transformer = torch.zeros(num_clusters, self.num_embeddings, device=x.device)
                    transformer.scatter_(0, non_noise_labels[:, -self.num_embeddings:], 1)
                    transformer *= 1 / (transformer.sum(1, keepdim=True) + 1e-9)

                    self.ema_normalizer.data = transformer @ self.ema_normalizer.data
                    self.ema_weight.data = transformer @ self.ema_weight.data

                    ################################################################################

                    ### Use EMA to update the embedding vectors ###

                    self.ema_normalizer.data = self.beta * self.ema_normalizer.data + (1 - self.beta) * encodings.sum(0, keepdim=True).T
                    # Laplace smoothing of the ema_normalizer
                    n = self.ema_normalizer.data.sum()
                    self.ema_normalizer.data = (self.ema_normalizer.data + self.laplace) / (n + num_clusters * self.laplace) * n

                    self.ema_weight.data = self.beta * self.ema_weight.data + (1 - self.beta) * (encodings.T @ x_flatten)

                    self.embeddings.weight.data = self.ema_weight.data / self.ema_normalizer.data
                    self.num_embeddings = num_clusters

            ################################################################

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
