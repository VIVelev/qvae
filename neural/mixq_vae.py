# pylint: disable=no-member
# pylint: disable=not-callable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convolutional import ResEncoder, ResDecoder

__all__ = [
    'GMEmbedding',
    'MixqVAE',
]


class GMEmbedding(nn.Module):
    '''Gaussian Mixture Model Embeddings
    
    GMM PyTorch optimization taken from:
    https://angusturner.github.io/generative_models/2017/11/03/pytorch-gaussian-mixture-model.html

    Legend:
    -----------
    mu : mean
    logvar : log scale variance
    posterior : pi
    
    Parameters:
    -----------
     : 
    
    '''

    log_norm_constant = -0.5 * math.log(2 * math.pi)

    def __init__(self, embedding_dim=4, beta=0.9, lmbd=0.25, num_iter=3, num_embeddings=8):
        super(GMEmbedding, self).__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_dim = embedding_dim
        self.beta = beta
        self.lmbd = lmbd
        self.num_iter = num_iter
        self.num_embeddings = num_embeddings

        # Gaussian Embeddings
        self.embeddings_mu = torch.empty(num_embeddings, embedding_dim,
            device=self._device).uniform_(-1/num_embeddings, 1/num_embeddings)

        self.embeddings_logvar = torch.empty(num_embeddings, embedding_dim,
            device=self._device).uniform_(-1/num_embeddings, 1/num_embeddings)

        self.embeddings_pi = torch.empty(num_embeddings, num_embeddings,
            device=self._device).uniform_(-1/num_embeddings, 1/num_embeddings)

        # Gaussian Batch-Parameters
        self._batch_mu = []
        self._batch_logvar = []
        self._batch_pi = []

        self._init_random_batch_gaussians()

    def _init_random_batch_gaussians(self):
        '''Initialize gaussians randomly'''

        self._batch_mu = torch.randn(self.num_embeddings, self.embedding_dim, device=self._device)
        self._batch_logvar = torch.ones(self.num_embeddings, self.embedding_dim, device=self._device)
        self._batch_pi = torch.empty(self.num_embeddings, self.num_embeddings, device=self._device).fill_(1. / self.num_embeddings)

    def log_gaussian(self, x, mu, logvar):
        logexp = -0.5 * (logvar + ((x - mu) ** 2) / logvar.exp())
        return logexp + GMEmbedding.log_norm_constant

    def _get_likelihoods(self, x, log=True):
        '''Calculate the likelihood over all samples and gaussians'''

        # get feature-wise log-likelihoods (num_embeddings, examples, features)
        log_likelihoods = self.log_gaussian(
            x[None, :, :], # (1, examples, features)
            self._batch_mu[:, None, :], # (num_embeddings, 1, features)
            self._batch_logvar[:, None, :] # (num_embeddings, 1, features)
        )

        # sum over the feature dimension
        log_likelihoods = log_likelihoods.sum(-1)

        if not log:
            return log_likelihoods.exp()

        return log_likelihoods

    def _logsumexp(self, x, dim, keepdim=False):
        max_values = x.max(dim=dim, keepdim=True).values
        return max_values + (x - max_values).exp().sum(dim=dim, keepdim=keepdim).log()

    def _get_posteriors(self, log_likelihoods, log=True):
        log_posteriors = log_likelihoods - self._logsumexp(log_likelihoods, dim=0, keepdim=True)

        if not log:
            return log_posteriors.exp()

        return log_posteriors

    def _update_batch_parameters(self, x, eps=1e-6, min_var=1e-6):
        posteriors = self._get_posteriors(self._get_likelihoods(x), log=False)

        # compute `N_k` the proxy "number of points" assigned to each distribution.
        N_k = torch.sum(posteriors, dim=1)
        N_k = N_k.view(self.num_embeddings, 1, 1)

        # get the means by taking the weighted combination of points
        # (K, 1, examples) @ (1, examples, features) -> (K, 1, features)
        mu = posteriors[:, None] @ x[None, ]
        mu = mu / (N_k + eps)
        self._batch_mu = mu.squeeze(1)

        # compute the diagonal covar. matrix, by taking a weighted combination of
        # the each point's square distance from the mean
        A = x - mu
        var = posteriors[:, None] @ (A ** 2) # (K, 1, features)
        var = var / (N_k + eps)
        logvar = var.clamp(min=min_var).log()
        self._batch_logvar = logvar.squeeze(1)

        # recompute the priors
        self._batch_pi = (N_k / N_k.sum()).squeeze()

    def forward(self, x):
        # Convert inputs `x` from BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        input_shape = x.shape

        # Flatten input
        x_flatten = x.reshape(-1, self.embedding_dim)

        with torch.no_grad():
            # Run EM for `num_iter` iterations
            for _ in range(self.num_iter):
                self._update_batch_parameters(x_flatten)

            # Update Embeddings
            self.embeddings_mu = self.beta * self.embeddings_mu + (1 - self.beta) * self._batch_mu
            self.embeddings_logvar = self.beta * self.embeddings_logvar + (1 - self.beta) * self._batch_logvar
            self.embeddings_pi = self.beta * self.embeddings_pi + (1 - self.beta) * self._batch_pi

            # Encoding
            encoding_indices = self._get_likelihoods(x_flatten, log=False).T.argmax(1, keepdim=True)
            encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=self._device)
            encodings.scatter_(1, encoding_indices, 1)
            
            # Quantize
            quantized = encodings @ self.embeddings_mu
            quantized_reshaped = quantized.reshape(input_shape).permute(0, 3, 1, 2)
        
        return encodings, quantized_reshaped

    def loss_function(self, x, quantized):
        return self.lmbd * F.mse_loss(x, quantized.detach())
    

class MixqVAE(nn.Module):

    def __init__(self, in_channels=3, num_hiddens=128, num_res_hiddens=32, num_res_layers=2,
                    embedding_dim=4, beta=0.9, lmbd=0.25, num_iter=3, num_embeddings=8, rgb=True):

        super(MixqVAE, self).__init__()

        self.encoder = ResEncoder(in_channels, num_hiddens, num_res_hiddens, num_res_layers)

        self.pre_conv = nn.Conv2d(num_hiddens, embedding_dim, 
                kernel_size=1, stride=1)

        self.gm_embedding = GMEmbedding(embedding_dim, beta, lmbd, num_iter, num_embeddings)

        self.decoder = ResDecoder(embedding_dim, num_hiddens, num_res_hiddens, num_res_layers, rgb=rgb)

    def forward(self, x):
        encoded = self.pre_conv(
            self.encoder(x)
        )

        _, quantized = self.gm_embedding(encoded)
        quantized_skip_grad = encoded + (quantized - encoded).detach()

        recon_x = self.decoder(quantized_skip_grad)

        return encoded, quantized, recon_x

    def loss_function(self, x, recon_x, encoded, quantized):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        embedding_loss = self.gm_embedding.loss_function(encoded, quantized)

        return recon_loss + embedding_loss
