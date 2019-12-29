# pylint: disable=no-member
# pylint: disable=not-callable

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'GMEmbedding',
    'MixqVAE',
]


class GMEmbedding(nn.Module):

    def __init__(self, embedding_dim, beta=0.33, lmbd=0.25, tolerance=1e-8, em_iter=3, num_embeddings=8):
        super(GMEmbedding, self).__init__()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_dim = embedding_dim
        self.beta = beta
        self.lmbd = lmbd
        self.tolerance = tolerance
        self.em_iter = 3
        self.num_embeddings = num_embeddings

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

        self.priors = []
        self.gaussians_params = []
        self._init_random_gaussians()

        self.scaled_probabilites = []
        self._max_probs = []

    def _init_random_gaussians(self):
        '''Initialize gaussians randomly'''

        self.priors = (1 / self.num_embeddings) * torch.ones(self.num_embeddings, requires_grad=False, device=self._device)
        for _ in range(self.num_embeddings):
            params = {}
            params['mean'] = torch.randn(self.embedding_dim, requires_grad=False, device=self._device)
            params['cov'] = torch.eye(self.embedding_dim, requires_grad=False, device=self._device)
            self.gaussians_params.append(params)

    def likelihood(self, x, params):
        mean = params['mean']
        covar = params['cov']

        likelihood = torch.zeros(x.shape[0], requires_grad=False, device=self._device)

        for i, sample in enumerate(x):
            coeff = 1.0 / torch.sqrt(((2.0 * math.pi) ** self.embedding_dim) * torch.det(covar))
            exponent = torch.exp(-0.5 * (sample - mean).T @ torch.pinverse(covar) @ (sample - mean))
            likelihood[i] = coeff * exponent

        return likelihood

    def _get_likelihoods(self, x):
        '''Calculate the likelihood over all samples and gaussians'''

        likelihoods = torch.zeros(x.shape[0], self.num_embeddings, requires_grad=False, device=self._device)
        for i in range(self.num_embeddings):
            likelihoods[:, i] = self.likelihood(x, self.gaussians_params[i])

        return likelihoods
    
    def _expectation(self, x):
        '''Calculate the probabilites'''

        with torch.no_grad():
            # Calculate probabilities of x belonging to the different gaussians
            probabilities = self._get_likelihoods(x) * self.priors
            sum_probabilities = probabilities.sum(1, keepdim=True)

            self.scaled_probabilites = probabilities / sum_probabilities
            # Save value for convergence check
            self._max_probs.append(self.scaled_probabilites.max(1).values)

    def _maximization(self, x):
        '''Update the parameters and priors'''

        with torch.no_grad():
            # Iterate through clusters and recalculate mean and covariance
            for i in range(self.num_embeddings):
                weights = self.scaled_probabilites[:, i].unsqueeze(1)
                mean = (weights * x).sum(0) / weights.sum()
                covariance = ((x - mean).T @ ((x - mean) * weights)) / weights.sum()
                self.gaussians_params[i]['mean'] = self.beta * self.gaussians_params[i]['mean'] + (1 - self.beta) * mean
                self.gaussians_params[i]['cov'] = self.beta * self.gaussians_params[i]['cov'] + (1 - self.beta) * covariance

            # Update weights
            self.priors = self.beta * self.priors + (1 - self.beta) * self.scaled_probabilites.sum(0) / x.shape[0]

    def _converged(self):
        '''Covergence if || max_prob - last_max_prob || < tolerance'''

        if len(self._max_probs) < 2:
            return False

        diff = torch.norm(self._max_probs[-1] - self._max_probs[-2])
        return diff <= self.tolerance

    def forward(self, x):
        # Convert inputs `x` from BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        input_shape = x.shape

        # Flatten input
        x_flatten = x.reshape(-1, self.embedding_dim)

        # Run EM until convergence or for max iterations
        for _ in range(self.em_iter):
            self._expectation(x_flatten)    # E-step
            self._maximization(x_flatten)   # M-step

            # Check convergence
            if self._converged():
                break
        
        # Final assignment
        self._expectation(x_flatten)

        # Update Embeddings
        for i in range(self.num_embeddings):
            self.embeddings.weight.data[i, :] = self.gaussians_params[i]['mean']

        # Encoding
        encoding_indices = self.scaled_probabilites.argmax(1, keepdim=True)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, requires_grad=False, device=self._device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = encodings @ self.embeddings.weight
        quantized_reshaped = quantized.reshape(input_shape).permute(0, 3, 1, 2)
        
        return encodings, quantized_reshaped

    def loss_function(self, x, quantized):
        return self.lmbd * F.mse_loss(x, quantized.detach())
    

class MixqVAE(nn.Module):

    def __init__(self, embedding_dim, beta=0.9):
        super(MixqVAE, self).__init__()

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

        self.gm_embedding = GMEmbedding(embedding_dim, beta=beta)

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

        _, quantized = self.gm_embedding(encoded)
        quantized_skip_grad = encoded + (quantized - encoded).detach()

        recon_x = self.decoder(
            self.post_vq_conv(quantized_skip_grad)
        )

        return encoded, quantized, recon_x

    def loss_function(self, x, recon_x, encoded, quantized):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        embedding_loss = self.gm_embedding.loss_function(encoded, quantized)

        return recon_loss + embedding_loss
