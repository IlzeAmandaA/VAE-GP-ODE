from model.misc.constraint_utils import softplus, invsoftplus

import numpy as np
import torch
from torch import nn
from torch.nn import init

from torch.distributions import Normal
import gpflow

prior_weights = Normal(0.0, 1.0)


def sample_normal(shape, seed=None):
    rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    return torch.tensor(rng.normal(size=shape).astype(np.float32))


class RBF(torch.nn.Module):
    """
    Implements squared exponential kernel with kernel computation and weights and frequency sampling for Fourier features
    """

    def __init__(self, D_in, D_out=None, dimwise=False):
        """
        @param D_in: Number of input dimensions
        @param D_out: Number of output dimensions
        @param dimwise: If True, different kernel parameters are given to output dimensions
        """
        super(RBF, self).__init__()
        self.D_in = D_in
        self.D_out = D_in if D_out is None else D_out
        self.dimwise = dimwise
        lengthscales_shape = (self.D_out, self.D_in) if dimwise else (self.D_in,)
        variance_shape = (self.D_out,) if dimwise else (1,)
        self.unconstrained_lengthscales = nn.Parameter(torch.ones(size=lengthscales_shape),
                                                       requires_grad=True)
        self.unconstrained_variance = nn.Parameter(torch.ones(size=variance_shape),
                                                   requires_grad=True)
        self._initialize()

    def _initialize(self):
        init.constant_(self.unconstrained_lengthscales, invsoftplus(torch.tensor(1.3)).item())
        init.constant_(self.unconstrained_variance, invsoftplus(torch.tensor(0.5)).item())

    @property
    def lengthscales(self):
        return softplus(self.unconstrained_lengthscales)

    @property
    def variance(self):
        return softplus(self.unconstrained_variance)

    def square_dist_dimwise(self, X, X2=None):
        """
        Computes squared euclidean distance (scaled) for dimwise kernel setting
        @param X: Input 1 (N,D_in)
        @param X2: Input 2 (M,D_in)
        @return: Tensor (D_out, N,M)
        """
        X = X.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,N,D_in)
        Xs = torch.sum(torch.pow(X, 2), dim=2)  # (D_out,N)
        if X2 is None:
            return -2 * torch.einsum('dnk, dmk -> dnm', X, X) + \
                   Xs.unsqueeze(-1) + Xs.unsqueeze(1)  # (D_out,N,N)
        else:
            X2 = X2.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,M,D_in)
            X2s = torch.sum(torch.pow(X2, 2), dim=2)  # (D_out,N)
            return -2 * torch.einsum('dnk, dmk -> dnm', X, X2) + Xs.unsqueeze(-1) + X2s.unsqueeze(1)  # (D_out,N,M)

    def square_dist(self, X, X2=None):
        """
        Computes squared euclidean distance (scaled) for non dimwise kernel setting
        @param X: Input 1 (N,D_in)
        @param X2: Input 2 (M,D_in)
        @return: Tensor (N,M)
        """
        X = X / self.lengthscales  # (N,D_in)
        Xs = torch.sum(torch.pow(X, 2), dim=1)  # (N,)
        if X2 is None:
            return -2 * torch.matmul(X, X.t()) + \
                   torch.reshape(Xs, (-1, 1)) + torch.reshape(Xs, (1, -1))  # (N,1)
        else:
            X2 = X2 / self.lengthscales  # (M,D_in)
            X2s = torch.sum(torch.pow(X2, 2), dim=1)  # (M,)
            return -2 * torch.matmul(X, X2.t()) + torch.reshape(Xs, (-1, 1)) + torch.reshape(X2s, (1, -1))  # (N,M)

    def K(self, X, X2=None):
        """
        Computes K(X, X_2)
        @param X: Input 1 (N,D_in)
        @param X2:  Input 2 (M,D_in)
        @return: Tensor (D,N,M) if dimwise else (N,M)
        """
        if self.dimwise:
            sq_dist = torch.exp(- 0.5 * self.square_dist_dimwise(X, X2))  # (D_out,N,M)
            return self.variance[:, None, None] * sq_dist  # (D_out,N,M)
        else:
            sq_dist = torch.exp(-0.5 * self.square_dist(X, X2))  # (N,M)
            return self.variance * sq_dist  # (N,M)

    def sample_freq(self, S, seed=None, device='cpu'):
        """
        Computes random samples from the spectral density for Squared exponential kernel
        @param S: Number of features
        @param seed: random seed
        @return: Tensor a random sample from standard Normal (D_in, S, D_out) if dimwise else (D_in, S)
        """
        omega_shape = (self.D_in, S, self.D_out) if self.dimwise else (self.D_in, S)
        omega = sample_normal(omega_shape, seed).to(device)  # (D_in, S, D_out) or (D_in, S)
        lengthscales = self.lengthscales.T.unsqueeze(1) if self.dimwise else self.lengthscales.unsqueeze(
            1)  # (D_in,1,D_out) or (D_in,1)
        return omega / lengthscales  # (D_in, S, D_out) or (D_in, S)

class Periodic(torch.nn.Module):
    pass 
#combination pass first through period then through rbf (rbf +periodic)

class DivergenceFreeKernel(RBF):
    def __init__(self, D_in, D_out, dimwise=False):
        super(DivergenceFreeKernel, self).__init__(D_in=D_in, D_out=D_out,dimwise=dimwise)

    @property
    def num_latent_gps(self):
        pass

    @property
    def latent_kernels(self):
        pass

    def difference_matrix(self, X, X2=None):
        '''
        Computes (X-X2^T)
        '''
        X = X / self.lengthscales  # (N,D_in)
        if X2 is None:
            X2=X
            diff = X[..., :, None,:]-X2[...,None,:,:]  # (N,N,D_in)
            return diff
        X2 = X2 / self.lengthscales  # (M,D_in)
        Xshape = X.shape
        X2shape = X2.shape
        X = torch.reshape(X, (-1, Xshape[-1]))
        X2 = torch.reshape(X2, (-1, X2shape[-1]))
        diff = X[:,None, :] - X2[None, :, :]
        diff = torch.reshape(diff, (list(Xshape[:-1]) + list(X2shape[:-1]) + list([Xshape[-1]]))) #N, M, D_in
        return diff

    def difference_matrix_dimwise(self, X, X2=None):
        '''
        Computes (X-X2^T)
        '''
        X = X.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,N,D_in)
        if X2 is None:
            X2=X
            diff = X[..., :, None,:]-X2[...,None,:,:]  # (D_out,M,D_in)
            return diff
        X2 = X2.unsqueeze(0) / self.lengthscales.unsqueeze(1)  # (D_out,M,D_in)
        Xshape = X.shape
        X2shape = X2.shape
        X = torch.reshape(X, (-1, Xshape[-1]))
        X2 = torch.reshape(X2, (-1, X2shape[-1]))
        diff = X[:,None, :] - X2[None, :, :]
        diff = torch.reshape(diff, (list(Xshape[:-1]) + list(X2shape[:-1]) + list([Xshape[-1]]))) #D_out, N, D_out, M, D_in
        return diff
    
    def identity(self, X, X2=None):
        if X2 is None:
            return torch.eye(self.D_out).unsqueeze(0).repeat(X.shape[0], X.shape[0], 1, 1)
        else:
            return torch.eye(self.D_out).unsqueeze(0).repeat(X.shape[0], X2.shape[0], 1, 1)

    def K(self, X, X2=None, full_output_cov=True):
        """
        Computes K(X, X_2)
        @param X: Input 1 (N,D_in)
        @param X2:  Input 2 (M,D_in)
        @return: Tensor (D,N,M) if dimwise else (N,M)
        """
        if self.dimwise:
            sq_dist = self.square_dist_dimwise(X, X2)  # (D_out,N,M)
            diff = self.difference_matrix_dimwise(X, X2)  #D_out, N, D_out, M, D_in
            diff1 = torch.unsqueeze(diff, -1) 
            diff2 = torch.permute(torch.unsqueeze(diff, -1), (0,1,2,3,5,4)) 
            K1_term = diff1 @ diff2 #D_out, N, D_out, M, D_in, D_in
        else:
            sq_dist = self.square_dist(X, X2)  # (N,M)
            diff = self.difference_matrix(X, X2) #N, M, D_in
            diff1 = torch.unsqueeze(diff, -1) #N,M,D_in,1
            diff2 = torch.permute(torch.unsqueeze(diff, -1), (0,1,3,2)) #N,M,1,D_in
            K1_term = diff1 @ diff2 # N,M,D_in,D_in


        # dist = square_distance(self.scale(X), self.scale(X2))
        K2 = torch.exp(-0.5 * sq_dist)
        # K2 = tf.exp(0.5 * -dist)
        # K2 = tf.expand_dims(tf.expand_dims(K2, -1), -1)


        K3 = self.D_out - 1.0 - sq_dist #D_out, N,M if dimwise else N,M
        K3 = K3.unsqueeze(-1).unsqueeze(-1)
        K3 = K3 * self.identity(X,X2)

        K = (K1_term + K3) * K2
        K = (self.variance / torch.pow(self.lengthscale,2)) * K

        if full_output_cov:
            return tf.transpose(K, [0, 2, 1, 3])
        else:
            K = tf.linalg.diag_part(K)
            return tf.transpose(K, [2, 0, 1])

    def K_diag(self, X, full_output_cov=True):
        if full_output_cov:
            return self.K(X, full_output_cov=full_output_cov)
        else:
            k = self.K(X, full_output_cov=full_output_cov)
            return tf.transpose(tf.linalg.diag_part(k))
