from model.misc.constraint_utils import softplus, invsoftplus

import numpy as np
import torch
from torch import nn
from torch.nn import init

from torch.distributions import Normal

prior_weights = Normal(0.0, 1.0)
jitter = 1e-5

def sample_normal(shape, seed=None):
    '''
    Draw samples from a normal (Gaussian) distribution 
    '''
    rng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    return torch.tensor(rng.normal(size=shape).astype(np.float32))

def sample_uniform(shape, seed=None):
    # random Uniform sample of a given shape
    if seed is not None:
        rng = np.random.RandomState(seed)
        return torch.tensor(rng.uniform(low=0.0, high=1.0, size=shape).astype(np.float32))
    else:
        return torch.tensor(np.random.uniform(low=0.0, high=1.0, size=shape).astype(np.float32))


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
        @return: Tensor (D_out,N,M) if dimwise else (N,M)
        """
        if self.dimwise:
            sq_dist = torch.exp(- 0.5 * self.square_dist_dimwise(X, X2))  # (D_out,N,M)
            return self.variance[:, None, None] * sq_dist  # (D_out,N,M)
        else:
            sq_dist = torch.exp(-0.5 * self.square_dist(X, X2))  # (N,M)
            return self.variance * sq_dist  # (N,M)

    def sample_freq(self, S, seed=None, device='cpu'):
        """
        Computes random samples from the spectral density for Squared exponential kernel: omega ~ N(0, A),
        where A is diagonal matrix collecting lengthscale parameters of the kernel. 
        @param S: Number of features
        @param seed: random seed
        @return: Tensor a random sample from standard Normal (D_in, S, D_out) if dimwise else (D_in, S)
        """
        omega_shape = (self.D_in, S, self.D_out) if self.dimwise else (self.D_in, S)
        omega = sample_normal(omega_shape, seed).to(device)  # (D_in, S, D_out) or (D_in, S)
        lengthscales = self.lengthscales.T.unsqueeze(1) if self.dimwise else self.lengthscales.unsqueeze(
            1)  # (D_in,1,D_out) or (D_in,1)
        return omega / lengthscales  # (D_in, S, D_out) or (D_in, S)

    def build_cache(self, S, device):
        """
        Generate and fix parameters of Fourier features
        @param S: Number of features to consider for Fourier feature maps
        Set omega and b of phi(x)=cos(omega*x + b) of rff
        Set w of w*phi(x) of prior update
        """
        # generate parameters required for the Fourier feature maps
        self.rff_weights = sample_normal((S, self.D_out)).to(device)  # (S,D_out)
        self.rff_omega = self.sample_freq(S, device=device)  # (D_in,S) or (D_in,S,D_out)
        phase_shape = (1, S, self.D_out) if self.dimwise else (1, S)
        self.rff_phase = sample_uniform(phase_shape).to(device) * 2 * np.pi  # (S,D_out)


    def rff_forward(self, x, S):
        """
        Calculates samples from the GP prior with random Fourier Features
        @param x: input tensor (N,D)
        @return: function values (N,D_out)
        """
        # compute feature map
        xo = torch.einsum('nd,dfk->nfk' if self.dimwise else 'nd,df->nf', x, self.rff_omega)  # (N,S) or (N,S,D_in)
        phi_ = torch.cos(xo + self.rff_phase)  # (N,S) or (N,S,D_in)
        phi = phi_ * torch.sqrt(self.variance / S)  # (N,S) or (N,S,D_in)

        # compute function values
        f = torch.einsum('nfk,fk->nk' if self.dimwise else 'nf,fd->nd', phi, self.rff_weights)  # (N,D_out)
        return f  # (N,D_out)

    def compute_nu(self,Ku, u_prior,inducing_val):
        '''
        @param Lu: lower triangular
        @param u_prior: phi(x)
        @param induving_val: u
        compute the term nu = k(Z,Z)^{-1}(u-f(Z)) in whitened form of inducing variables
        equation (13) from http://proceedings.mlr.press/v119/wilson20a/wilson20a.pdf
        '''
        Lu = torch.linalg.cholesky(Ku + torch.eye(Ku.shape[-1]).to(Ku.device) * jitter)  # (M,M) or (D,M,M)
        if not self.dimwise:
            nu = torch.triangular_solve(u_prior, Lu, upper=False)[0]  # (M,D)
            nu = torch.triangular_solve((inducing_val - nu),
                                        Lu.T, upper=True)[0]  # (M,D)
        else:
            nu = torch.triangular_solve(u_prior.T.unsqueeze(2), Lu, upper=False)[0]  # (D,M,1)
            nu = torch.triangular_solve((inducing_val.T.unsqueeze(2) - nu),
                                        Lu.permute(0, 2, 1), upper=True)[0]  # (D,M,1)
        self.nu = nu  # (D,M)

    def f_update(self, x, x2):
        if not self.dimwise:
            Kuf = self.K(x2, x)  # (M,N)
            f_update = torch.einsum('md, mn -> nd', self.nu, Kuf)  # (N,D)
        else:
            Kuf = self.K(x2, x)  # (D,M,N)
            f_update = torch.einsum('dm, dmn -> nd', self.nu.squeeze(2), Kuf)  # (N,D)
        return f_update

class Periodic(torch.nn.Module):
    pass 
#combination pass first through period then through rbf (rbf +periodic)

class DivergenceFreeKernel(RBF):
    def __init__(self, D_in, D_out):
        super(DivergenceFreeKernel, self).__init__(D_in=D_in, D_out=D_out,dimwise=False)

    def difference_matrix(self, X, X2=None):
        '''
        Computes (X-X2)
        '''
        X = (X / self.lengthscales).unsqueeze(-1)  # (N,D_in,1)
        if X2 is None:
            X2=X
        else:
            X2 = (X2 / self.lengthscales).unsqueeze(-1)# (M,D_in,1)
        X2 = torch.permute(X2, (2,1,0)) # (1, D_in, M)
        return torch.subtract(X,X2)  #N,D_in,M
    
    def eye_like(self, X, d:int, X2=None) -> torch.Tensor:
        """
        Return a tensor with same batch size as x, that has a nxn eye matrix in each sample in batch.

        Args:
            @param X: Input 1 (N,D_in)
            @param X2:  Input 2 (M,D_in)
            @param n: Input dimensions
        Returns:
            tensor of shape (N, M, n, n) that has the same dtype and device as x.
        """
        N = X.shape[0]
        M = X2.shape[0] if X2 != None else N
        return torch.eye(d, d,device=X.device).unsqueeze(0).repeat(N,M, 1, 1)

    def reshape(self, K, X, X2=None):
        N = X.shape[0]
        M = N if X2==None else X2.shape[0]
        return torch.reshape(K,(N*self.D_in,M*self.D_in))


    def K(self, X, X2=None):
        """
        Computes K(X, X_2)
        @param X: Input 1 (N,D_in)
        @param X2:  Input 2 (M,D_in)
        @return: Tensor (N,M,D_in,D_in)
        """
        sq_dist = self.square_dist(X, X2)  # (N,M)
        rbf_term = self.variance * torch.exp(-0.5 * sq_dist)[:,:,None,None]  # (N,M, 1,1)
        diff = self.difference_matrix(X,X2) #(N,D_in,M)
        
        diff1 = torch.permute(diff.unsqueeze(-1), (0,2,1,3)) # (N, M, D_in, 1)
        diff2 = torch.permute(diff.unsqueeze(-1), (0,2,3,1)) # (N, M, 1, D_in)
        term1 = torch.multiply(diff1, diff2) #N,M, D_in, D_in


        term2 = torch.multiply(((self.D_in - 1.0) - sq_dist[:,:,None,None]), self.eye_like(X,self.D_in,X2)) #N,M,D_in, D_in
        hes_term  = term1 + term2 

        K = rbf_term * hes_term / torch.square(self.lengthscales)
        K = self.reshape(torch.permute(K, (0,2,1,3)), X, X2) 
       
        return K #(N*D_in, M*D_in) 

    def rff_forward(self, x, S):
        """
        Calculates samples from the GP prior with operator random Fourier Features
        @param x: input tensor (N,D)
        @return: function values (N,D_out)
        according to http://proceedings.mlr.press/v63/Brault39.pdf derivation of ORRF for Div-free
        """
        #compute B^*(omega)
        #norm = torch.linalg.vector_norm(self.rff_omega, ord=2, dim=0, keepdim=True)
        #print('norm',norm.shape)
        norm = torch.norm(self.rff_omega)
        w_w = torch.matmul(self.rff_omega,self.rff_omega.T)/norm
        #print('mamt', w_w)
        # n = torch.norm(self.rff_omega)
        #norm = self.rff_omega.pow(2).sum(dim=0)
        #I_matrix = torch.eye(self.D_in, device=x.device).unsqueeze(0).repeat(S,1,1)
        #print('I_matrix', I_matrix.shape)
        #b_omega = torch.sqrt(norm*torch.eye(self.D_in, device=x.device) - w_w) #(D_in, D_in)
        b_omega = norm*torch.eye(self.D_in, device=x.device) - w_w #(D_in, D_in)
       # b_omega = norm*I_matrix - matm
        #print('b_omega', b_omega)
        # compute feature map
        xo = torch.einsum('nd,df->nf', x, self.rff_omega)  # (N,S) 
        #print('xo', xo[0,0:10])
        phi_ = torch.cos(xo + self.rff_phase)  # (N,S)
        phi_ = torch.kron(phi_,b_omega) # (N*D_in , S*D_in)
        phi = phi_ * torch.sqrt(self.variance / S)  # (N*D_in , S*D_in) 
        phi = torch.reshape(phi, (x.shape[0],S,self.D_in,self.D_in)) #(N,S,D_in,D_in)
       # print('phi', phi[0,0,:,:])

        # compute function values
        f = torch.einsum('nfij,fd->ndij', phi, self.rff_weights)  # ( N, D_out, D_in, D_in )
        f = torch.permute(f, (0,2,1,3)) # (N,D_in, D_out,D_in)
        f = torch.reshape(f,(x.shape[0]*self.D_in, self.D_out*self.D_in))
        return f  # ND_in, D_outD_in 

    def compute_nu(self,Ku, u_prior,inducing_val):
        '''
        @param Lu: lower triangular
        @param u_prior: phi(x)
        @param induving_val: u
        compute the term nu = k(Z,Z)^{-1}(u-f(Z)) in whitened form of inducing variables
        equation (13) from http://proceedings.mlr.press/v119/wilson20a/wilson20a.pdf
        '''
        Lu = torch.linalg.cholesky(Ku + torch.eye(Ku.shape[0]).to(Ku.device) * jitter)  #MD_in,MD_in
        nu = torch.triangular_solve(u_prior, Lu, upper=False)[0]  # (MD_in, D_outD_in)
        nu = torch.reshape(nu,(inducing_val.shape[0],self.D_in,self.D_out,self.D_in))
        delta = torch.reshape(inducing_val[:,None,:,None] - nu,(inducing_val.shape[0]*self.D_in,self.D_out*self.D_in)) #MD_in,D_outD_in
        nu = torch.triangular_solve(delta,Lu.T, upper=True)[0]  # (D_in, D_in, M,D_out)
        self.nu = nu  # (MD_in, D_outD_in)


    def f_update(self, x, x2):
        Kuf = self.K(x2, x)  #(MD_in, ND_in) 
        #print('Kuf', Kuf[0,0:10])
        f_update = torch.einsum('md, mn -> nd', self.nu, Kuf)  # NDin,D_outDin
        return f_update 

