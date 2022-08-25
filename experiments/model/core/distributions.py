from model.misc.constraint_utils import softplus, invsoftplus

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

PI = torch.from_numpy(np.asarray(np.pi))

class Bernoulli(nn.Module):
    """
    Bernoulli likelihood
    """

    def __init__(self):
        super(Bernoulli, self).__init__()

    @property
    def _probability(self):
        return 'Bernoulli'

    def log_prob(self, X, XrecL, L=1):
        ''' 
        X           - input images [N,T,nc,d,d]
        XrecL       - reconstructions [L,N,T,nc,d,d]
        '''
        XL = X.repeat([L,1,1,1,1,1]) # L,N,T,nc,d,d 
        return torch.log(1e-3+XrecL)*XL + torch.log(1e-3+1-XrecL)*(1-XL) # L,N,T,nc,d,d
        #lhood = lhood_L.sum([2,3,4,5]).mean(0) # N


class Multivariate_Standard(nn.Module):
    """
    Multivariate Standard Gaussian Distribution
    """

    def __init__(self, L, device):
        super(Multivariate_Standard, self).__init__()
        # params weights
        self.means = torch.zeros(L).to(device)
        self.covariance = torch.eye(L).to(device)

    @property
    def _probability(self):
        return 'Standard Gaussian'
    
    def get_params(self):
        return self.means, self.covariance

    def log_prob(self, x_m):
        '''
        log standrad normal 
        '''
        d = x_m.shape[-1]
        return torch.log(1 / torch.sqrt((2 * torch.pi)**d* torch.det(self.covariance)) * \
             torch.exp(-0.5 * torch.sum((x_m-self.means)*torch.mm((x_m-self.means),self.covariance), dim=1)))


