import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torchdiffeq import odeint

from .utils import *

class Encoder(nn.Module):
    def __init__(self, steps = 1, n_filt=8, q=8):
        super(Encoder, self).__init__()

        h_dim = n_filt*4**3 # encoder output is [4*n_filt,4,4]

        self.cnn = nn.Sequential(
            nn.Conv2d(steps, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.ReLU(),
            Flatten()
        )
        self.fc1 = nn.Linear(h_dim, q)
        self.fc2 = nn.Linear(h_dim, q)

    def forward(self, x):
        h = self.cnn(x)
        z0_mu, z0_log_sig_sq = self.fc1(h), self.fc2(h) # N,q & N,q
        return z0_mu, z0_log_sig_sq


class Decoder(nn.Module):
    def __init__(self, n_filt=8):
        super(Encoder, self).__init__()

        h_dim = n_filt*4**3 # encoder output is [4*n_filt,4,4]

        self.decnn = nn.Sequential(
            UnFlatten(4),
            nn.ConvTranspose2d(h_dim//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=(1,1)),
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)),
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*2, 1, kernel_size=5, stride=1, padding=(2,2)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.decnn(x)
        return h #z0_mu, z0_log_sig_sq