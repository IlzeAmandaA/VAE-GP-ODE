import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torchdiffeq import odeint

from model.misc.torch_utils import Flatten, UnFlatten
from model.core.distributions import Multivariate_Standard, log_normal_diag





class VAE(nn.Module):
    def __init__(self, steps = 1, n_filt=8, q=8, d=16, device='cpu', order=1, distribution='bernoulli'):
        super(VAE, self).__init__()


        # encoder position
        self.encoder_s = Encoder(steps= 1, n_filt=n_filt, q=q)

        if order == 2:
            # encoder velocity
            self.encoder_v = Encoder(steps = steps, n_filt=n_filt, q=q)

        # decoder
        self.decoder = Decoder(n_filt=n_filt, q=q, distribution=distribution)

        # prior 
        self.prior = MultivariateNormal(torch.zeros(d).to(device), torch.eye(d).to(device)) # Multivariate_Standard(L=d, device=device)

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

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def log_prob(self, mu_s, logvar_s, mu_v, logvar_v, z,L):
        [N, q] = z.shape 

        if mu_v is None or logvar_v is None:
            means = mu_s.repeat([L,1])  # L*N,q
            std_  = 1e-3+logvar_s.exp().repeat([L,1])
        
        else:
            means = torch.cat((mu_s,mu_v), dim=1).repeat([L,1]) # N, 2*q
            log_v = torch.cat((logvar_s,logvar_v), dim=1) # N, 2*q
            std_  = 1e-3+log_v.exp().repeat([L,1]) #N, 2q
         
        covariance = torch.eye(q, q, device=std_.device).unsqueeze(0).repeat(N, 1, 1) * std_[:,None,:] #N,D,D
        mn = MultivariateNormal(means, covariance) #.to(z.device)
        return mn.log_prob(z)



    @property
    def device(self):
        return next(self.parameters()).device


class Decoder(nn.Module):
    def __init__(self, n_filt=8, q=8, distribution = 'bernoulli'):
        super(Decoder, self).__init__()

        self.distribution = distribution

        h_dim = n_filt*4**3 # encoder output is [4*n_filt,4,4]
        self.fc = nn.Linear(q, h_dim)

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
        L,N,T,q = x.shape
        s = self.fc(x.contiguous().view([L*N*T,q]) ) # N*T,q #might be detrimental 
        h = self.decnn(s)
        return h #z0_mu, z0_log_sig_sq
    
    @property
    def device(self):
        return next(self.parameters()).device


    def log_prob(self, x,z, L=1):
        '''
        x           - input images [N,T,nc,d,d]
        z           - reconstructions [L,N,T,nc,d,d]
        '''
        XL = x.repeat([L,1,1,1,1,1]) # L,N,T,nc,d,d 
        if self.distribution == 'bernoulli':
            log_p = torch.log(1e-3+z)*XL + torch.log(1e-3+1-z)*(1-XL) # L,N,T,nc,d,d
        else:
            raise ValueError('Currently only bernoulli dist implemented')

        return log_p
