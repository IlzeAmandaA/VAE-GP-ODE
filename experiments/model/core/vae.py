import torch
import torch.nn as nn
from torch.distributions import Normal
from functools import reduce
from model.misc.torch_utils import Flatten, UnFlatten

EPSILON = 1e-3

class VAE(nn.Module):
    def __init__(self, steps = 1, n_filt=8, q=8, D_in=16, device='cpu', order=1, distribution='bernoulli'):
        super(VAE, self).__init__()


        # encoder position
        self.encoder_s = Encoder(steps= 1, n_filt=n_filt, q=q)

        if order == 2:
            # encoder velocity
            self.encoder_v = Encoder(steps = steps, n_filt=n_filt, q=q)

        # decoder
        self.decoder = Decoder(n_filt=n_filt, q=q, distribution=distribution)

        # prior 
     
        self.prior = Normal(torch.zeros(D_in).to(device), torch.ones(D_in).to(device))

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

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )

        # in_features = 8 * reduce(lambda x, y: (x - 4) * (y - 4), (28,28))

        self.fc1 = nn.Linear(h_dim, q)
        self.fc2 = nn.Linear(h_dim, q)

        # self.fc1 = nn.Linear(in_features, q)
        # self.fc2 = nn.Linear(in_features, q)

    def forward(self, x):
        h = self.cnn(x)
        z0_mu, z0_log_sig_sq = self.fc1(h), self.fc2(h) # N,q & N,q
        return z0_mu, z0_log_sig_sq

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def q_dist(self, mu_s, logvar_s, mu_v=None, logvar_v=None):

        if mu_v is not None:
            means = torch.cat((mu_s,mu_v), dim=1)
            log_v = torch.cat((logvar_s,logvar_v), dim=1)
        else:
            means = mu_s
            log_v = logvar_s

        try:
            std_ = nn.functional.softplus(log_v)
        except:
            std_ = EPSILON + nn.functional.softplus(log_v)

        return Normal(means, std_) #N,q

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

        # out_features = 8 * reduce(lambda x, y: (x - 4) * (y - 4), (28,28))
        # self.decnn = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3, stride=1),
        #     nn.Sigmoid()
        # )
        # self.fc = nn.Linear(q, out_features)
        # self.relu = nn.ReLU()

    def forward(self, x):
        
        #x= self.fc(x)
        # x = self.relu(x)
        # img_shapes = [x - 4 for x in (28,28)]
        # x = x.reshape(x.shape[0], 8, *img_shapes)
        L,N,T,q = x.shape
        s = self.fc(x.contiguous().view([L*N*T,q]) ) # N*T,q
        h = self.decnn(s)
        return h 
    
    @property
    def device(self):
        return next(self.parameters()).device


    def log_prob(self, x,z, L=1):
        '''
        x           - input images [N,T,1,nc,nc]
        z           - reconstructions [L,N,T,1,nc,nc]
        '''
        XL = x.repeat([L,1,1,1,1,1]) # L,N,T,nc,d,d 
        if self.distribution == 'bernoulli':
            try:
                log_p = torch.log(z)*XL + torch.log(1-z)*(1-XL) # L,N,T,nc,d,d
            except:
                log_p = torch.log(EPSILON+z)*XL + torch.log(EPSILON+1-z)*(1-XL) # L,N,T,nc,d,d
        else:
            raise ValueError('Currently only bernoulli dist implemented')

        return log_p
