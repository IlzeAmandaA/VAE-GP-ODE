import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, kl_divergence as kl

from torchdiffeq import odeint

from mlp import MLP 
from utilities import build_encoder, build_decoder, Dataset, get_dataset


# model implementation
class ODEVAE(nn.Module):
    def __init__(self, order=1, n_filt=8, q=8):
        super().__init__()
        self.order = order
        n = order*q
        h_dim = n_filt*4**3 # encoder output is [4*n_filt,4,4]
        # encoder
        self.encoder = build_encoder(n_filt)
        self.fc1 = nn.Linear(h_dim, n)
        self.fc2 = nn.Linear(h_dim, n)
        # differential function
        self.f   = MLP(n, q, L=2, H=100, act='relu')
        # decoder
        self.fc3 = nn.Linear(q, h_dim)
        self.decoder = build_decoder(h_dim, n_filt) 
        self.sp = nn.Softplus()

    @property
    def device(self):
        return list(self.fc1.parameters())[0].device

    def odevae_rhs(self, t, vs):
        if self.order==1:
            return self.f(vs)
        else:
            q   = vs.shape[1]//2
            dv  = self.f(vs) # N,q 
            ds  = vs[:,:q]  # N,q
            dvs = torch.cat([dv,ds],1) # N,2q
            return dvs

    def elbo(self, qz_m, qz_v, X, Xrec, Ndata, eps=1e-5):
        ''' Input:
                qz_m        - latent means [N,2q]
                qz_logv     - latent logvars [N,2q]
                X           - input images [N,T,nc,d,d]
                Xrec        - reconstructions [N,T,nc,d,d]
                Ndata       - number of sequences in the dataset (required for elbo
            Returns:
                likelihood
                prior on initial values
        '''
        Tdata = X.shape[1]
        Xrec_ = Xrec[:,:Tdata]
        # prior
        p = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_m))
        q = Normal(qz_m, qz_v)
        kl_z = kl(q,p)
        # likelihood
        lhood = torch.log(eps+Xrec_)*X + torch.log(eps+1-Xrec_)*(1-X) # N,T,nc,d,d
        lhood = lhood.sum([1,2,3,4]).mean(0) # N
        return Ndata*lhood.mean(), Ndata*kl_z.mean()

    def forward(self, X, Ndata=1, method='rk4', dt=0.1, Tode=None):
        ''' Input
                X          - input images [N,T,nc,d,d]
                Ndata      - number of sequences in the dataset (required for elbo)
                inst_enc   - whether instant encoding is used or not
                method     - numerical integration method
                dt         - numerical integration step size 
            Returns
                Xrec     - reconstructions from latent samples     - [N,nc,D,D]
                qz_m       - mean of the latent embeddings           - [N,q]
                qz_logv    - log variance of the latent embeddings   - [N,q]
                lhood-kl_z - ELBO   
                lhood      - reconstruction likelihood
                kl_z       - KL
        '''
        # encode
        [N,T,nc,d,d] = X.shape
        Tode = T if Tode is None else Tode
        h = self.encoder(X[:,0])
        qz0_m, qz0_logv = self.fc1(h), self.fc2(h) # N,2q & N,2q
        qz0_v = self.sp(qz0_logv)
        # ODE
        z0 = qz0_m + torch.randn_like(qz0_m)*qz0_v # N,2q
        t  = dt * torch.arange(Tode, dtype=torch.float).to(z0.device)
        f  = lambda t,vs: self.odevae_rhs(t, vs) # make the ODE forward function
        zt = odeint(f, z0, t, method=method) # T,N,2q & T,N
        zt = zt.permute([1,0,2]) # N,T,2q
        # decode
        if self.order==1:
            q  = qz0_m.shape[1]
            st = zt.contiguous().view([N*Tode,q])
        else:
            q  = qz0_m.shape[1]//2
            st = zt[:,:,q:].contiguous().view([N*Tode,q]) # NT,q
        Xrec  = self.decoder(self.fc3(st)) # N*T,nc,d,d
        Xrec  = Xrec.view([N,Tode,nc,d,d]) # N,T,nc,d,d
        # likelihood and elbo
        lhood, kl_z = self.elbo(qz0_m, qz0_v, X, Xrec, Ndata)
        elbo = lhood - kl_z
        return Xrec, qz0_m, qz0_v, zt, lhood, kl_z