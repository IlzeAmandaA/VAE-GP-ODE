import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torchdiffeq import odeint

from .utils import *
from .gpode import GPODE
from .vae import Encoder, Decoder


# model implementation
class ODEGPVAE(nn.Module):
    def __init__(self, n_filt=8, q=8, device="cuda", v_steps=10):
        super(ODEGPVAE, self).__init__()
        
        #number of data points from time 0 used in encoding velocity
        self.v_steps = v_steps

        # encoder position
        self.enc_s = Encoder(steps= 1, n_filt=n_filt, q=q)

        # encoder velocity
        self.enc_v = Encoder(steps = v_steps, n_filt=n_filt, q=q)

        h_dim = n_filt*4**3 # encoder output is [4*n_filt,4,4]
        self.fc3 = nn.Linear(q, h_dim)

        # differential function
        self.ode_model = GPODE(2*q, q, device=device)

        # decoder
        self.decoder = Decoder(n_filt=n_filt)

        #prior distriobution p(Z)
        self._zero_mean = torch.zeros(2*q).to(device)
        self._eye_covar = torch.eye(2*q).to(device) 
        self.mvn = MultivariateNormal(self._zero_mean, self._eye_covar)

        #lop0 dummy var
        self.mvn_q = MultivariateNormal(torch.zeros(q).to(device),torch.eye(q).to(device))

        # downweighting the BNN KL term is helpful if self.bnn is heavily overparameterized
        self.beta = 1.0 # 2*q/self.bnn.kl().numel()

    def elbo(self, q, zode_L, logpL, X, XrecL, Ndata, qz_enc_m=None, qz_enc_logv=None):
        ''' Input:
                q           - latent shape  [q]
                zode_L      - latent trajectory samples [L,N,T,2q]
                logpL       - densities of latent trajectory samples [L,N,T]
                X           - input images [N,T,nc,d,d]
                XrecL       - reconstructions [L,N,T,nc,d,d]
                Ndata       - number of sequences in the dataset (required for elbo
                qz_enc_m    - encoder density means  [N*T,2*q]
                qz_enc_logv - encoder density variances [N*T,2*q]
            Returns:
                likelihood
                prior on ODE trajectories KL[q_ode(z_{0:T})||N(0,I)]
                prior on BNN weights
                instant encoding term KL[q_ode(z_{0:T})||q_enc(z_{0:T}|X_{0:T})] (if required) 
        '''
        [N,T,nc,d,d] = X.shape
        L = zode_L.shape[0]

        # prior p(Z), MUltivariate Gaussian
        log_pzt = self.mvn.log_prob(zode_L.contiguous().view([L*N*T,2*q])) # L*N*T
        log_pzt = log_pzt.view([L,N,T]) # L,N,T

        #KL inducing loss, GP 
        kl_u    = self.ode_model.kl_inducing

        # reconstruction likelihood log p(X|Z), Bernoulli dist 
        XL = X.repeat([L,1,1,1,1,1]) # L,N,T,nc,d,d 
        lhood_L = torch.log(1e-3+XrecL)*XL + torch.log(1e-3+1-XrecL)*(1-XL) # L,N,T,nc,d,d
        lhood = lhood_L.sum([2,3,4,5]).mean(0) # N

        # kl_zt := KL[q(Z|X,f)||p(Z)]
        kl_zt   = logpL - log_pzt  # L,N,T
        kl_z    = kl_zt.sum(2).mean(0) # N

        if qz_enc_m is not None: # instant encoding
            qz_enc_mL    = qz_enc_m.repeat([L,1])  # L*N*T,2*q
            qz_enc_logvL = qz_enc_logv.repeat([L,1])  # L*N*T,2*q
            mean_ = qz_enc_mL.contiguous().view(-1) # L*N*T*2*q
            std_  = 1e-3+qz_enc_logvL.exp().contiguous().view(-1) # L*N*T*2*q
            qenc_zt_ode = Normal(mean_,std_).log_prob(zode_L.contiguous().view(-1)).view([L,N,T,2*q])
            qenc_zt_ode = qenc_zt_ode.sum([3]) # L,N,T
            inst_enc_KL = logpL - qenc_zt_ode
            inst_enc_KL = inst_enc_KL.sum(2).mean(0) # N
            return Ndata*lhood.mean(), Ndata*kl_z.mean(), kl_u, Ndata*inst_enc_KL.mean()
        else:

            return Ndata*lhood.mean(), Ndata*kl_z.mean(), kl_u

    def forward(self, X, Ndata, L=1, inst_enc=False, method='dopri5', dt=0.1):
        ''' Input
                X          - input images [N,T,nc,d,d]
                Ndata      - number of sequences in the dataset (required for elbo)
                L          - number of Monta Carlo draws (from GP)
                inst_enc   - whether instant encoding is used or not
                method     - numerical integration method
                dt         - numerical integration step size 
            Returns
                Xrec_mu    - reconstructions from the mean embedding - [N,nc,D,D]
                Xrec_L     - reconstructions from latent samples     - [L,N,nc,D,D]
                qz_m       - mean of the latent embeddings           - [N,q]
                qz_logv    - log variance of the latent embeddings   - [N,q]
                lhood-kl_z - ELBO   
                lhood      - reconstruction likelihood
                kl_z       - KL
        '''
        ######## encode ###########
        [N,T,nc,d,d] = X.shape
        s0_mu, s0_logv = self.enc_s(X[:,0])
        v0_mu, v0_logv = self.enc_v(torch.squeeze(X[:,0:self.v_steps]))
        
        q = s0_mu.shape[1]

        # latent samples
        eps_s0   = torch.randn_like(s0_mu) #N,q
        eps_v0   = torch.randn_like(v0_mu) #N,q
        s0 = s0_mu + eps_s0*torch.exp(s0_logv) #N,q
        v0 = v0_mu + eps_v0*torch.exp(v0_logv) #N,q

        #z0    = qz0_m + eps*torch.exp(qz0_logv) # N,2q
        # TODO ask: in principle, this a dummy variable, right?
        logp0 = self.mvn_q.log_prob(eps_s0) + self.mvn_q.log_prob(eps_v0) # N 

        ######## ODE ###########
        z0 = torch.concat([v0,s0],dim=1)
        t  = dt * torch.arange(T,dtype=torch.float).to(z0.device)
        ztL   = []
        logpL = []
        # sample L trajectories
        for l in range(L):
            zt,logp = self.ode_model(z0, logp0, t, method) # T,N,2q & T,N
            ztL.append(zt.permute([1,0,2]).unsqueeze(0)) # 1,N,T,2q
            logpL.append(logp.permute([1,0]).unsqueeze(0)) # 1,N,T

        ztL   = torch.cat(ztL,0) # L,N,T,2q 1x25x16x16
        logpL = torch.cat(logpL) # L,N,T
        
        ######## decode ######### 
        st_muL = ztL[:,:,:,q:] # L,N,T,q Only the position is decoded
        s = self.fc3(st_muL.contiguous().view([L*N*T,q]) ) # L*N*T,h_dim
        Xrec = self.decoder(s) # L*N*T,nc,d,d
        Xrec = Xrec.view([L,N,T,nc,d,d]) # L,N,T,nc,d,d


        # likelihood and elbo
        if inst_enc:
            h = self.encoder(X.contiguous().view([N*T,nc,d,d]))
            qz_enc_m, qz_enc_logv = self.fc1(h), self.fc2(h) # N*T,2q & N*T,2q
            lhood, kl_z, kl_w, inst_KL = \
                self.elbo(q, ztL, logpL, X, Xrec, Ndata, qz_enc_m, qz_enc_logv)
            elbo = lhood - kl_z - inst_KL - self.beta*kl_w

        else:
            lhood, kl_z, kl_w = self.elbo(q, ztL, logpL, X, Xrec, Ndata) # TODO check loss term for GP-ODE (might have to adjust the loss here)
            elbo = lhood - kl_z - self.beta*kl_w 


        return Xrec, (s0_mu, s0_logv), (v0_mu, v0_logv), ztL, elbo, lhood, kl_z, self.beta*kl_w

    def mean_rec(self, X, method='dopri5', dt=0.1):
        [N,T,nc,d,d] = X.shape
        ######## encode ###########
        [N,T,nc,d,d] = X.shape
        s0_mu, s0_logv = self.enc_s(X[:,0])
        v0_mu, v0_logv = self.enc_v(torch.squeeze(X[:,0:self.v_steps]))

        q = s0_mu.shape[1]

        ######## ODE ###########
        z0 = torch.concat([v0_mu,s0_mu],dim=1)
        t  = dt * torch.arange(T,dtype=torch.float).to(z0.device)
        zt = self.ode_model.sample(z0, t, method).permute([1,0,2])  # N,T,2q

        # decode
        st_mu = zt[:,:,q:] # N,T,q
        s = self.fc3(st_mu.contiguous().view([N*T,q]) ) # N*T,q
        Xrec_mu = self.decoder(s) # N*T,nc,d,d
        Xrec_mu = Xrec_mu.view([N,T,nc,d,d]) # N,T,nc,d,d
        # error
        mse = torch.mean((Xrec_mu-X)**2)
        return Xrec_mu,mse