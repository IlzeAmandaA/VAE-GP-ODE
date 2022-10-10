import torch
import torch.nn as nn
from torch.distributions import kl_divergence as kl


# model implementation
class ODEGPVAE(nn.Module):
    def __init__(self, flow, vae, num_observations, steps, order=2, ts_dense_scale=1, dt=0.1):
        super(ODEGPVAE, self).__init__()

        self.flow = flow #Dynamics 
        self.num_observations = num_observations
        self.ts_dense_scale = ts_dense_scale
        self.vae = vae
        self.dt = dt
        self.v_steps = steps
        self.order = order

    def build_decoding(self, ztL, dims):
        """
        Given a mean of the latent space decode the input back into the original space.

        @param ztL: latent variable (L,N,T,2q)
        @param dims: dimensionality of the original variable 
        @return Xrec: reconstructed in original data space (L,N,T,nc,d,d)
        """
        L,N,T,nc,d,d = dims
        if self.order == 1:
            Xrec = self.vae.decoder(ztL) # L*N*T,nc,d,d
            Xrec = Xrec.view([L,N,T,nc,d,d]) # L,N,T,nc,d,d
        elif self.order == 2:
            q = ztL.shape[-1]//2
            st_muL = ztL[:,:,:,:q] # L,N,T,q Only the position is decoded
            Xrec = self.vae.decoder(st_muL) # L*N*T,nc,d,d
            Xrec = Xrec.view([L,N,T,nc,d,d]) # L,N,T,nc,d,d
        return Xrec
    
    def sample_trajectories(self,z0, T,L=1):
        ztL = []
        ts  = self.dt * torch.arange(T,dtype=torch.float).to(z0.device)
        #sample L trajectories
        for l in range(L):
            zt = self.flow(z0, ts) # N,T,2q 
            ztL.append(zt.unsqueeze(0)) # 1,N,T,2q
        ztL   = torch.cat(ztL,0) # L,N,T,2q
        return ztL

        
    def forward(self, X, L=1, T_custom=None):
        [N,T,nc,d,d] = X.shape

        if T_custom:
            T_orig = T
            T = T_custom

        #encode
        s0_mu, s0_logv = self.vae.encoder_s(X[:,0]) #N,q
        z0 = self.vae.encoder_s.sample(mu = s0_mu, logvar = s0_logv)
        v0_mu, v0_logv = None, None
        if self.order == 2:
            v0_mu, v0_logv = self.vae.encoder_v(torch.squeeze(X[:,0:self.v_steps]))
            v0 = self.vae.encoder_v.sample(mu= v0_mu, logvar = v0_logv)
            z0 = torch.concat([z0,v0],dim=1) #N, 2q

        #sample
        ztL = self.sample_trajectories(z0,T,L) # L,N,T,2q

        #decode
        Xrec = self.build_decoding(ztL, (L,N,T,nc,d,d))

        return Xrec, (s0_mu, s0_logv), (v0_mu, v0_logv)

    