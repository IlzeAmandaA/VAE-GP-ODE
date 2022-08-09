from pipes import quote
import torch
import torch.nn as nn


# model implementation
class ODEGPVAE(nn.Module):
    def __init__(self, flow, enc_s, enc_v, decoder, num_observations, likelihood, prior, prior_q, steps, ts_dense_scale=1, beta=1, dt=0.1):
        super(ODEGPVAE, self).__init__()

        self.flow = flow #Dynamics 
        self.num_observations = num_observations
        self.likelihood = likelihood
        self.ts_dense_scale = ts_dense_scale
        self.enc_s = enc_s
        self.enc_v = enc_v
        self.decoder = decoder
        self.prior = prior
        self.prior_q = prior_q
        self.beta = beta
        self.dt = dt
        self.v_steps = steps

    def build_encoding(self, X):
        s0_mu, s0_logv = self.enc_s(X[:,0])
        v0_mu, v0_logv = self.enc_v(torch.squeeze(X[:,0:self.v_steps]))
        # latent samples
        eps_s0   = torch.randn_like(s0_mu) #N,q
        eps_v0   = torch.randn_like(v0_mu) #N,q
        s0 = s0_mu + eps_s0*torch.exp(s0_logv) #N,q
        v0 = v0_mu + eps_v0*torch.exp(v0_logv) #N,q
        # TODO ask: in principle, this a dummy variable, right?
        logp0 = self.prior_q.log_prob(eps_s0) + self.prior_q.log_prob(eps_v0) # N 
        z0 = torch.concat([v0,s0],dim=1)
        return z0, logp0

    def build_decoding(self, st_muL, dims):
        L,N,T,nc,d,d = dims
        Xrec = self.decoder(st_muL) # L*N*T,nc,d,d
        Xrec = Xrec.view([L,N,T,nc,d,d]) # L,N,T,nc,d,d
        return Xrec


    def build_flow(self, z0, logp0, T, sample=False):
        """
        Given an initial state and time sequence, perform forward ODE integration
        Optionally, the time sequence can be made dense based on self.ts_dense_scale parameter

        @param x0: initial state tensor (N,D)
        @param ts: time sequence tensor (T,)
        @return: forward solution tensor (N,T,D)
        """
        ts  = self.dt * torch.arange(T,dtype=torch.float).to(z0.device)
        zt, logp = self.flow(z0, logp0, ts, sample=sample)
        return zt, logp 

    def build_kl(self):
        """
        Computes KL divergence between inducing prior and posterior.

        @return: inducing KL scaled by the number of observations
        """
        return self.flow.kl() / self.num_observations #TODO divide or multiply by number of observations

    def build_lowerbound_terms(self, X, L):
        """
        Given observed states and time, builds the individual terms for the lowerbound computation

        @param X: observed sequence tensor (N,T,D)
        @param L: number of MC samples
        @return: ll, kl_z
        """
        [N,T,nc,d,d] = X.shape
        #encode
        z0, logp0 = self.build_encoding(X) #N,2q & N,q (25,16) (25,8)
        q = z0.shape[1]//2
        ztL = []
        logpL = []
        #sample L trajectories
        for l in range(L):
            zt, logp = self.build_flow(z0, logp0, T) # T,N,2q & T,N
            ztL.append(zt.unsqueeze(0)) # 1,N,T,2q
            logpL.append(logp.unsqueeze(0)) # 1,N,T
        ztL   = torch.cat(ztL,0) # L,N,T,2q 1x25x16x16
        logpL = torch.cat(logpL) # L,N,T    
        #decode
        st_muL = ztL[:,:,:,q:] # L,N,T,q Only the position is decoded
        Xrec = self.build_decoding(st_muL, (L,N,T,nc,d,d))

        ##### compute loss terms ######
        #log p(z)
        log_pzt = self.prior.log_prob(ztL.contiguous().view([L*N*T,2*q])) # L*N*T
        log_pzt = log_pzt.view([L,N,T]) # L,N,T
        # kl_zt := KL[q(Z|X,f)||p(Z)]
        kl_zt   = logpL - log_pzt  # L,N,T
        kl_z    = kl_zt.sum(2).mean(0) # N
        #ll
        loglik = self.likelihood.log_prob(X,Xrec,L) 
        
        return loglik.mean(), kl_z.mean() / self.num_observations #TODO divide or multiply by number of observations
    
    def forward(self, X):
        [N,T,nc,d,d] = X.shape
        #encode
        z0, logp0 = self.build_encoding(X) #N,2q 
        q = z0.shape[1]//2
        #sample flow
        zt, _ = self.build_flow(z0, logp0, T, sample=True) # T,N,2q & None
        # decode
        st_mu = zt[:,:,q:] # N,T,q
        Xrec_mu = self.build_decoding(st_mu.unsqueeze(0),(1,N,T,nc,d,d)) # N,T,nc,d,d
        # error
        mse = torch.mean((Xrec_mu-X)**2)
        return Xrec_mu, mse