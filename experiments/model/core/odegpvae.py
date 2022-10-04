import torch
import torch.nn as nn


# model implementation
class ODEGPVAE(nn.Module):
    def __init__(self, flow, vae, num_observations, steps, order=2, ts_dense_scale=1, beta=1, dt=0.1):
        super(ODEGPVAE, self).__init__()

        self.flow = flow #Dynamics 
        self.num_observations = num_observations
        self.ts_dense_scale = ts_dense_scale
        self.vae = vae
        self.beta = beta
        self.dt = dt
        self.v_steps = steps
        self.order = order

    def build_encoding(self, X):
        """
        Encode the high-dimensional input into a latent space 

        @param X: batch of original data (N,T,nc,d,d)
        @return z0: encoded representation (N, 2q)
        @return logp0: initial probability density (N)
        """
        if self.order ==1:
            s0_mu, s0_logv = self.vae.encoder_s(X[:,0]) #N,2q 
            eps_s0   = torch.randn_like(s0_mu) #N,2q
            z0 = s0_mu + eps_s0*torch.exp(s0_logv) #N,2q
            logp0 = self.prior_q.log_prob(eps_s0) # N
        elif self.order==2:
            s0_mu, s0_logv = self.vae.encoder_s(X[:,0])
            v0_mu, v0_logv = self.vae.encoder_v(torch.squeeze(X[:,0:self.v_steps]))
            # latent samples
            eps_s0   = torch.randn_like(s0_mu) #N,q
            eps_v0   = torch.randn_like(v0_mu) #N,q
            s0 = s0_mu + eps_s0*torch.exp(s0_logv) #N,q
            v0 = v0_mu + eps_v0*torch.exp(v0_logv) #N,q
            logp0 = self.prior_q.log_prob(eps_s0) + self.prior_q.log_prob(eps_v0) # N 
            z0 = torch.concat([v0,s0],dim=1) #N, 2q
        return z0, logp0

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


    def build_flow(self, z0, T):
        """
        Given an initial state and time sequence, perform forward ODE integration
        Optionally, the time sequence can be made dense based on self.ts_dense_scale parameter

        @param z0: initial latent state tensor (N,2q)
        @param logp0: initial probability (N)
        @param ts: time sequence tensor (T,)
        @return flow: forward solution
        """
        ts  = self.dt * torch.arange(T,dtype=torch.float).to(z0.device)
        return self.flow(z0, ts)

    def build_kl(self):
        """
        Computes KL divergence between inducing prior and posterior.

        @return: inducing KL scaled by the number of observations
        """
        return self.flow.kl() #/ self.num_observations (N*T*D)

    # def sample_trajectores_trace(self,z0,logp0,T,L,trace):
    #     ztL = []
    #     logpL = []
    #     #sample L trajectories
    #     for l in range(L):
    #         zt, logp = self.build_flow(z0, logp0, T, trace) # N,T,2q & N,T
    #         ztL.append(zt.unsqueeze(0)) # 1,N,T,2q
    #         logpL.append(logp.unsqueeze(0)) # 1,N,T
    #     ztL   = torch.cat(ztL,0) # L,N,T,2q 1x25x16x16
    #     logpL = torch.cat(logpL) # L,N,T 
    #     return ztL, logpL
    
    def sample_trajectories(self,z0, T,L):
        ztL = []
        #sample L trajectories
        for l in range(L):
            zt = self.build_flow(z0, T) # N,T,2q & N,T
            ztL.append(zt.unsqueeze(0)) # 1,N,T,2q
        ztL   = torch.cat(ztL,0) # L,N,T,2q
        return ztL

    def build_vae_terms(self, X, L):
        """
        Given observed states and time, builds the individual terms for the lowerbound computation

        @param X: observed sequence tensor (N,T,nc,d,d)
        @param L: number of MC samples
        @return loglikelihood: The reconstruction likelihood, assume Bernoulli distribution
        @return kl_z: The apprxomiated KL between q_ode() distriubtion and prior p(z)
        """
        [N,T,nc,d,d] = X.shape
        #encode
        s0_mu, s0_logv = self.vae.encoder_s(X[:,0]) #N,q
        z0 = self.vae.encoder_s.sample(mu = s0_mu, logvar = s0_logv)
        v0_mu, v0_logv = None, None
        if self.order == 2:
            v0_mu, v0_logv = self.vae.encoder_v(torch.squeeze(X[:,0:self.v_steps]))
            v0 = self.vae.encoder_v.sample(mu= v0_mu, logvar = v0_logv)
            z0 = torch.concat([z0,v0],dim=1) #N, 2q

        ztL = self.sample_trajectories(z0,T,L) # L,N,T,2q

        #decode
        Xrec = self.build_decoding(ztL, (L,N,T,nc,d,d))

        ##### loss terms ######
        #KL regularizer
        log_pz = self.vae.prior.log_prob(ztL[:,:,0,:].view([L*N,ztL.shape[-1]])) # L*N
        log_q_enc = self.vae.encoder_s.log_prob(s0_mu, s0_logv, v0_mu, v0_logv, ztL[:,:,0,:].view([L*N,ztL.shape[-1]]), L) #L*N
        KL_reg = (log_pz - log_q_enc) # 1,L*N

        #Reconstruction log-likelihood
        RE = self.vae.decoder.log_prob(X,Xrec,L) #L,N,T,d,nc,nc
        RE = RE.sum([2,3,4,5]).mean(0) #N
        
        return RE.mean(), KL_reg.mean()  

    def forward(self, X, T_custom=None):
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

        #sample flow
        zt = self.build_flow(z0, T) # N,T,2q & None
        # decode
        Xrec_mu = self.build_decoding(zt.unsqueeze(0),(1,N,T,nc,d,d)).squeeze(0) # N,T,nc,d,d
        # error
        if T_custom:
            mse = torch.mean((Xrec_mu[:,:T_orig,:]-X)**2)
        else:
            mse = torch.mean((Xrec_mu-X)**2)
        return Xrec_mu, mse

    