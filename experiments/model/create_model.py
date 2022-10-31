from model.core.svpy import SVGP_Layer
from model.core.flow import Flow
from model.core.vae import VAE  
from model.core.odegpvae import ODEGPVAE
import torch
from torch.distributions import kl_divergence as kl


def build_model(args):
    """
    Builds a model object of odevaegp.ODEVAEGP based on training sequence

    @param args: model setup arguments
    @return: an object of ODEVAEGP class
    """
    gp = SVGP_Layer(D_in=args.D_in, D_out=args.D_out, #2q, q
                     M=args.num_inducing,
                     S=args.num_features,
                     dimwise=args.dimwise,
                     q_diag=args.q_diag,
                     device= args.device,
                     kernel = args.kernel)

    flow = Flow(diffeq=gp, order=args.ode, solver=args.solver, use_adjoint=args.use_adjoint)

    vae = VAE(frames = args.frames, n_filt=args.n_filt, latent_dim=args.latent_dim ,order= args.ode, device=args.device)

    odegpvae = ODEGPVAE(flow=flow,
                        vae= vae,
                        num_observations= args.Ndata,
                        order = args.ode,
                        steps=args.frames,
                        dt = args.dt)

    return odegpvae

def elbo(model, X, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L):
    ''' Input:
            qz_m        - latent means [N,2q]
            qz_logv     - latent logvars [N,2q]
            X           - input images [L,N,T,nc,d,d]
            Xrec        - reconstructions [L,N,T,nc,d,d]
        Returns:
            likelihood
            kl terms
    '''
    # KL reg
    q = model.vae.encoder.q_dist(s0_mu, s0_logv, v0_mu, v0_logv)
    kl_reg = kl(q, model.vae.prior).sum(-1) #N

    #Reconstruction log-likelihood
    lhood = model.vae.decoder.log_prob(X,Xrec,L) #L,N,T,d,nc,nc
    lhood = lhood.sum([2,3,4,5]).mean(0) #N

    # KL inudcing
    kl_u = model.flow.kl()

    return lhood.mean(), kl_reg.mean(), kl_u 


def compute_loss(model, data, L):
    """
    Compute loss for optimization
    @param model: a odegpvae object
    @param data: true observation sequence 
    @param L: number of MC samples
    @param Ndata: number of training data points 
    @return: loss, nll, regularizing_kl, inducing_kl
    """
    Xrec,(s0_mu, s0_logv), (v0_mu, v0_logv) = model(data,L)
    lhood, kl_reg, kl_gp = elbo(model, data, Xrec, s0_mu, s0_logv, v0_mu, v0_logv,L)
    loss = - (lhood * model.num_observations - kl_reg * model.num_observations - kl_gp)
    return loss, -lhood, kl_reg, kl_gp

def compute_test_error(X, Xrec):
    assert list(X.shape) == list(Xrec.shape), f'incorrect shapes X: {list(X.shape)}, X_Rec: {list(Xrec.shape)}'
    return torch.mean((Xrec-X)**2)


