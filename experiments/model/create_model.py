import torch
from torch.distributions import MultivariateNormal 
from model.core.dsvpy import DSVGP_Layer
from model.core.flow import Flow
from model.core.distributions import Bernoulli, Gaussian
from model.core.vae import Encoder, Decoder
from model.core.odegpvae import ODEGPVAE



def build_model(args):
    """
    Builds a model object of gpode.SequenceModel based on training sequence

    @param data_ys: observed/training sequence of (N,T,D) dimensions
    @param args: model setup arguments
    @return: an object of gpode.SequenceModel class
    """
    gp = DSVGP_Layer(D_in=args.q*2, D_out=args.q, #2q, q
                     M=args.num_inducing,
                     S=args.num_features,
                     dimwise=args.dimwise,
                     q_diag=args.q_diag,
                     device=args.device)

    flow = Flow(diffeq=gp, solver=args.solver, use_adjoint=args.use_adjoint)

    #likelihood = Gaussian(ndim=D) #2q
    likelihood = Bernoulli() #2q

    #prior distriobution p(Z)
   # prior = Gaussian(ndim=args.q*2) #2q
    prior = MultivariateNormal(torch.zeros(args.q*2).to(args.device),torch.eye(args.q*2).to(args.device)) 

    #dummy prior for q_ode
    prior_q =  MultivariateNormal(torch.zeros(args.q).to(args.device),torch.eye(args.q).to(args.device)) 
    #prior_q = Gaussian(ndim=args.q) #q

    # encoder position
    encoder_s = Encoder(steps= 1, n_filt=args.n_filt, q=args.q)

    # encoder velocity
    encoder_v = Encoder(steps = args.steps, n_filt=args.n_filt, q=args.q)

    # decoder
    decoder = Decoder(n_filt=args.n_filt, q=args.q)

    odegpvae = ODEGPVAE(flow=flow,
                        enc_s = encoder_s,
                        enc_v = encoder_v,
                        decoder = decoder,
                        num_observations= args.batch * args.T, #TODO N*T*D
                        likelihood=likelihood,
                        prior = prior,
                        prior_q= prior_q,
                        ts_dense_scale=args.ts_dense_scale,
                        beta=args.beta,
                        steps=args.steps)

    return odegpvae

def compute_loss(model, data, L):
    """
    Compute loss for ODEGPVAE optimization
    @param model: a odegpvae object
    @param data: true observation sequence 
    @param L: number of MC samples
    @return: loss, nll, initial_state_kl, inducing_kl
    """
    lhood, kl_z = model.build_lowerbound_terms(data, L) #should correspond to reconstruction likelihood and prior on z 
    kl_u = model.build_kl() #kl inducing 
    loss = - (lhood - kl_z - model.beta*kl_u)
    return loss, -lhood, kl_z, kl_u
