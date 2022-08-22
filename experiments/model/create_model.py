from telnetlib import GA
import torch
from torch.distributions import MultivariateNormal 
from model.core.dsvpy import DSVGP_Layer
from model.core.flow import Flow
from model.core.distributions import Bernoulli, Multivariate_Standard
from model.core.vae import Encoder, Decoder
from model.core.odegpvae import ODEGPVAE



def build_model(args):
    """
    Builds a model object of odevaegp.ODEVAEGP based on training sequence

    @param args: model setup arguments
    @return: an object of ODEVAEGP class
    """
    gp = DSVGP_Layer(D_in=args.D_in, D_out=args.D_out, #2q, q
                     M=args.num_inducing,
                     S=args.num_features,
                     dimwise=args.dimwise,
                     q_diag=args.q_diag)

    flow = Flow(diffeq=gp, order=args.order, solver=args.solver, use_adjoint=args.use_adjoint)

    #likelihood = Gaussian(ndim=D) #2q
    likelihood = Bernoulli() #2q

    #prior distriobution p(Z)
    prior = Multivariate_Standard(args.D_in).to(args.device)

    #prior for q_ode
    prior_q = Multivariate_Standard(args.q).to(args.device)

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
                        order = args.order,
                        ts_dense_scale=args.ts_dense_scale,
                        beta=args.beta,
                        steps=args.steps)

    return odegpvae

def compute_loss(model, data, L, args):
    """
    Compute loss for ODEGPVAE optimization
    @param model: a odegpvae object
    @param data: true observation sequence 
    @param L: number of MC samples
    @param Ndata: number of training data points 
    @return: loss, nll, initial_state_kl, inducing_kl
    """
    lhood, kl_z = model.build_lowerbound_terms(data, L, args)
    kl_u = model.build_kl() 
    loss = - (lhood * args.Ndata - kl_z * args.Ndata - model.beta*kl_u) 
    return loss, -lhood, kl_z, kl_u
