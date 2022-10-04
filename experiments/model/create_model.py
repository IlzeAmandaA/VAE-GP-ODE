from model.core.svpy import SVGP_Layer
from model.core.flow import Flow
from model.core.vae import VAE  
from model.core.odegpvae import ODEGPVAE

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

    flow = Flow(diffeq=gp, order=args.order, solver=args.solver, use_adjoint=args.use_adjoint)

    vae = VAE(steps = args.steps, n_filt=args.n_filt, q=args.q, order= args.order, device=args.device, distribution='bernoulli')

    odegpvae = ODEGPVAE(flow=flow,
                        vae= vae,
                        num_observations= args.Ndata,
                        order = args.order,
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
    @param Ndata: number of training data points 
    @return: loss, nll, initial_state_kl, inducing_kl
    """
    RE, KL_reg = model.build_vae_terms(data, L)
    KL_u = model.build_kl() 
    loss = - (RE * model.num_observations  + KL_reg * model.num_observations  - model.beta*KL_u) 
    return loss, -RE, KL_reg, KL_u 

