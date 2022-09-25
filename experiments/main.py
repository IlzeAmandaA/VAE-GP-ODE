import os 
import time
from datetime import timedelta
import argparse
import torch
import sys
from datetime import datetime
from data.wrappers import load_data
from model.create_model import build_model, compute_loss
from model.create_plots import plot_results
from model.misc.plot_utils import *
from model.misc import io_utils
from model.misc import log_utils 
from model.misc.torch_utils import seed_everything
from model.misc.settings import settings
from model.core.initialization import initialize_and_fix_kernel_parameters

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]
KERNELS = ['RBF', 'DF']
parser = argparse.ArgumentParser('Learning latent dyanmics with OdeVaeGP')

# model parameters
parser.add_argument('--num_features', type=int, default=256,
                    help="Number of Fourier basis functions (for pathwise sampling from GP)")
parser.add_argument('--num_inducing', type=int, default=100,
                    help="Number of inducing points for the sparse GP")
parser.add_argument('--variance', type=float, default=0.25,
                    help="Initial value for rbf variance")
parser.add_argument('--lengthscale', type=float, default=0.65,
                    help="Initial value for rbf lengthscale")
parser.add_argument('--dimwise', type=eval, default=True,
                    help="Specify separate lengthscales for every output dimension")
parser.add_argument('--q_diag', type=eval, default=False,
                    help="Diagonal posterior approximation for inducing variables")
parser.add_argument('--num_latents', type=int, default=5,
                    help="Number of latent dimensions for training")
parser.add_argument('--trace', type=eval, default=True,
                    help="Compute trace")
parser.add_argument('--kl_0', type=eval, default=False,
                    help="Specifies to set initial KL to 0")
parser.add_argument('--order', type=int, default=2,
                    help="order of ODE")
parser.add_argument('--continue_training', type=eval, default=False,
                    help="If set to True continoues training of a previous model")
parser.add_argument('--model_path', type=str, default='None',
                    help="path from where to load previous model, should be of the form results/mnist_*/*.pth")



# data processing arguments
parser.add_argument('--data_root', type=str, default='data/',
                    help="Data location")
parser.add_argument('--task', type=str, default='mnist',
                    help="Experiment type")
parser.add_argument('--mask', type=eval, default=True,
                    help="select a subset of mnist data")
parser.add_argument('--value', type=int, default=3,
                    help="training choice")
parser.add_argument('--data_seqlen', type=int, default=100,
                    help="Training sequence length")
parser.add_argument('--batch', type=int, default=25,
                    help="batch size")
parser.add_argument('--T', type=int, default=16,
                    help="Number of time points")
parser.add_argument('--Ndata', type=int, default=500,
                    help="Number training data points")
parser.add_argument('--rotrand', type=eval, default=False,
                    help="if True multiple initial rotatio angles")

#vae arguments
parser.add_argument('--q', type=int, default=8,
                    help="Latent space dimensionality")
parser.add_argument('--n_filt', type=int, default=8,
                    help="Number of filters in the cnn")
parser.add_argument('--steps', type=int, default=5,
                    help="Number of timesteps used for encoding velocity")

# ode solver arguments
parser.add_argument('--D_in', type=int, default=16,
                    help="ODE f(x) input dimensionality")
parser.add_argument('--D_out', type=int, default=8,
                    help="ODE f(x) output dimensionality")
parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--ts_dense_scale', type=int, default=2,
                    help="Factor for making a dense integration time grid (useful for explicit solvers)")
parser.add_argument('--use_adjoint', type=eval, default=False,
                    help="Use adjoint method for gradient computation")
parser.add_argument('--beta', type=int, default=1,
                    help="Factor to scale the inducing KL loss effect")
parser.add_argument('--kernel', type=str, default='RBF', choices=KERNELS,
                    help="ODE solver for numerical integration")

# training arguments
parser.add_argument('--Nepoch', type=int, default=500, #10_000
                    help="Number of gradient steps for model training")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate for model training")
parser.add_argument('--eval_sample_size', type=int, default=128,
                    help="Number of posterior samples to evaluate the model predictive performance")
parser.add_argument('--save', type=str, default='results/mnist',
                    help="Directory name for saving all the model outputs")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--log_freq', type=int, default=5,
                    help="Logging frequency while training")
parser.add_argument('--device', type=str, default='cuda:0',
                    help="device")

#plotting arguments
parser.add_argument('--Tlong', type=int, default=3,
                    help="future prediction")


if __name__ == '__main__':
    args = parser.parse_args()

    ######### setup output directory and logger ###########
    args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save+datetime.now().strftime('_%d_%m_%Y-%H:%M'), '')
    io_utils.makedirs(args.save)
    io_utils.makedirs(os.path.join(args.save, 'plots'))
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs'))
    logger.info('Results stored in {}'.format(args.save))

    ########## set global random seed ###########
    seed_everything(args.seed)

    ########### device #######
    #args.device = device
    logger.info('Running model on {}'.format(args.device))

    ########### data ############ 
    trainset, testset = load_data(args, plot=True)

    ########### model ###########
    odegpvae = build_model(args)
    odegpvae.to(args.device)
    odegpvae = initialize_and_fix_kernel_parameters(odegpvae, lengthscale_value=args.lengthscale, variance_value=args.variance, fix=False) #1.25, 0.5, 0.65 0.25

    logger.info('********** Model Built {} ODE **********'.format(args.order))
    logger.info('Model parameters: num features {} | num inducing {} | num epochs {} | lr {} | trace computation {}| kl_0 {} | order {} | D_in {} | D_out {} | beta {} | kernel {} | latent_dim {} | variance {} |lengthscale {}'.format(
                    args.num_features, args.num_inducing, args.Nepoch,args.lr, args.trace, args.kl_0, args.order, args.D_in, args.D_out, args.beta, args.kernel, args.q, args.variance, args.lengthscale))

    if args.continue_training:
        fname = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.model_path)
        odegpvae.load_state_dict(torch.load(fname,map_location=torch.device(args.device)))
        logger.info('Resume training for model {}'.format(fname))


    ########### log loss values ########
    elbo_meter = log_utils.CachedRunningAverageMeter(10)
    nll_meter = log_utils.CachedRunningAverageMeter(10)
    reg_kl_meter = log_utils.CachedRunningAverageMeter(10)
    inducing_kl_meter = log_utils.CachedRunningAverageMeter(10)
    logpL_meter = log_utils.CachedRunningAverageMeter(10)
    logztL_meter = log_utils.CachedRunningAverageMeter(10)
    mse_meter = log_utils.CachedAverageMeter()
    time_meter = log_utils.CachedAverageMeter()

    # ########### train ###########
    optimizer = torch.optim.Adam(odegpvae.parameters(),lr=args.lr)

    logger.info('********** Started Training **********')
    if args.kl_0: logger.info("Set KLs to 0")
    begin = time.time()
    global_itr = 0
    [N,T,nc,d,d] = next(iter(trainset)).shape
    for ep in range(args.Nepoch):
        L = 1 if ep<args.Nepoch//2 else 5 # increasing L as optimization proceeds is a good practice
        for itr,local_batch in enumerate(trainset):
            minibatch = local_batch.to(args.device) # B x T x 1 x 28 x 28 (batch, time, image dim)
            loss, nlhood, kl_reg, kl_u, logpL, log_pzt = compute_loss(odegpvae, minibatch, L, args)
            if torch.isnan(loss):
                logger.info('************** Obtained nan Loss at Epoch:{:4d}/{:4d}*************'.format(ep, args.Nepoch))
                logger.info('Laoding previous model for plotting')
                fname = os.path.join(args.save, 'odegpvae_mnist.pth')
                odegpvae = build_model(args)
                odegpvae.to(args.device)
                odegpvae.load_state_dict(torch.load(fname,map_location=torch.device(args.device)))
                odegpvae.eval()
                logger.info("Kernel lengthscales {}".format(odegpvae.flow.odefunc.diffeq.kern.lengthscales.data))
                logger.info("Kernel variance {}".format(odegpvae.flow.odefunc.diffeq.kern.variance.data))
                plot_results(odegpvae, trainset, testset, args, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter, logpL_meter, logztL_meter)
                sys.exit()

            if args.kl_0:
                kl_reg = kl_reg * 0.0
                kl_u = kl_u * 0.0
                loss = nlhood - kl_reg  - kl_u
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            #store values 
            elbo_meter.update(loss.item(), global_itr)
            nll_meter.update(nlhood.item(), global_itr)
            reg_kl_meter.update(kl_reg.item(), global_itr)
            inducing_kl_meter.update(kl_u.item(), global_itr)
            logpL_meter.update(logpL.item(), global_itr)
            logztL_meter.update(log_pzt.item(), global_itr)
            time_meter.update(time.time() - begin, global_itr)
            global_itr +=1

            if itr % args.log_freq == 0 :
                logger.info('Iter:{:<2d} | Time {} | elbo {:8.2f}({:8.2f}) | nlhood:{:8.2f}({:8.2f}) | kl_reg:{:<8.2f}({:<8.2f}) | kl_u:{:8.5f}({:8.5f})'.\
                    format(itr, timedelta(seconds=time_meter.val), 
                                elbo_meter.val, elbo_meter.avg,
                                nll_meter.val, nll_meter.avg,
                                reg_kl_meter.val, reg_kl_meter.avg,
                                inducing_kl_meter.val, inducing_kl_meter.avg)) 

        with torch.no_grad():
            mse_meter.reset()
            for itr_test,test_batch in enumerate(testset):
                test_batch = test_batch.to(args.device)
                Xrec_mu, test_mse = odegpvae(test_batch)
                plot_rot_mnist(test_batch, Xrec_mu.squeeze(0), False, fname=os.path.join(args.save, 'plots/rot_mnist.png'))
                torch.save(odegpvae.state_dict(), os.path.join(args.save, 'odegpvae_mnist.pth'))
                mse_meter.update(test_mse.item(),itr_test)
                break
        logger.info('Epoch:{:4d}/{:4d}| tr_elbo:{:8.2f}({:8.2f}) | test_mse:{:5.3f}({:5.3f})\n'.format(ep, args.Nepoch, elbo_meter.val, elbo_meter.avg, mse_meter.val, mse_meter.avg))

    logger.info('********** Optimization completed **********')
    logger.info("Kernel lengthscales {}".format(odegpvae.flow.odefunc.diffeq.kern.lengthscales.data))
    logger.info("Kernel variance {}".format(odegpvae.flow.odefunc.diffeq.kern.variance.data))

    plot_results(odegpvae, trainset, testset, args, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter, logpL_meter, logztL_meter)

