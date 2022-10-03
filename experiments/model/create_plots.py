
import torch
from model.misc.plot_utils import *


def plot_results(odegpvae, trainset, testset, args, elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter): #, logpL_meter, logztL_meter):
    #visualize latent dynamics with pca 
    with torch.no_grad():
        plot_latent_dynamics(odegpvae, next(iter(trainset)).to(args.device), args, fname=os.path.join(args.save, 'plots/dynamics_train'))
        plot_latent_dynamics(odegpvae, next(iter(testset)).to(args.device), args, fname=os.path.join(args.save, 'plots/dynamics_test'))

    #plot loss
    plot_trace(elbo_meter, nll_meter, reg_kl_meter, inducing_kl_meter, args) # logpL_meter, logztL_meter, args)

    #plot longer rollouts 
    with torch.no_grad():
        test_batch = next(iter(testset))[:3,:].to(args.device) #sample 3 images
        plot_data(test_batch, fname=os.path.join(args.save, 'plots/rollout_original.png'), size=3)
        Xrec_mu, test_mse = odegpvae(test_batch, args.Tlong*args.T)
        plot_rollout(Xrec_mu,fname=os.path.join(args.save, 'plots/rollout.png'))