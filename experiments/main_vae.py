from cgi import test
import os 
import time
from datetime import timedelta
import argparse
import zipapp
import torch
import torch.nn as nn
import sys
from datetime import datetime
from data.mnist import load_rotating_mnist_data, create_rotating_dataset
from model.core.vae import VAE

from model.misc.plot_utils import *
from model.misc import io_utils
from model.misc import log_utils 
from model.misc.torch_utils import seed_everything
from model.misc.settings import settings


from torch.distributions import kl_divergence as kl

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]
KERNELS = ['RBF', 'DF']
parser = argparse.ArgumentParser('Learning Latent Encoding with VAE')

# model parameters
parser.add_argument('--weight', type=int, default=1,
                    help="lhood weight")
parser.add_argument('--order', type=int, default=1,
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
parser.add_argument('--batch', type=int, default=64,
                    help="batch size")
parser.add_argument('--T', type=int, default=16,
                    help="Number of time points")
parser.add_argument('--Ndata', type=int, default=360,
                    help="Number training data points")
parser.add_argument('--Ntest', type=int, default=40,
                    help="Number valid data points")
parser.add_argument('--rotrand', type=eval, default=False,
                    help="if True multiple initial rotatio angles")

#vae arguments
parser.add_argument('--q', type=int, default=16,
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
parser.add_argument('--Nepoch', type=int, default=300, #10_000
                    help="Number of gradient steps for model training")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate for model training")
parser.add_argument('--eval_sample_size', type=int, default=128,
                    help="Number of posterior samples to evaluate the model predictive performance")
parser.add_argument('--save', type=str, default='results/vae',
                    help="Directory name for saving all the model outputs")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--log_freq', type=int, default=20,
                    help="Logging frequency while training")
parser.add_argument('--device', type=str, default='cuda:0',
                    help="device")

parser.add_argument('--output_path', type=str, default='experiments/data/moving_mnist',
                    help="device")
parser.add_argument('--n_angle', type=int, default=64,
                    help="device")

#plotting arguments
parser.add_argument('--Tlong', type=int, default=3,
                    help="future prediction")

def kl_divergence(mean, logvar):
    """
    KL Divergence value between the input distribution specified with mean and logvar and N(0,I) considering
    diagonal covariance.
    """
    var = torch.exp(logvar)
    mean2 = mean * mean
    loss = -0.5 * torch.mean(1 + logvar - mean2 - var)
    return loss

if __name__ == '__main__':
    args = parser.parse_args()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train VAE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    logger.info('Model parameters:  num epochs {} | lr {} | latent_dim {} |'.format(
                     args.Nepoch,args.lr, args.q))


    ########### data ############ 
    rotating_mnist_train_dataset = os.path.join(args.output_path, "rotating_mnist_train_3_64_angles.npy")
    rotating_mnist_test_dataset = os.path.join(args.output_path, "rotating_mnist_test_3_64_angles.npy")

    if os.path.exists(rotating_mnist_train_dataset) and os.path.exists(rotating_mnist_test_dataset):
        train_rotated_imgs = np.load(rotating_mnist_train_dataset)
        test_rotated_imgs = np.load(rotating_mnist_test_dataset)
    else:
        train_rotated_imgs, test_rotated_imgs = create_rotating_dataset(args.output_path, digit=args.value, train_n=args.Ndata,
                                                                        test_n=args.Ntest, n_angles=args.n_angle)
        np.save(rotating_mnist_train_dataset, train_rotated_imgs)
        np.save(rotating_mnist_test_dataset, test_rotated_imgs)


    trainset = load_rotating_mnist_data(rotating_mnist_train_dataset, args)
    testset = load_rotating_mnist_data(rotating_mnist_test_dataset, args, plot=False)
    vae_model_path = os.path.join(args.save, "MNIST-VAE")

    ########### model ###########
    vae = VAE(steps = args.steps, n_filt=args.n_filt, q=args.q, D_in=args.q, order= args.order, device=args.device, distribution='bernoulli')
    vae.to(args.device)

    ########### log loss values ########
    elbo_meter = log_utils.CachedRunningAverageMeter(10)
    nll_meter = log_utils.CachedRunningAverageMeter(10)
    reg_kl_meter = log_utils.CachedRunningAverageMeter(10)
    mse_meter = log_utils.CachedAverageMeter()
    time_meter = log_utils.CachedAverageMeter()

    #train
    optimizer = torch.optim.Adam(list(vae.encoder_s.parameters()) + list(vae.decoder.parameters()), lr=args.lr)
    logger.info('********** Started Training **********')
    begin = time.time()
    global_itr = 0
    bce = nn.BCELoss(reduction="mean")
    for ep in range(args.Nepoch):
        running_loss = []
        vae.train()
        for itr,(local_batch, _) in enumerate(trainset):
            x = local_batch.to(args.device) # B x d x nc x  nc (batch, image dim)
            enc_mean, enc_logvar = vae.encoder_s(x)
            z = vae.encoder_s.sample(enc_mean, enc_logvar) #B,q
            #Xrec = vae.decoder(z[None,:,None,:])  #B, d, nc, nc
            Xrec = vae.decoder(z)
            #Reconstruction log-likelihood
            # lhood = vae.decoder.log_prob(x,Xrec,1) # L,T,N,d,nc,nc
            # lhood = lhood.sum([0,1,3,4,5]) #N
            
            lhood = bce(Xrec,x)
            #lhood = lhood.mean([0,1,3,4,5]) #N

            # KL reg
            # q = vae.encoder_s.q_dist(enc_mean, enc_logvar)
            # kl_reg = kl(q, vae.prior).sum(-1) #N
            kl_reg = kl_divergence(enc_mean, enc_logvar)


            #lhood = lhood.mean()
            #kl_reg = kl_reg.mean()

            #loss = - (lhood - kl_reg)
            #loss = lhood*args.Ndata + kl_reg*args.Ndata
            loss = lhood*1000 + kl_reg

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            #store values 
            elbo_meter.update(loss.item(), global_itr)
            #nll_meter.update(-lhood.item(), global_itr)
            nll_meter.update(lhood.item(), global_itr)
            reg_kl_meter.update(kl_reg.item(), global_itr)
            time_meter.update(time.time() - begin, global_itr)
            global_itr +=1

            if itr % args.log_freq == 0 :
                logger.info('Iter:{:<2d} | Time {} | elbo {:8.2f}({:8.2f}) | nlhood:{:8.2f}({:8.2f}) | kl_reg:{:<8.2f}({:<8.2f})'.\
                    format(itr, timedelta(seconds=time_meter.val), 
                                elbo_meter.val, elbo_meter.avg,
                                nll_meter.val, nll_meter.avg,
                                reg_kl_meter.val, reg_kl_meter.avg)) 
        
        with torch.no_grad():
            vae.eval()
            mse_meter.reset()
            for itr_test,(test_batch, _) in enumerate(testset):
                test_batch = test_batch.to(args.device) #B,1,nc,nc
                enc_mean, enc_logvar = vae.encoder_s(x)
                z = vae.encoder_s.sample(enc_mean, enc_logvar)
                #Xrec = vae.decoder(z[None,:,None,:]) #B,1,nc,nc
                Xrec = vae.decoder(z)
                test_mse = torch.mean((Xrec-test_batch)**2)
                plot_rand_rot_mnist(test_batch, Xrec, False, fname=os.path.join(args.save, 'plots/rot_mnist.png'))
                torch.save(vae.state_dict(), os.path.join(args.save, 'vae_mnist.pth'))
                mse_meter.update(test_mse.item(),itr_test)
                break
        logger.info('Epoch:{:4d}/{:4d}| tr_elbo:{:8.2f}({:8.2f}) | test_mse:{:5.3f}({:5.3f})\n'.format(ep, args.Nepoch, elbo_meter.val, elbo_meter.avg, mse_meter.val, mse_meter.avg))

    logger.info('********** Training completed **********')

    plot_vae_embeddings(vae.encoder_s, testset, 100, args.device, n_classes=args.T, output_path=args.save)
    plot_trace_vae(elbo_meter, nll_meter, reg_kl_meter, args) # logpL_meter, logztL_meter, args)
    print('created plots')


