import os 
import argparse
import torch
from datetime import datetime
from data.wrappers import load_data
from model.create_model import build_model, compute_loss
from model.misc.plot_utils import plot_rot_mnist
from model.misc import io_utils
from model.misc.torch_utils import seed_everything
from model.misc.settings import settings
from model.core.initialization import initialize_and_fix_kernel_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]
parser = argparse.ArgumentParser('Learning human motion dynamics with GPODE')

# model parameters
parser.add_argument('--num_features', type=int, default=256,
                    help="Number of Fourier basis functions (for pathwise sampling from GP)")
parser.add_argument('--num_inducing', type=int, default=100,
                    help="Number of inducing points for the sparse GP")
parser.add_argument('--dimwise', type=eval, default=True,
                    help="Specify separate lengthscales for every output dimension")
parser.add_argument('--q_diag', type=eval, default=False,
                    help="Diagonal posterior approximation for inducing variables")
parser.add_argument('--num_latents', type=int, default=5,
                    help="Number of latent dimensions for training")

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

#vae arguments
parser.add_argument('--q', type=int, default=8,
                    help="Latent space dimensionality")
parser.add_argument('--n_filt', type=int, default=8,
                    help="Number of filters in the cnn")
parser.add_argument('--steps', type=int, default=5,
                    help="Number of timesteps used for encoding velocity")

# ode solver arguments
parser.add_argument('--solver', type=str, default='euler', choices=SOLVERS,
                    help="ODE solver for numerical integration")
parser.add_argument('--ts_dense_scale', type=int, default=2,
                    help="Factor for making a dense integration time grid (useful for explicit solvers)")
parser.add_argument('--use_adjoint', type=eval, default=False,
                    help="Use adjoint method for gradient computation")
parser.add_argument('--beta', type=int, default=1,
                    help="Factor to scale the inducing KL loss effect")

# training arguments
parser.add_argument('--Nepoch', type=int, default=500, #10_000
                    help="Number of gradient steps for model training")
parser.add_argument('--lr', type=float, default=0.005,
                    help="Learning rate for model training")
parser.add_argument('--eval_sample_size', type=int, default=128,
                    help="Number of posterior samples to evaluate the model predictive performance")
parser.add_argument('--save', type=str, default='results/mnist',
                    help="Directory name for saving all the model outputs")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--log_freq', type=int, default=20,
                    help="Logging frequency while training")
parser.add_argument('--device', type=str, default='cpu',
                    help="device")

                

if __name__ == '__main__':
    args = parser.parse_args()

    ######### setup output directory and logger ###########
    args.save = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.save+datetime.now().strftime('_%d_%m_%Y'), '')
    print('Results stored in', args.save)
    io_utils.makedirs(args.save)
    io_utils.makedirs(os.path.join(args.save, 'plots'))
    logger = io_utils.get_logger(logpath=os.path.join(args.save, 'logs'))

    ########## set global random seed ###########
    seed_everything(args.seed)

    ########### device #######
    args.device = device
    print('Running model on', args.device)

    ########### data ############ 
    trainset, testset = load_data(args, plot=True)

    ########### model ###########
    odegpvae = build_model(args)
    odegpvae.to(device)


    ########### initialize model #######
    #TODO how can we do this for the latent space? (can we do this even?)
    odegpvae = initialize_and_fix_kernel_parameters(odegpvae, lengthscale_value=1.25, variance_value=0.5, fix=False)
    #init of inducing variables (?) model = initialize_inducing(model, data_pca.trn.ys, data_pca.trn.ts.max(), 1e0)
    #init initial dist x0 (?)

    # ########### train ###########
    optimizer = torch.optim.Adam(odegpvae.parameters(),lr=args.lr)

    for ep in range(args.Nepoch):
        L = 1 if ep<args.Nepoch//2 else 5 # increasing L as optimization proceeds is a good practice
        for i,local_batch in enumerate(trainset):
            minibatch = local_batch.to(device) # B x T x 1 x 28 x 28 (batch, time, image dim)
            loss, lhood, kl_z, kl_u = compute_loss(odegpvae, minibatch, L, args.Ndata)
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            print('Iter:{:<2d} lhood:{:8.2f}  kl_z:{:<8.2f} kl_u:{:8.5f}'.\
                format(i, lhood.item(), kl_z.item(), kl_u.item())) 

        with torch.set_grad_enabled(False):
            for test_batch in testset:
                test_batch = test_batch.to(device)
                Xrec_mu, test_mse = odegpvae(test_batch)
                plot_rot_mnist(test_batch, Xrec_mu.squeeze(0), False, fname=os.path.join(args.save, 'plots/rot_mnist.png'))
                torch.save(odegpvae.state_dict(), os.path.join(args.save, 'odegpvae_mnist.pth'))
                break

        print('Epoch:{:4d}/{:4d} tr_elbo:{:8.2f}  test_mse:{:5.3f}\n'.format(ep, args.Nepoch, loss.item(), test_mse.item()))


