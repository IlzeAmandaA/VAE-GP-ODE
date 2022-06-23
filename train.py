import argparse
from model.odegpvae import ODEGPVAE
from model.utils import plot_rot_mnist
from data.data import Dataset
from torch.utils import data
from scipy.io import loadmat
import torch
import os

def train(args: argparse):


    #check device available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #log results
    if not os.path.exists(os.path.join(args.ckpt_dir, args.task)):
        os.makedirs(os.path.join(args.ckpt_dir, args.task))
    if not os.path.exists(os.path.join('plots', args.task)):
        os.makedirs(os.path.join('plots', args.task))

    #load data
    X = loadmat(args.file)['X'].squeeze() # (N, 16, 784)
    Xtr   = torch.tensor(X[:args.N],dtype=torch.float32).view([args.N,args.T,1,28,28])
    Xtest = torch.tensor(X[args.N:],dtype=torch.float32).view([-1,args.T,1,28,28])
    # Generators
    trainset = Dataset(Xtr)
    trainset = data.DataLoader(trainset, batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
    testset  = Dataset(Xtest)
    testset  = data.DataLoader(testset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
    print('trainset loaded')
    print(next(iter(trainset)).shape)

    #load model 
    odegpvae = ODEGPVAE(q=8,n_filt=16).to(device)

    #set optimizer
    optimizer = torch.optim.Adam(odegpvae.parameters(),lr=args.lr)

    #training
    for ep in range(args.n_epoch):
        L = 1 if ep<args.n_epoch//2 else 5 # increasing L as optimization proceeds is a good practice
        for i,local_batch in enumerate(trainset):
            minibatch = local_batch.to(device)
            elbo, lhood, kl_z, kl_w = odegpvae(minibatch, len(trainset), L=L, inst_enc=True, method='rk4')[4:]
            tr_loss = -elbo
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
            print('Iter:{:<2d} lhood:{:8.2f}  kl_z:{:<8.2f}  kl_w:{:8.2f}'.\
                format(i, lhood.item(), kl_z.item(), kl_w.item()))
        with torch.set_grad_enabled(False):
            for test_batch in testset:
                test_batch = test_batch.to(device)
                Xrec_mu, test_mse = odegpvae.mean_rec(test_batch, method='rk4')
                plot_rot_mnist(test_batch, Xrec_mu, False, fname='rot_mnist.png')
                torch.save(odegpvae.state_dict(), 'odegpvae_mnist.pth')
                break
        print('Epoch:{:4d}/{:4d} tr_elbo:{:8.2f}  test_mse:{:5.3f}\n'.format(ep, args.n_epoch, tr_loss.item(), test_mse.item()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a VAE-ODE-GP')

    #Log 
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_dir',
                        help='checkpoint directory')

    #Data
    parser.add_argument('--data_root', type=str, default='data',
                        help='pass training data path')
    parser.add_argument('--task', type=str, default='mnist',
                        help='pass data type')
    parser.add_argument('--file', type=str, default='data/rot_mnist/rot-mnist-3s.mat',
                        help='pass filename')
    parser.add_argument('--N', type=int, default=500,
                        help='train/test split')
    parser.add_argument('--T', type=int, default=16,
                        help='time moments')
    parser.add_argument('--batch_size', type=int, default=25,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers')

    #Training
    parser.add_argument('--n_epoch', type=int, default=500,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')

    args = parser.parse_args()

    train(args)
