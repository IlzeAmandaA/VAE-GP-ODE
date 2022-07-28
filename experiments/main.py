import numpy as np 
import torch
from data.wrappers import load_data
from model.odegpvae import ODEGPVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_root = 'data/'
task = 'mnist'

########### data ############ 
trainset, testset = load_data(data_root, task, plot=True)


########### model ###########
odegpvae = ODEGPVAE(q=8,n_filt=16).to(device)


# ########### train ###########
Nepoch = 500
optimizer = torch.optim.Adam(odegpvae.parameters(),lr=1e-3)

for ep in range(Nepoch):
    L = 1 if ep<Nepoch//2 else 5 # increasing L as optimization proceeds is a good practice
    for i,local_batch in enumerate(trainset):
        minibatch = local_batch.to(device) # B x T x 1 x 28 x 28 (batch, time, image dim)
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
            Xrec_mu, test_mse = ode2vae.mean_rec(test_batch, method='rk4')
            plot_rot_mnist(test_batch, Xrec_mu, False, fname='rot_mnist.png')
            torch.save(ode2vae.state_dict(), 'ode2vae_mnist.pth')
            break
    print('Epoch:{:4d}/{:4d} tr_elbo:{:8.2f}  test_mse:{:5.3f}\n'.format(ep, Nepoch, tr_loss.item(), test_mse.item()))


