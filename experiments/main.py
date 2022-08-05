import numpy as np 
import torch
from data.wrappers import load_data
from model.odegpvae import ODEGPVAE
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on ', device)
data_root = 'data/'
task = 'mnist'
mask = True
value = 3 

# plotting
def plot_rot_mnist(X, Xrec, show=False, fname='rot_mnist.png'):
    N = min(X.shape[0],10)
    Xnp = X.detach().cpu().numpy()
    Xrecnp = Xrec.detach().cpu().numpy()
    T = X.shape[1]
    plt.figure(2,(T,3*N))
    for i in range(N):
        for t in range(T):
            plt.subplot(2*N,T,i*T*2+t+1)
            plt.imshow(np.reshape(Xnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
        for t in range(T):
            plt.subplot(2*N,T,i*T*2+t+T+1)
            plt.imshow(np.reshape(Xrecnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
    plt.savefig(fname)
    if show is False:
        plt.close()

if __name__ == '__main__':
    ########### data ############ 
    trainset, testset = load_data(data_root, task, mask, value, plot=True)


    ########### model ###########
    odegpvae = ODEGPVAE(q=8,n_filt=16, device=device).to(device)


    # ########### train ###########
    Nepoch = 500
    optimizer = torch.optim.Adam(odegpvae.parameters(),lr=1e-3)

    for ep in range(Nepoch):
        L = 1 if ep<Nepoch//2 else 5 # increasing L as optimization proceeds is a good practice
        for i,local_batch in enumerate(trainset):
            minibatch = local_batch.to(device) # B x T x 1 x 28 x 28 (batch, time, image dim)
            elbo, lhood, kl_z, kl_u = odegpvae(minibatch, len(trainset), L=L, inst_enc=False, method='euler')[4:]
            tr_loss = -elbo
            optimizer.zero_grad()
            tr_loss.backward(retain_graph=True) #TODO had to add this (?)  
            optimizer.step()
            print('Iter:{:<2d} lhood:{:8.2f}  kl_z:{:<8.2f} kl_u:{:8.5f}'.\
                format(i, lhood.item(), kl_z.item(), kl_u.item())) 
        with torch.set_grad_enabled(False):
            for test_batch in testset:
                test_batch = test_batch.to(device)
                Xrec_mu, test_mse = odegpvae.mean_rec(test_batch, method='euler')
                plot_rot_mnist(test_batch, Xrec_mu, False, fname='rot_mnist.png')
                torch.save(odegpvae.state_dict(), 'odegpvae_mnist.pth')
                break
        print('Epoch:{:4d}/{:4d} tr_elbo:{:8.2f}  test_mse:{:5.3f}\n'.format(ep, Nepoch, tr_loss.item(), test_mse.item()))


