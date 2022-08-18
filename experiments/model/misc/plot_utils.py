import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import os

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

def plot_latent_dynamics(model, data, args, fname):
    [N,T,nc,d,d] = data.shape
    z0, logp0 = model.build_encoding(data)
    zt = model.build_flow(z0, logp0, T, trace=False)
    if args.order == 1:
        plot_latent_state(zt, show=False, fname=fname)
    elif args.order ==2:
        st_mu = zt[:,:,args.q:] # N,T,q
        vt_mu = zt[:,:,:args.q] # N,T,q
        plot_latent_state(st_mu, show=False, fname=fname)
        plot_latent_velocity(vt_mu, show=False, fname=fname)

def plot_latent_state(st_mu, show=False, fname='latent_dyanamics'):
    N,T,q = st_mu.shape
    st_mu = st_mu.detach() # N,T,q
    st_mu = st_mu.reshape(N*T,q) #NT,q
    U,S,V = torch.pca_lowrank(st_mu)
    st_pca = st_mu@V[:,:2] 
    st_pca =  st_pca.reshape(N,T,2).cpu().numpy() # N,T,2
    plt.figure(1,(5,5))
    for n in range(N):
        p, = plt.plot(st_pca[n,0,0],st_pca[n,0,1],'o',markersize=10)
        plt.plot(st_pca[n,:,0],st_pca[n,:,1],'-*', color=p.get_color())

    plt.xlabel('PCA-1',fontsize=15)
    plt.ylabel('PCA-2',fontsize=15)
    plt.title('Latent trajectories',fontsize=18)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(fname + '_state.png')
        plt.close()

def plot_latent_velocity(vt_mu, show=False, fname='latent_dyanamics'):
    N,T,q = vt_mu.shape
    vt_mu = vt_mu.detach() # N,T,q
    vt_mu = vt_mu.reshape(N*T,q) #NT,q
    U,S,V = torch.pca_lowrank(vt_mu)
    vt_pca = vt_mu@V[:,:2] 
    vt_pca =  vt_pca.reshape(N,T,2).cpu().numpy() # N,T,2
    ts = [t for t in range(T)]
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))
    fig.suptitle("Latent Dynamics", fontsize=18, y=0.95)
    for n in range(N):
        p1, = ax1.plot(ts[0], vt_pca[n,0,0],'o',markersize=10)
        p2, = ax2.plot(ts[0],vt_pca[n,0,1],'o',markersize=10)
        ax1.plot(ts, vt_pca[n,:,0],'-*', color=p1.get_color())
        ax2.plot(ts, vt_pca[n,:,1], '-*', color=p2.get_color())
        ax1.set_title('Velocity 1st component')
        ax2.set_title('Velocity 2nd component')

    ax1.grid()
    ax2.grid()
    
    if show:
        plt.show()
    else:
        plt.savefig(fname+'_velocity.png')
        plt.close()

def plot_trace(elbo_meter, nll_meter,  z_kl_meter, inducing_kl_meter, args, make_plot=False):
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    axs[0][0].plot(elbo_meter.iters, elbo_meter.vals)
    axs[0][0].set_title("Loss (-elbo) function")
    axs[0][0].grid()
    axs[0][1].plot(nll_meter.iters, nll_meter.vals)
    axs[0][1].set_title("Observation NLL")
    axs[0][1].grid()
    axs[1][0].plot(z_kl_meter.iters, z_kl_meter.vals)
    axs[1][0].set_title("KL z (ode - p(z))")
    axs[1][0].grid()
    axs[1][1].plot(inducing_kl_meter.iters,inducing_kl_meter.vals)
    axs[1][1].set_title("Inducing KL")
    axs[1][1].grid()

    fig.subplots_adjust()
    if make_plot:
        plt.show()
    else:
        fig.savefig(os.path.join(args.save, 'plots/optimization_trace.png'), dpi=160,
                    bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)


def plot_vectorfield():

    pass