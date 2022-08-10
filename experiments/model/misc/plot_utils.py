import matplotlib.pyplot as plt
import numpy as np

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

def plot_latent_dynamics(zt, N, pca, show=False, fname='latent_dyanamics.png'):
    q=8
    st_mu = zt[:,:,q:] # N,T,q
    vt_mu = zt[:,:,:q] # N,T,q
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))
    fig.suptitle("Latent Dynamics", fontsize=18, y=0.95)
    for n in range(N):
        st_pca = pca.fit_transform(st_mu[n]) # T,2
        vt_pca = pca.fit_transform(vt_mu[n])

        ax1.plot(st_pca[:,0], st_pca[:,1],lw=3)
        ax1.scatter(st_pca[:,0], st_pca[:,1], s = 25, zorder=2.5)
        ax1.set_title('Latent state pca')
        ax2.plot(vt_pca[:,0], vt_pca[:,1], lw=3)
        ax2.scatter(vt_pca[:,0], vt_pca[:,1], s = 25, zorder=2.5)
        ax2.set_title('Latent velocity pca')

    ax1.grid()
    ax2.grid()
    plt.savefig(fname)
    if show is False:
        plt.close()