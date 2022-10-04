import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_data(x, fname='plots/data.png', size=6):
    X = x.detach().cpu().numpy()
    plt.figure(1,(20,8))
    for j in range(size):
        for i in range(16):
            plt.subplot(7,20,j*20+i+1)
            plt.imshow(np.reshape(X[j,i,:],[28,28]), cmap='gray');
            plt.xticks([]); plt.yticks([])
    plt.savefig(fname)
    plt.close()


def plot_rollout(Xrec, show=False, fname='future.png'):
    [N, T, d, nc, nc] =Xrec.shape #3,48, 
    Xrecnp = Xrec.detach().cpu().numpy()
    plt.figure(1, (T,N))
    for i in range(N):
        for t in range(T):
            plt.subplot(N,T,i*T+t+1)
            plt.imshow(np.reshape(Xrecnp[i,t],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
    if show:
        plt.show()
    else:
        plt.savefig(fname) 
        plt.close()

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
    s0_mu, s0_logv = model.vae.encoder_s(data[:,0]) #N,q
    z0 = model.vae.encoder_s.sample(mu = s0_mu, logvar = s0_logv)
    v0_mu, v0_logv = None, None
    if model.order == 2:
        v0_mu, v0_logv = model.vae.encoder_v(torch.squeeze(data[:,0:model.v_steps]))
        v0 = model.vae.encoder_v.sample(mu= v0_mu, logvar = v0_logv)
        z0 = torch.concat([z0,v0],dim=1) #N, 2q
    zt = model.build_flow(z0, T)
    if args.order == 1:
        plot_latent_state(zt, show=False, fname=fname)
    elif args.order ==2:
        st_mu = zt[:,:,:args.q] # N,T,q
        vt_mu = zt[:,:,args.q:] # N,T,q
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
    plt.grid()
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

def plot_trace(elbo_meter, nll_meter,  z_kl_meter, inducing_kl_meter, args, make_plot=False): #logpL_meter, logztL_meter, 
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    axs[0][0].plot(elbo_meter.iters, elbo_meter.vals)
    axs[0][0].set_title("Loss (-elbo) function")
    axs[0][0].grid()
    axs[0][1].plot(nll_meter.iters, nll_meter.vals)
    axs[0][1].set_title("Observation NLL")
    axs[0][1].grid()
    axs[1][0].plot(z_kl_meter.iters, z_kl_meter.vals)
    axs[1][0].set_title("KL rec")
    axs[1][0].grid()
    axs[1][1].plot(inducing_kl_meter.iters,inducing_kl_meter.vals)
    axs[1][1].set_title("Inducing KL")
    axs[1][1].grid()
    # axs[2][0].plot(logpL_meter.iters, logpL_meter.vals)
    # axs[2][0].set_title("Loss log trace")
    # axs[2][0].grid()
    # axs[2][1].plot(logztL_meter.iters, logztL_meter.vals)
    # axs[2][1].set_title("Loss log p(z)")
    # axs[2][1].grid()

    fig.subplots_adjust()
    if make_plot:
        plt.show()
    else:
        fig.savefig(os.path.join(args.save, 'plots/optimization_trace.png'), dpi=160,
                    bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)


def plot_vae_embeddings(encoder, dataloader, n_samples, device, n_classes=16, output_path=None):
    """Visualize the embeddings in the latent space"""
    # classes = list(np.linspace(0, n_classes-1, n_classes).astype(np.str))
    n = 0
    codes, labels = [], []
    with torch.no_grad():
        for b_inputs, b_labels in dataloader:
            batch_size = b_inputs.size(0)
            b_codes = encoder(b_inputs.to(device))[0]
            #b_codes, b_labels = b_codes.cpu().data.numpy(), b_labels.cpu().data.numpy()
            if n + batch_size > n_samples:
                codes.append(b_codes[: n_samples - n])
                labels.append(b_labels[: n_samples - n])
                break
            else:
                codes.append(b_codes)
                labels.append(b_labels)
                n += batch_size
    #codes = np.vstack(codes)
    codes = torch.stack(codes)
    if codes.shape[1] > 2:
        #do PCA here 
        U,S,V = torch.pca_lowrank(codes)
        codes = codes@V[:,:2] 
        codes =  codes.reshape(N,T,2).cpu().numpy() # N,T,2
        #codes = TSNE().fit_transform(codes)
    labels = np.hstack(labels)

    fig, ax = plt.subplots(1)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height], which="both")
    color_map = plt.cm.get_cmap('hsv', n_classes)

    for iclass in range(min(labels), max(labels) + 1):
        ix = labels == iclass
        ax.plot(codes[ix, 0], codes[ix, 1], ".", c=color_map(iclass))

    # plt.legend(classes, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.suptitle("Latent embeddings of a sample datapoint using HSV colorcode", y=1)
    if output_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_path, "plots/latent-embeddings.png"))


def plot_vectorfield():

    pass