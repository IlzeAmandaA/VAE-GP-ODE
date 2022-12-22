import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sklearn.manifold import TSNE

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
    [L, N, T, d, nc, nc] =Xrec.shape  
    Xrecnp = Xrec.squeeze(0).detach().cpu().numpy()
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

def plot_rot_mnist(X, Xrec, show=False, fname='rot_mnist.png', N=None):
    if N is None:
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
    if show:
        plt.show()
    else:
        plt.savefig(fname)
        plt.close()


def plot_rand_rot_mnist(X, Xrec, show=False, fname='rot_mnist.png', rows=4):
    N = min(X.shape[0],4)
    Xnp = X.detach().cpu().numpy() #B,1,nc,nc
    Xrecnp = Xrec.detach().cpu().numpy()
    plt.figure(2,(N,3*rows))
    idx_x = 0
    idx_rec = 0
    for r in range(rows):
        for i in range(N):
            plt.subplot(2*rows,N,r*N*2+i+1)
            plt.imshow(np.reshape(Xnp[idx_x],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
            idx_x += 1
        for i in range(N):
            plt.subplot(2*rows,N,r*N*2+i+N+1)
            plt.imshow(np.reshape(Xrecnp[idx_rec],[28,28]), cmap='gray')
            plt.xticks([]); plt.yticks([])
            idx_rec += 1
        idx_x += 1
        idx_rec += 1
        

    plt.savefig(fname)
    if show is False:
        plt.close()

def plot_latent_dynamics(model, data, args, fname):
    [N,T,nc,d,d] = data.shape
    s0_mu, s0_logv = model.vae.encoder(data[:,0]) #N,q
    z0 = model.vae.encoder.sample(mu = s0_mu, logvar = s0_logv)
    v0_mu, v0_logv = None, None
    if model.order == 2:
        v0_mu, v0_logv = model.vae.encoder_v(torch.squeeze(data[:,0:model.v_steps]))
        v0 = model.vae.encoder_v.sample(mu= v0_mu, logvar = v0_logv)
        z0 = torch.concat([z0,v0],dim=1) #N, 2q
    zt = model.sample_trajectories(z0,T).squeeze(0) # N,T,2q
    if args.ode == 1:
        plot_latent_state(zt, show=False, fname=fname)
    elif args.ode ==2:
        st_mu = zt[:,:,:args.latent_dim] # N,T,q
        vt_mu = zt[:,:,args.latent_dim:] # N,T,q
        plot_latent_state(st_mu, show=False, fname=fname)
        plot_latent_velocity(vt_mu, show=False, fname=fname)

def plot_latent_state(st_mu, show=False, fname='latent_dyanamics'):
    N,T,q = st_mu.shape
    st_mu = st_mu.detach() # N,T,q   
    if q>2:
        st_mu = st_mu.reshape(N*T,q) #NT,q
        U,S,V = torch.pca_lowrank(st_mu)
        st_pca = st_mu@V[:,:2] 
        st_pca =  st_pca.reshape(N,T,2).cpu().numpy() # N,T,2
        S = S / S.sum()
    else:
        st_pca = st_mu.cpu().numpy()
    plt.figure(1,(5,5))
    for n in range(N):
        p, = plt.plot(st_pca[n,0,0],st_pca[n,0,1],'o',markersize=10)
        plt.plot(st_pca[n,:,0],st_pca[n,:,1],'-*', color=p.get_color())
    if q>2:
        plt.xlabel('PCA-1  ({:.2f})'.format(S[0]),fontsize=15)
        plt.ylabel('PCA-2  ({:.2f})'.format(S[1]),fontsize=15)

    # plt.xlabel('PCA-1',fontsize=15)
    # plt.ylabel('PCA-2',fontsize=15)
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
    if q>2:
        vt_mu = vt_mu.reshape(N*T,q) #NT,q
        U,S,V = torch.pca_lowrank(vt_mu)
        vt_pca = vt_mu@V[:,:2] 
        vt_pca =  vt_pca.reshape(N,T,2).cpu().numpy() # N,T,2
    else:
        vt_pca = vt_mu.cpu().numpy()
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

def plot_params(hyperparam_meter, args, show=False):
    values = np.array(hyperparam_meter.vals)
    plt.figure(1,(12,6))
    for n in range(args.D_out):
        plt.plot(hyperparam_meter.iters,values[:,n])

    plt.ylabel('hyperparameter value',fontsize=15)
    plt.xlabel('iterations',fontsize=15)
    plt.title('Variance updates',fontsize=18)
    plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(args.save, 'plots/hyperparam_trace.png'))
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
    axs[1][0].set_title("KL rec")
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
        np.save(os.path.join(args.save, 'elbo.npy'), np.stack((elbo_meter.iters, elbo_meter.vals), axis=1))
        np.save(os.path.join(args.save, 'nll.npy'), np.stack((nll_meter.iters, nll_meter.vals), axis=1))
        np.save(os.path.join(args.save, 'zkl.npy'), np.stack((z_kl_meter.iters, z_kl_meter.vals), axis=1))
        np.save(os.path.join(args.save, 'inducingkl.npy'), np.stack((inducing_kl_meter.iters,inducing_kl_meter.vals), axis=1))

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
    codes = torch.cat(codes) # n_samples, q
    labels = torch.cat(labels) # n_samples
    labels = labels.cpu().numpy()
    if codes.shape[1] > 2:
        #do PCA here 
        U,S,V = torch.pca_lowrank(codes)
        codes = codes@V[:,:2] # n_samples, 2
        codes = codes.cpu().numpy()
        #codes = TSNE().fit_transform(codes)
   # labels = np.hstack(labels)
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
        plt.savefig(os.path.join(output_path, "plots/latent-embeddings_pca.png"))


def plot_trace_vae(elbo_meter, nll_meter,  z_kl_meter, args, make_plot=False): 
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))

    axs[0].plot(elbo_meter.iters, elbo_meter.vals)
    axs[0].set_title("Loss (-elbo) function")
    axs[0].grid()
    axs[1].plot(nll_meter.iters, nll_meter.vals)
    axs[1].set_title("Observation NLL")
    axs[1].grid()
    axs[2].plot(z_kl_meter.iters, z_kl_meter.vals)
    axs[2].set_title("KL rec")
    axs[2].grid()

    fig.subplots_adjust()
    if make_plot:
        plt.show()
    else:
        fig.savefig(os.path.join(args.output_path, 'plots/optimization_trace.png'), dpi=160,
                    bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)

def visualize_output(vae, x, output_path=None, logger=None):
    y = vae.test(x)
   # plot_rot_mnist(x, y, show=False, fname=os.path.join(output_path, "rot-mnist.png"))
    plot_rand_rot_mnist(x,y,show=False, fname=os.path.join(output_path, "rot-mnist.png"), rows=3)
    mse = torch.mean((y-x)**2)
    std = torch.std((y-x)**2)
    logger.info('MSE {} (std {})'.format(mse.item(), std.item()))
    y = y.cpu().detach().numpy().reshape(16, 28, 28)   

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for ax, img in zip(axs.flat, x.cpu()):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.axis('off')

    plt.suptitle("Original image", y=1)
    plt.tight_layout()
    if output_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_path, "vae-original.png"))

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for ax, img in zip(axs.flat, y):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.axis('off')
    plt.suptitle("Predicted image", y=1)
    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_path, "vae-prediction.png"))

def visualize_embeddings(encoder, dataloader, n_samples, device, n_classes=16, output_path=None):
    """Visualize the embeddings in the latent space"""
    # classes = list(np.linspace(0, n_classes-1, n_classes).astype(np.str))
    n = 0
    codes, labels = [], []
    with torch.no_grad():
        for b_inputs, b_labels in dataloader:
            batch_size = b_inputs.size(0)
            b_codes = encoder(b_inputs.to(device))[0]
            b_codes, b_labels = b_codes.cpu().data.numpy(), b_labels.cpu().data.numpy()
            if n + batch_size > n_samples:
                codes.append(b_codes[: n_samples - n])
                labels.append(b_labels[: n_samples - n])
                break
            else:
                codes.append(b_codes)
                labels.append(b_labels)
                n += batch_size
    codes = np.vstack(codes)
    if codes.shape[1] > 2:
        codes = TSNE().fit_transform(codes)
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
        plt.savefig(os.path.join(output_path, "latent-embeddings.png"))