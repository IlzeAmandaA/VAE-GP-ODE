import os 
import time
from datetime import timedelta
import argparse
import torch
from datetime import datetime
from data.mnist import load_rotating_mnist_data, create_rotating_dataset
from model.core.vae import VAE

from model.misc.plot_utils import *
from model.misc import io_utils
from model.misc import log_utils 
from model.misc.torch_utils import seed_everything


from torch.distributions import kl_divergence as kl

parser = argparse.ArgumentParser('Learning Latent Encoding with VAE')

#data arguments
parser.add_argument('--digit', type=int, default=3,
                    help="For which digit to create the training data")
parser.add_argument('--n_angle', type=int, default=16,
                    help="Data set time steps of a full rotation")
parser.add_argument('--n_train', type=int, default=180,
                    help="Directory name for saving all the model outputs")
parser.add_argument('--n_test', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--batch', type=int, default=64,
                    help="batch size")

#vae arguments
parser.add_argument('--latent_dim', type=int, default=6,
                    help="Learning rate for model training")

# training arguments
parser.add_argument('--device', type=str, default='cuda:0',
                    help="device")
parser.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate for model training")
parser.add_argument('--seed', type=int, default=121,
                    help="Global seed for the training run")
parser.add_argument('--vae_epochs', type=int, default=300, 
                    help="Number of gradient steps for model training")

# misc arguments
parser.add_argument('--output_path', type=str, default='results/vae',
                    help="Directory name for saving all the model outputs")
parser.add_argument('--save', type=str, default='data/moving_mnist',
                    help="Directory name for saving all the model outputs")
parser.add_argument('--log_freq', type=int, default=20,
                    help="Logging frequency while training")



def vae_train(args, rotating_mnist_train_dataset, epochs, output_model_path):

    # ########### log loss values ########
    elbo_meter = log_utils.CachedRunningAverageMeter(10)
    nll_meter = log_utils.CachedRunningAverageMeter(10)
    reg_kl_meter = log_utils.CachedRunningAverageMeter(10)
    time_meter = log_utils.CachedAverageMeter()

    # Load data
    train_loader = load_rotating_mnist_data(rotating_mnist_train_dataset,args)

    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)

    vae = VAE(device=args.device, latent_dim=args.latent_dim)

    encoder_path = os.path.join(output_model_path, "encoder.pt")
    decoder_path = os.path.join(output_model_path, "decoder.pt")

    vae.print_summary()


    optim = torch.optim.Adam(
            list(vae.encoder.parameters()) + list(vae.decoder.parameters()), lr=args.lr
        )

    print("------------------------------------------------------------------------------------")
    print("VAE Train")
    print("------------------------------------------------------------------------------------")
    begin = time.time()
    global_itr = 0
    for ep in range(epochs):
        running_loss = []
        vae.encoder.train()
        vae.decoder.train()
        for itr,(x, _) in enumerate(train_loader):
            optim.zero_grad()
            x = x.to(args.device)
            enc_mean, enc_logvar = vae.encoder(x)
            z = vae.encoder.sample(enc_mean, enc_logvar)

            q = vae.encoder.q_dist(enc_mean, enc_logvar)
            kl_reg = kl(q,vae.prior).sum(-1).mean(0)
            
            y = vae.decoder(z)
            lhood = vae.decoder.log_prob(x,y, pretrain=True).sum([1,2,3]).mean(0) #L,N,T,d,nc,nc
            loss_val = kl_reg - lhood

            loss_val.backward()
            optim.step()
            running_loss.append(loss_val.item())

            #store values 
            elbo_meter.update(loss_val.item(), global_itr)
            nll_meter.update(-lhood.item(), global_itr)
            reg_kl_meter.update(kl_reg.item(), global_itr)
            time_meter.update(time.time() - begin, global_itr)
            global_itr +=1

            if itr % args.log_freq == 0 :
                logger.info('Iter:{:<2d} | Time {} | elbo {:8.2f}({:8.2f}) | nlhood:{:8.2f}({:8.2f}) | kl_reg:{:<8.2f}({:<8.2f})'.\
                    format(itr, timedelta(seconds=time_meter.val), 
                                elbo_meter.val, elbo_meter.avg,
                                nll_meter.val, nll_meter.avg,
                                reg_kl_meter.val, reg_kl_meter.avg)) 


        logger.info('Epoch:{:4d}/{:4d}| tr_elbo:{:8.2f}({:8.2f}))\n'.format(ep, args.vae_epochs, elbo_meter.val, elbo_meter.avg))


    print("------------------------------------------------------------------------------------")
    vae.save(encoder_path, decoder_path)

    return vae, elbo_meter, nll_meter, reg_kl_meter

if __name__ == '__main__':
    args = parser.parse_args()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train VAE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ######### setup output directory and logger ###########
    args.output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_path+datetime.now().strftime('_%d_%m_%Y-%H:%M'), '')
    io_utils.makedirs(args.output_path)
    io_utils.makedirs(os.path.join(args.output_path, 'plots'))
    logger = io_utils.get_logger(logpath=os.path.join(args.output_path, 'logs'))
    logger.info('Results stored in {}'.format(args.output_path))

    ########## set global random seed ###########
    seed_everything(args.seed)

    ########### device #######
    #args.device = device
    logger.info('Running model on {}'.format(args.device))
    logger.info('Model parameters:  num epochs {} | lr {} | latent_dim {} | n_angles {}'.format(
                     args.vae_epochs,args.lr, args.latent_dim, args.n_angle))


    ########### data ############ 
    rotating_mnist_train_dataset = os.path.join(args.save, "rotating_mnist_train_3_" + str(args.n_angle)+"_angles.npy")
    rotating_mnist_test_dataset = os.path.join(args.save, "rotating_mnist_test_3_" + str(args.n_angle)+"_angles.npy")

    if os.path.exists(rotating_mnist_train_dataset) and os.path.exists(rotating_mnist_test_dataset):
        train_rotated_imgs = np.load(rotating_mnist_train_dataset)
        test_rotated_imgs = np.load(rotating_mnist_test_dataset)
    else:
        train_rotated_imgs, test_rotated_imgs = create_rotating_dataset(args.save, digit=args.digit, train_n=args.n_train,
                                                                        test_n=args.n_test, n_angles=args.n_angle)
        np.save(rotating_mnist_train_dataset, train_rotated_imgs)
        np.save(rotating_mnist_test_dataset, test_rotated_imgs)

    # visualize
    sample_img = train_rotated_imgs[1].reshape(-1, 28, 28)
    _, axs = plt.subplots(1, args.n_angle, figsize=(120, 5))
    for i in range(args.n_angle):
        axs[i].imshow(sample_img[i].reshape((28, 28)), cmap="gray")
        axs[i].axis('off')
        axs[i].set_title(f"t={i}")
    plt.suptitle("Dataset Sample", fontsize=24)
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig(os.path.join(args.output_path, "sample-dataset.png"))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train VAE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    vae_model_path = os.path.join(args.output_path, "MNIST-VAE")
    vae, elbo_meter, nll_meter, reg_kl_meter = vae_train(args, rotating_mnist_train_dataset, epochs=args.vae_epochs, output_model_path=vae_model_path)

    test_loader = load_rotating_mnist_data(rotating_mnist_test_dataset, args)

    x = next(iter(test_loader))[0].to(args.device)
    # only first 16 images
    visualize_output(
        vae, x[:16], args.output_path, logger
    )

    visualize_embeddings(
        vae.get_encoder(), test_loader, 1000, args.device, n_classes=args.n_angle, output_path=args.output_path
    )

    plot_vae_embeddings(vae.encoder, test_loader, 1000, args.device, n_classes=args.n_angle, output_path=args.output_path)

    plot_trace_vae(elbo_meter, nll_meter, reg_kl_meter, args) 