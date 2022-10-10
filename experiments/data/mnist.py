import scipy.io as sio
import numpy as np
import torch
from torch.utils import data
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .utils import Dataset, Dataset_labels

import torchvision
import torchvision.transforms as transforms
from scipy.ndimage import rotate

def rot_start(Xtr, T, N ):
	start_angle = np.random.randint(0,T,N)
	X_angle = []
	for n in range(N):
		start = Xtr[n,start_angle[n]:,:,:,:]
		end = torch.flip(Xtr[n,1:start_angle[n]+1,:,:,:],dims=(1,))
		new_X = torch.cat((start,end), dim=0).unsqueeze(0)
		X_angle.append(new_X)
	return torch.cat(X_angle,dim=0)


def load_mnist_data(args, plot=True):
	fullname = os.path.join(args.data_root, "rot_mnist", "rot-mnist.mat")
	dataset = sio.loadmat(fullname)
	
	X = np.squeeze(dataset['X'])
	if args.mask:
		Y = np.squeeze(dataset['Y'])
		X = X[Y==args.value,:,:]

	N = args.Ndata
	Nt = args.Ntest + N
	T = args.T #16
	Xtr   = torch.tensor(X[:N],dtype=torch.float32).view([N,T,1,28,28])
	Xtest = torch.tensor(X[N:Nt],dtype=torch.float32).view([-1,T,1,28,28])

	if args.rotrand:
		Xtr = rot_start(Xtr,T,N)
		Xtest = rot_start(Xtest, T, Xtest.shape[0])

	# Generators
	params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': 2} #25
	trainset = Dataset(Xtr)
	trainset = data.DataLoader(trainset, **params)
	testset  = Dataset(Xtest)
	testset  = data.DataLoader(testset, **params)

	if plot:
		x = next(iter(trainset))
		plt.figure(1,(20,8))
		for j in range(6):
			for i in range(16):
				plt.subplot(7,20,j*20+i+1)
				plt.imshow(np.reshape(x[j,i,:],[28,28]), cmap='gray');
				plt.xticks([]); plt.yticks([])
		plt.savefig(os.path.join(args.save, 'plots/data.png'))
		plt.close()
	return trainset, testset


def load_mat_mnist_data(args, plot=True):
	fullname = os.path.join(args.data_root, "rot_mnist", "rot-mnist.mat")
	dataset = sio.loadmat(fullname)
	
	X = np.squeeze(dataset['X'])
	if args.mask:
		Y = np.squeeze(dataset['Y'])
		X = X[Y==args.value,:,:]

	N = args.Ndata
	Nt = args.Ntest + N
	T = args.T #16
	Xtr   = torch.tensor(X[:N],dtype=torch.float32).reshape([N*T,1,28,28]) #Ndata*T, 1, nc, nc
	Xtest = torch.tensor(X[N:Nt],dtype=torch.float32).reshape([args.Ntest*T,1,28,28]) #Ntest*T, 1, nc, nc

	t = np.linspace(0, T - 1, T).astype(np.uint8).reshape((1, -1)) # 1, T
	tr_t = np.repeat(t, Xtr.shape[0] // T, axis=0).reshape((-1, 1)) #Ndata*T,1
	te_t = np.repeat(t, Xtest.shape[0] // T, axis=0).reshape((-1, 1)) #Ntest*T,1

	# Generators
	params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': 2} #20
	trainset = Dataset_labels(Xtr, tr_t)
	trainset = data.DataLoader(trainset, **params)
	testset  = Dataset_labels(Xtest, te_t)
	testset  = data.DataLoader(testset, **params)


	if plot:
		x, _ = next(iter(trainset))[:16]
		fig, axs = plt.subplots(4, 4, figsize=(8, 8))
		for ax, img in zip(axs.flat, x.cpu()):
			ax.imshow(img.reshape(28, 28), cmap="gray")
			ax.axis('off')
		plt.savefig(os.path.join(args.save, 'plots/data.png'))
		plt.close()
		print('labels', _)

	return trainset, testset

def load_rotating_mnist_data(data_path, args, plot=True):
	x_true = np.load(data_path).reshape((-1, 1, 28, 28))
	t = np.linspace(0, args.n_angle - 1, args.n_angle).astype(np.uint8).reshape((1, -1))
	tr_t = np.repeat(t, x_true.shape[0] //args.n_angle, axis=0).reshape((-1, 1))
	tr_dataset = Dataset_labels(x_true, tr_t)
	data_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.batch, shuffle=True)
	if plot:
		sample_img = x_true[0:args.n_angle].reshape(-1, 28, 28)
		_, axs = plt.subplots(1, args.n_angle, figsize=(120, 5))
		for i in range(args.n_angle):
			axs[i].imshow(sample_img[i].reshape((28, 28)), cmap="gray")
			axs[i].axis('off')
			axs[i].set_title(f"t={i}")
		plt.suptitle("Dataset Sample", fontsize=24)
		plt.tight_layout(h_pad=0, w_pad=0)
		plt.savefig(os.path.join(args.save, "sample-dataset.png"))
	return data_loader


def rotate_img(img, angles: list):
    """ Rotate the input MNIST image in angles specified """
    rotated_imgs = np.array(img).reshape((-1, 1, 28, 28))
    for a in angles:
        rotated_imgs = np.concatenate(
            (
                rotated_imgs,
                rotate(img, a, axes=(1, 2), reshape=False).reshape((-1, 1, 28, 28)),
            ),
            axis=1,
        )
    return rotated_imgs

def create_rotating_dataset(data_path, digit=3, train_n=100, test_n=10, n_angles=64):
    """
        Takes the MNIST data path as input and returns the rotating data by rotating the digit uniformly
        for n_angles angles.
    """
    mnist_train = torchvision.datasets.mnist.MNIST(
        data_path, download=True, transform=transforms.ToTensor()
    )
    mnist_test = torchvision.datasets.mnist.MNIST(
        data_path, download=True, train=False, transform=transforms.ToTensor()
    )

    angles = np.linspace(0, 2 * np.pi, n_angles)[1:]
    angles = np.rad2deg(angles)

    train_digit_idx = torch.where(mnist_train.train_labels == digit)
    train_digit_imgs = mnist_train.train_data[train_digit_idx]
    random_idx = np.random.randint(0, train_digit_imgs.shape[0], train_n)
    train_digit_imgs = train_digit_imgs[random_idx]
    train_rotated_imgs = rotate_img(train_digit_imgs, angles)
    train_rotated_imgs = train_rotated_imgs / 255
    train_rotated_imgs = train_rotated_imgs.astype(np.float32)

    test_digit_idx = torch.where(mnist_test.test_labels == digit)
    test_digit_imgs = mnist_test.train_data[test_digit_idx]
    random_idx = np.random.randint(0, test_digit_imgs.shape[0], test_n)
    test_digit_imgs = test_digit_imgs[random_idx]
    test_rotated_imgs = rotate_img(test_digit_imgs, angles)
    test_rotated_imgs = test_rotated_imgs / 255

    test_rotated_imgs = test_rotated_imgs.astype(np.float32)

    return train_rotated_imgs, test_rotated_imgs