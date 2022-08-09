import scipy.io as sio
import numpy as np
import torch
from torch.utils import data
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .utils import Dataset



def load_mnist_data(args, plot=True):
	fullname = os.path.join(args.data_root, "rot_mnist", "rot-mnist.mat")
	dataset = sio.loadmat(fullname)
	
	X = np.squeeze(dataset['X'])
	if args.mask:
		Y = np.squeeze(dataset['Y'])
		X = X[Y==args.value,:,:]

	N = 500
	T = args.T #16
	Xtr   = torch.tensor(X[:N],dtype=torch.float32).view([N,T,1,28,28])
	Xtest = torch.tensor(X[N:],dtype=torch.float32).view([-1,T,1,28,28])

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

def plot_mnist_recs(X,Xrec,idxs=[0,1,2,3,4],show=False,fname='reconstructions.png'):
	if X.shape[0]<np.max(idxs):
		idxs = np.arange(0,X.shape[0])
	tt = X.shape[1]
	plt.figure(2,(tt,3*len(idxs)))
	for j, idx in enumerate(idxs):
		for i in range(tt):
			plt.subplot(2*len(idxs),tt,j*tt*2+i+1)
			plt.imshow(np.reshape(X[idx,i,:],[28,28]), cmap='gray');
			plt.xticks([]); plt.yticks([])
		for i in range(tt):
			plt.subplot(2*len(idxs),tt,j*tt*2+i+tt+1)
			plt.imshow(np.reshape(Xrec[idx,i,:],[28,28]), cmap='gray');
			plt.xticks([]); plt.yticks([])
	plt.savefig(fname)
	if show is False:
		plt.close()
