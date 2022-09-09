import scipy.io as sio
import numpy as np
import torch
from torch.utils import data
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .utils import Dataset

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
	T = args.T #16
	Xtr   = torch.tensor(X[:N],dtype=torch.float32).view([N,T,1,28,28])
	Xtest = torch.tensor(X[N:],dtype=torch.float32).view([-1,T,1,28,28])

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