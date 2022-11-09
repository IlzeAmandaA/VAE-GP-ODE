from .mnist import load_mnist_data

def load_data(args, plot=False):
	if args.task=='mnist':
		trainset, testset = load_mnist_data(args, plot = plot)
	return trainset, testset #, N, T, D