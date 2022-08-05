from .mnist import load_mnist_data
# from .mnist_nonuniform import *
# from .mocap_single import *
# from .mocap_many import *
# from .bballs import *


def load_data(data_dir,task,dt=0.1,subject_id=0,plot=False):
	if task=='mnist':
		trainset, testset = load_mnist_data(data_dir,dt=dt,plot=plot)
	# if task=='mnist_nonuniform':
	# 	dataset = load_mnist_nonuniform_data(data_dir,dt=dt,plot=plot)
	# elif task=='mocap_many':
	# 	dataset = load_mocap_data_many_walks(data_dir,dt=dt,plot=plot)
	# elif task=='mocap_single':
	# 	dataset = load_mocap_data_single_walk(data_dir,subject_id=subject_id,dt=dt,plot=plot)
	# elif task=='bballs':
	# 	dataset = load_bball_data(data_dir,dt=dt,plot=plot)
	#[N,T,D] = trainset.shape
	return trainset, testset #, N, T, D