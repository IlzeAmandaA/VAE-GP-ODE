# Latent GP-ODEs with Informative Priors

Pytroch implementation of Neurips 22 workshop paper Latent GP-ODEs with Informative Priors by Ilze Amanda Auzina, Cagatay Yildiz and Efstratios Gavves.

We tackle the exisitng limitations of GP-ODE models:

- applicable only on low-dimensional data settings. The input is either some low dimension observations or simulation data of the actual ODE system.

- no the use of prior knowledge of the system, an integral part of GP modeling.

We propose VAE-GP-ODE: a probabilistic dynamic model that extends previous work by
learning dynamics from high-dimensional data with a structured GP prior. Our model is trained end-to-end using variational inference. 

## Replicating the Experiments

The code was developed and tested with `python3.8` and `Pytorch 1.13`.

## Datasets
**Fixed Initial Angle** : the rotating mnist dataset with fixed initial angle can be donwloaded from [here](https://drive.google.com/drive/folders/1rOnMczoOXItqjM85VHo-LGvBjqcFctvn?usp=share_link). The data should be placed in `experiments/data/rot_mnist` directory.

**Random Initial Angle** : the rotating mnist dataset with random intial angle can be created by running the code with flag `--rotrand True`, the code with automatically shuffle the existing data in the correct manner.

## Running the experiments 

**1st ODE** The experiments can be run from the command line with passing arguments to the relevant experimental setup. If the current directory is experiments then an example command is as follows

```
python main.py --ode 1 --kernel RBF --D_in 6 --D_out 6 --latent_dim 6 --Nepoch 5000 --lr 0.001 --variance 1.0 --lengthscale 2.0 --rotrand True
```
The above command will run a first order ODE model with an RBF kernel with VAE latent space of 6. For experiment with divergence free kernel the above command can be adjust to `--kernel DF`. 

**2nd ODE** For a second orde ODE system the above command should be changed to:

```
python main.py --ode 2 --kernel RBF --D_in 6 --D_out 3 --latent_dim 3 --Nepoch 5000 --lr 0.001 --variance 1.0 --lengthscale 2.0 --rotrand True
```
The main different is that the ouput and latent dimensionality is reduced to 3 as there are 2 encoders now, one for the state and one for the velocity. 

**Pretrained VAE** To train a decoupled model (VAE training separate form GP-ODE training). You can run the `main_vae.py` file that will train a VAE, for example to train a model on dataset with 16 rotation angles (T=16) and with latent space dimensionality of 6 run the following

```
python main_vae.py --n_angle 16 --latent_dim 6
```
Runnin the above will create a new data repository 'data/moving_mnist' where the training data for the VAE training will be stored. The rotation angle can be increased to also 64, meaning that there will be 64 timesteps for a full rotation. 

Subsequently the trained VAE can be used in the GP-ODE training code as 

```
python main.py --ode 1 --rotrand True --pretrained True --vae_path 'specify your model path'
```

All results will be stores in directotory results with a corresponding timestamp as folder name. Each folder includes a log file with all experimental details and the training performance report. 

## Results from Trained Models

In the `plot_dynamics.ipynb` you can see the performance results of the trained models as reported in the final paper.

In the `plot_dynamics_extended.ipynb` there are additional models reported for further investigation of the models sensitivty to the training setup. 

