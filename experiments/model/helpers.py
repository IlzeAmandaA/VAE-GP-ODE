# -*- coding: utf-8 -*-
import os, logging
import numpy as np

import torch
import torch.nn as nn

import os,matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def euler_step(f,h,t,x):
    return h*f(t,x)

def rk4_step(f,h,t,x):
    k1 = h * (f(t, x))
    k2 = h * (f((t+h/2), (x+k1/2)))
    k3 = h * (f((t+h/2), (x+k2/2)))
    k4 = h * (f((t+h), (x+k3)))
    return (k1+2*k2+2*k3+k4) / 6

def integrate(f, x0, T, solver='rk4'):
    '''
    x0: N 2q
    '''
    if solver=='euler':
        step_fnc = euler_step
    elif solver=='rk4':
        step_fnc = rk4_step

    q = x0.shape[1]//2
    ds = x0[:,:q]
    X  = [x0]
    ode_steps = len(T)-1
    for i in range(ode_steps):
        h  = T[i+1]-T[i]
        t  = T[i]
        x  = X[i]
        x_next = x + step_fnc(f,h,t,x)
        X.append(x_next)
    X = torch.stack(X) # T,N,d
    return X
    
def fit_model(model, Y, ts, lr=1e-3, Niter=1000, print_every=100):
    '''
    Fits the model parameters using gradient descent.

    Parameters
    ----------
    Y : torch.Tensor
        observed training sequences - shape [T,N,D] where N denotes the 
        number of training sequences, T is sequence length and D is the
        observation dimensionaliy
    ts : torch.Tensor
        observation times - shape [T]. Assumes that all observation times
        of all training sequences are identical.

    Returns
    -------
    logger : Logger
        stores training metrics and the parameters of the parametric model

    '''
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Ntrain = Y.shape[1]
    for it in range(Niter):
        opt.zero_grad()
        idx  = torch.randperm(Ntrain)[:5] 
        Y_minibatch = Y[:,idx]
        Yhat = model.forward_trajectory(Y_minibatch[0], ts)
        loss = model.loss(Y_minibatch, Yhat, Ntrain)
        loss.backward()
        opt.step()
        if it%print_every == 0:
            print(loss.item())

def plot_fit(model, Y, ts, L=10, fname='fit.png'):
    T,N,D = Y.shape
    with torch.no_grad():
        Yhat = torch.stack([model.forward_trajectory(Y[0],ts).cpu() for _ in range(L)]) # L,T,N,D
    Y,ts = Y.cpu(),ts.cpu()
    plt.figure(1,(9*D,3*N))
    for n in range(N):
        for d in range(D):
            plt.subplot(N,D,n*D+d+1)
            h1, = plt.plot(ts, Y[:,n,d], '*r', label='Data')
            for l in range(L):
                h2, = plt.plot(ts, Yhat[l,:,n,d], '-b', label='Fit')
            if n==0 and d==0:
                plt.legend(handles=[h1,h2])
            if d==0:
                plt.ylabel(f'Sequence-{n}')
            plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    
