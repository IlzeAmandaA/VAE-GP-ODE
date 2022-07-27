# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:39:26 2022

@author: YIC2RNG
"""

import torch
import torch.nn as nn
import numpy as np
dtype = torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from damped_pendulum import DampedPendulum
from gpode           import GPODE
from helpers         import fit_model, plot_fit, integrate

# data generation
n   = 2   # observation dimensionality
N   = 10  # number of training sequences
T   = 25  # the length of each sequence
dt  = 0.1 # time difference between each observation pair
sig = 0.1 # standard deviation of the observation noise

unknown_parametric_ode = DampedPendulum(alpha=0.1, omega_0=1.0)
with torch.no_grad():
    ts = torch.arange(T, dtype=dtype, device=device) * dt
    x0 = torch.randn([N,n], dtype=dtype, device=device)
    X  = integrate(unknown_parametric_ode.odef, x0, ts)  # clean data sequences, [T,N,n]
    Y  = X + torch.randn_like(X)*sig
del unknown_parametric_ode


# model fit
ode_model = GPODE(n, n).to(device).to(dtype)
plot_fit(ode_model, Y[:,:3], ts, fname='before-fit.png') # plot three sequences
fit_model(ode_model, Y, ts, Niter=2000, lr=5e-3)
plot_fit(ode_model, Y[:,:3], ts, fname='after-fit.png')  # plot three sequences




















