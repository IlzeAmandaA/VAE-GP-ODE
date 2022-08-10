import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_nonadjoint


class ODEfunc(nn.Module):
    def __init__(self, diffeq):
        """
        Defines the ODE function:
            mainly calls layer.build_cache() method to fix the draws from random variables.
        Modified from https://github.com/rtqichen/ffjord/

        @param diffeq: Layer of GPODE/npODE/neuralODE
        """
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, sample, rebuild_cache):
        self.sample = sample
        self._num_evals.fill_(0)
        if rebuild_cache:
            self.diffeq.build_cache()

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t, vs_logp): #this forward is my oderhs 
        vs, logp = vs_logp
        self._num_evals += 1
        if self.sample:
            q = vs.shape[1]//2
            dv = self.diffeq(vs) # N,q
            ds = vs[:,:q]  # N,q
            return (torch.cat([dv,ds],1), logp) # N,2q    
        else:
            q = vs.shape[1]//2
            dv = self.diffeq(vs) # 25,8
            ds = vs[:,:q]  # N,q
            dvs = torch.cat([dv,ds],1) # N,2q
            ddvi_dvi = torch.stack(
                        [torch.autograd.grad(dv[:,i],vs,torch.ones_like(dv[:,i]),
                        retain_graph=True,create_graph=True)[0].contiguous()[:,i]
                        for i in range(q)],1) # N,q --> df(x)_i/dx_i, i=1..q
            tr_ddvi_dvi = torch.sum(ddvi_dvi,1) # N
            return (dvs, -tr_ddvi_dvi)


class Flow(nn.Module):
    def __init__(self, diffeq, solver='dopri5', atol=1e-6, rtol=1e-6, use_adjoint=False):
        """
        Defines an ODE flow:
            mainly defines forward() method for forward numerical integration of an ODEfunc object
        See https://github.com/rtqichen/torchdiffeq for more information on numerical ODE solvers.

        @param diffeq: Layer of GPODE/npODE/neuralODE
        @param solver: Solver to be used for ODE numerical integration
        @param atol: Absolute tolerance for the solver
        @param rtol: Relative tolerance for the solver
        @param use_adjoint: Use adjoint method for computing loss gradients, calls odeint_adjoint from torchdiffeq
        """
        super(Flow, self).__init__()
        self.odefunc = ODEfunc(diffeq)
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint

    def forward(self, z0, logp0, ts, sample=False):
        """
        Numerical solution of an IVP
        @param z0: Initial latent state (N,2q)
        @param logp0: Initial distirbution (N)
        @param ts: Time sequence of length T, first value is considered as t_0
        @return: zt, logp: (N,T,2q) tensor, (N,T) tensor
        """
        odeint = odeint_adjoint if self.use_adjoint else odeint_nonadjoint
        self.odefunc.before_odeint(sample=sample, rebuild_cache=True)
        if sample:
            zt, _ = odeint(
                self.odefunc,
                (z0,logp0),
                ts,
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver
            )
            return zt.permute([1,0,2]), _
        else:
            zt, logp = odeint(
                self.odefunc,
                (z0, logp0),
                ts,
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver
            )
            return zt.permute(1, 0, 2), logp.permute([1,0])  # (N,T,2q) (N,T)

    def num_evals(self):
        return self.odefunc.num_evals()

    def kl(self):
        """
        Calls KL() computation from the diffeq layer
        """
        return self.odefunc.diffeq.kl() #.sum()

    def log_prior(self):
        """
        Calls log_prior() computation from the diffeq layer
        """
        return self.odefunc.diffeq.log_prior().sum()

    '''
    Do we need this? (would have to adjust) consistency loss 
    def inverse(self, x0, ts, return_divergence=False):
        odeint = odeint_adjoint if self.use_adjoint else odeint_nonadjoint
        self.odefunc.before_odeint(return_divergence=return_divergence, rebuild_cache=False)
        if return_divergence:
            states = odeint(
                self.odefunc,
                (x0, torch.zeros(x0.shape[0], 1).to(x0)),
                torch.flip(ts, [0]),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver
            )
            xs, divergence = states
            return xs.permute(1, 0, 2), divergence.permute(1, 0, 2)  # (N,T,D), # (N,T,1)
        else:
            xs = odeint(
                self.odefunc,
                x0,
                torch.flip(ts, [0]),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver
            )
            return xs.permute(1, 0, 2)  # (N,T,D)
    ''' 