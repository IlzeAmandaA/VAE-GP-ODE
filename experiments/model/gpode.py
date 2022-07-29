import torch
import torch.nn as nn   
from torchdiffeq import odeint

from .svgp import SVGP
from .helpers import integrate

class ELBO(nn.Module):
    def __init__(self, sgp):
        super().__init__()  
        nobs = sgp.nout
        self.raw_sn   = nn.Parameter(torch.zeros(nobs), requires_grad=True)
        self.softplus = nn.Softplus()
        self.kl_func  = sgp.kl
        
    @property
    def signal_noise(self):
        '''
        Transforms the raw observation noise parameter into positive range

        Returns
        -------
        torch.Tensor
            observation noise parameter - shape [nobs,nobs]

        '''
        return self.softplus(self.raw_sn).diag()
    
    def forward(self, Y, Yhat, N):
        mvn   = torch.distributions.MultivariateNormal(Y,self.signal_noise) # T,N
        lhood = mvn.log_prob(Yhat).sum(0) # shape is Y.shape[0] 
        lhood = lhood.mean() * N
        kl    = self.kl_func()
        return -lhood + kl


class GPODE(nn.Module):
    """
    Implements a black-box ODE model based on sparse Gaussian processes.
    """
    
    def __init__(self, nin, nout, M=50, kernel='rbf', var_appr='diag'):
        '''
        Initiates the system.

        Parameters
        ----------
        nin : int
            the number of input dimensions
        nout : int
            the number of output dimensions
        M : int, optional
            the number of inducing points
        kernel : str, optional
            the covariance function. Must be in ['rbf','0.5','1.5','2.5']. The default is 'rbf'.
        var_appr : str, optional
            variational approximation distribution. Must be in ['chol','diag','delta'], where
            chol and diag are Gaussian posteriors with full and diagonal covariance
            approximations. Delta stands for point estimation. The default is 'diag'.

        Returns
        -------
        None.

        '''
        super().__init__()
        Z = torch.randn([M,nin])
        self.svgp  = SVGP(Z, nout, kernel=kernel, u_var=var_appr)
        self.loss = ELBO(self.svgp)

    def forward_trajectory(self, z0, logp0, ts):
        gp_draw = self.svgp.draw_posterior_function() #draw a differential function from GP
       # odef = lambda t,x: gp_draw(x)
        oderhs = lambda t, x: self.svgp.ode_rhs(t,x,gp_draw) # make the ODE forward function 
        zt, logp = odeint(oderhs, (z0, logp0), ts,method="rk4") # T,N,2q & T,N
        # integrate(odef, x0, ts)
        return zt, logp
        