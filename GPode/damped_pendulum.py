import torch

class DampedPendulum:
    """
    Implements a parametric ODE system called damped pendulum.
    This is a second-order ODE model with the following equation
    
    .. math::

    \begin{equation*}
        \frac{d^2\theta}{dt^2} + \alpha\frac{d\theta}{dt} + \omega_0^2\sin\theta = 0
    \end{equation*}
    
    where :math:`\theta(t)` is the angle, :math:`\alpha` is the damping coefficient, and 
    :math:`\omega_0` is the angular frequency of the undumped motion.
    """
    def __init__(self, alpha=0.1, omega_0=0.01):
        self.alpha   = alpha
        self.omega_0 = omega_0
        
    def odef(self, t, x):
        theta, dtheta = x[:,:1], x[:,1:] 
        ddtheta = -self.omega_0**2*torch.sin(theta) - self.alpha*dtheta
        return torch.cat([dtheta,ddtheta],-1)