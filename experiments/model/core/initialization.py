from model.misc.constraint_utils import invsoftplus
import torch


def initialize_and_fix_kernel_parameters(model, lengthscale_value=1.25, variance_value=0.5, fix=False):
    """
    Initializes and optionally fixes kernel parameter 

    @param model: a gpode.SequenceModel object
    @param lengthscale_value: initialization value for kernel lengthscales parameter
    @param variance_value: initialization value for kernel signal variance parameter
    @param fix: a flag variable to fix kernel parameters during optimization
    @return: the model object after initialization
    """
    model.flow.odefunc.diffeq.kern.unconstrained_lengthscales.data = invsoftplus(
        lengthscale_value * torch.ones_like(model.flow.odefunc.diffeq.kern.unconstrained_lengthscales.data))
    model.flow.odefunc.diffeq.kern.unconstrained_variance.data = invsoftplus(
        variance_value * torch.ones_like(model.flow.odefunc.diffeq.kern.unconstrained_variance.data))
    if fix:
        model.flow.odefunc.diffeq.kern.unconstrained_lengthscales.requires_grad_(False)
        model.flow.odefunc.diffeq.kern.unconstrained_variance.requires_grad_(False)
    return model