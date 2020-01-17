#!/usr/bin/env python
# coding: utf-8

"""Written by Sungjun Choi, CMU MSML'19.

Contains functions that compute the HVP and
hessian of functions (PyTorch computational
graphs) w.r.t. parameters.
"""

import torch
import numpy as np
from torch.autograd import grad
from torch import nn


def hessian_vector_product(ys, params, vs, params2=None):
    """
    :ys: scalar that is to be differentiated
    :params: list of vectors (torch.tensors) w.r.t. each of
            which the hessian is computed
    :vs: the list of vectors each of which is to be multiplied
            to the hessian w.r.t. each parameter
    :params2: another list of params for second `grad` call
            in case the second derivation is w.r.t. a
            different set of parameters
    """
    grads1 = grad(ys, params, create_graph=True)
    if params2 is not None:
        params = params2
    """Deprecated, does the same work as below
    grad_v_prods = [gr * v.detach() for gr, v in zip(grads1, vs)]    # Element-wise multiply
    grad_outputs2 = [torch.ones_like(gvp) for gvp in grad_v_prods]
    grads2 = grad(grad_v_prods, params,
                  grad_outputs=grad_outputs2,
                  allow_unused=True)
    """
    grads2 = grad(grads1, params, grad_outputs=vs)
    return grads2


def hessians(ys, params):
    """Returns a list of hessians of `ys` w.r.t. each
    parameter in `params`, i.e. differentiate `ys`
    twice w.r.t. each parameter.
    Each output in the list is obtained by differentiating
    `ys` w.r.t. only a single parameter - in no cases
    `ys` is differentiated by two different parameters.
    Based on Adam Paszke's Github Gist code
    (https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7)
    
    :ys: scalar that is to be differentiated
    :params: list of vectors (torch.tensors) w.r.t. each of
            which the hessian is computed
    :returns: a list of hessians whose length is the same
            as that of `params` (one hessian per param)
    """
    jacobians = grad(ys, params, create_graph=True)
    
    # Container for hessians
    outputs = []
    for j, param in zip(jacobians, params):
        hess = []
        j_flat = j.flatten()
        for i in range(len(j_flat)):
            grad_outputs = torch.zeros_like(j_flat)
            grad_outputs[i] = 1
            grad2 = grad(j_flat, param, grad_outputs=grad_outputs, retain_graph=True)[0]
            hess.append(grad2)
        outputs.append(torch.stack(hess).reshape(j.shape + param.shape))
    return outputs
