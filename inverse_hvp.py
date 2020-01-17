#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch.autograd import grad
from torch import nn
from torch.utils.data import Dataset, DataLoader

from hessian_hvp_utils import hessian_vector_product, hessians
from mnist_logistic_binary import create_binary_MNIST, preproc_binary_MNIST

DATA_DIR = "./data"
LRG_MODEL_PATH = "./model/mnist_logistic_reg.pt"


def get_inverse_hvp(model, criterion, dataset, vs,
                    approx_type="cg", approx_params={}, preproc_data_fn=None):
    """Wrapper for the two inverse-hvp computation methods.

    :model, criterion, dataset: needed to compute empirical risk
    :vs: list of vectors in the inverse-hvp, one per each parameter
    :approx_type: choice of method, 'cg' or 'lissa'
    :approx_params: parameters specific to 'lissa' method
    :preproc_data_fn: function to preprocess each minibatch
            from `dataset`
    :returns: list of inverse-hvps computed per each param
    """
    if approx_type == "cg":
        return get_inverse_hvp_cg(model, criterion, dataset, vs,
                                                   preproc_data_fn=preproc_data_fn)
    elif approx_type == "lissa":
        return get_inverse_hvp_lissa(model, criterion, dataset, vs,
                                                      **approx_params, preproc_data_fn=preproc_data_fn)
    else:
        raise NotImplementedError("ERROR: Only types 'cg' and 'lissa' are supported")

        
def get_inverse_hvp_cg(model, criterion, dataset, vs,
                       preproc_data_fn=None):
    """
    Compute the product of inverse hessian of empirical risk
    and the given vector 'v' using conjugate gradient method.
    
    :model, criterion, dataset: needed to compute empirical risk
    :vs: list of vectors in the inverse-hvp, one per each parameter
    :preproc_data_fn: function to preprocess each minibatch
            from `dataset`
    :returns: list of inverse-hvps computed per each param
    """
    raise NotImplementedError("ERROR: 'cg' has not yet been implemented")

    
def get_inverse_hvp_lissa(model, criterion, dataset, vs,
                          batch_size=1,
                          scale=10,
                          damping=0.0,
                          num_repeats=1,
                          recursion_depth=10000,
                          preproc_data_fn=None):
    """
    Compute the product of inverse hessian of empirical risk
    and the given vector 'v' numerically using LiSSA algorithm.
    
    :model: the model of interest
    :criterion: the objective used to compute loss
    :inputs, targets: dataset to compute loss with
    :vs: list of vectors in the inverse-hvp, one per each parameter
    :batch_size: size of minibatch sample at each iteration
    :scale: the factor to scale down loss (to keep hessian <= I)
    :damping: lambda added to guarantee hessian be p.d.
    :num_repeats: hyperparameter 'r' in in the paper (to reduce variance)
    :recursion_depth: number of iterations for LiSSA algorithm
    :returns: list of inverse-hvps computed per each param
    """
    inverse_hvp = None
    
    assert batch_size <= len(dataset), \
        "ERROR: Minibatch size for LiSSA should be less than dataset size"
    
    params = list(model.parameters())
    
    if isinstance(dataset, Dataset):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        try:
            data_loader = iter(dataset)
        except:
            raise
        
    for rep in range(num_repeats):
        if not isinstance(data_loader,DataLoader):
            cur_estimate = vs
            idx = 0
            while idx + batch_size <= len(dataset):
                if idx / batch_size >= recursion_depth:
                    break

                inputs, targets = dataset[idx]
                if preproc_data_fn is not None:
                    inputs, targets = preproc_data_fn(inputs, targets)

                loss = criterion(model(inputs), targets)
                for i in range(idx+1,idx+batch_size):
                    inputs, targets = dataset[i]
                    if preproc_data_fn is not None:
                        inputs, targets = preproc_data_fn(inputs, targets)
                    loss += criterion(model(inputs), targets)

                loss /= batch_size
                
                hvp = hessian_vector_product(loss, params, vs=cur_estimate)
                cur_estimate = [v + (1-damping) * ce - hv / scale for (v, ce, hv) in zip(vs, cur_estimate, hvp)]
                idx += batch_size
        else:
            cur_estimate = vs
            for it, (batch_inputs, batch_targets) in enumerate(data_loader):
                if it >= recursion_depth:
                    break

                if preproc_data_fn is not None:
                    batch_inputs, batch_targets = preproc_data_fn(batch_inputs, batch_targets)

                loss = criterion(model(batch_inputs), batch_targets) / batch_size

                hvp = hessian_vector_product(loss, params, vs=cur_estimate)
                #print(max([torch.max(item) for item in cur_estimate]))
                cur_estimate = [v + (1-damping) * ce - hv / scale for (v, ce, hv) in zip(vs, cur_estimate, hvp)]
            
        inverse_hvp = [hv1 + hv2 / scale for (hv1, hv2) in zip(inverse_hvp, cur_estimate)] \
                       if inverse_hvp is not None else [hv2 / scale for hv2 in cur_estimate]
    
    # Average over repetitions
    inverse_hvp = [item / num_repeats for item in inverse_hvp]
    
    return inverse_hvp


if __name__ == "__main__":
    
    # Target model

    logistic_reg = nn.Sequential(
        nn.Linear(784,1, bias=True),
        #nn.Sigmoid()
    )
    logistic_reg.load_state_dict(torch.load(LRG_MODEL_PATH))
    params = list(logistic_reg.parameters())

    # Test code

    mnist_train, mnist_test = create_binary_MNIST(data_dir=DATA_DIR)
    sample_vs = [torch.ones_like(param) for param in logistic_reg.parameters()]
    lissa_params = {
        "batch_size": 10,
        "num_repeats": 10,
        "recursion_depth": 5000,
        "damping": 0.01,
    }

    inverse_hvp = get_inverse_hvp(logistic_reg,
                                                     nn.BCEWithLogitsLoss(),
                                                     mnist_train,
                                                     sample_vs,
                                                     approx_type="lissa",
                                                     approx_params=lissa_params,
                                                     preproc_data_fn=preproc_binary_mnist,
                                                    )

    for item in inverse_hvp:
        print(item.shape)

    print(hvp)

