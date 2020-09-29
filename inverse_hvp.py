#!/usr/bin/env python
# coding: utf-8

"""Written by Sungjun Choi, CMU MSML'19.

Contains function that computes the inverse-HVP
of a function (PyTorch computational graph)
w.r.t. parameters in the two ways specified
in the original paper.
"""

import torch
import numpy as np
from torch.autograd import grad
from torch import nn
from torch.utils.data import Dataset, DataLoader

from influence_fn_pytorch.hessian_hvp_utils import hessian_vector_product, hessians
from influence_fn_pytorch.mnist_logistic_binary import create_binary_MNIST, preproc_binary_MNIST

DATA_DIR = "./data"
LRG_MODEL_PATH = "./model/mnist_logistic_reg.pt"


def get_inverse_hvp(model, criterion, dataset, vs,
                    approx_type="cg",
                    approx_params={},
                    collate_fn=None,
                    has_label=True,):
    """Wrapper for the two inverse-hvp computation methods.

    :model, criterion, dataset: needed to compute empirical risk
    :vs: list of vectors in the inverse-hvp, one per each parameter
    :approx_type: choice of method, 'cg' or 'lissa'
    :approx_params: parameters specific to 'lissa' method
    :collate_fn: function to preprocess each minibatch from `dataset`
    :returns: list of inverse-hvps computed per each param
    """
    if approx_type == "cg":
        return get_inverse_hvp_cg(model, criterion, dataset, vs,
                                  collate_fn=collate_fn)
    elif approx_type == "lissa":
        return get_inverse_hvp_lissa(model, criterion, dataset, vs,
                                     **approx_params,
                                     collate_fn=collate_fn,
                                     has_label=has_label)
    else:
        raise NotImplementedError("ERROR: Only types 'cg' and 'lissa' are supported")

        
def get_inverse_hvp_cg(model, criterion, dataset, vs,
                       collate_fn=None, has_label=True):
    """Compute the product of inverse hessian of empirical risk
    and the given vector 'v' using conjugate gradient method.
    
    :model, criterion, dataset: needed to compute empirical risk
    :vs: list of vectors in the inverse-hvp, one per each parameter
    :collate_fn: function to preprocess each minibatch
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
                          collate_fn=None,
                          has_label=True,
                          verbose=False):
    """Compute the product of inverse hessian of empirical risk
    and the given vector 'v' numerically using LiSSA algorithm.
    
    :model: the model of interest
    :criterion: the objective used to train the given model
    :inputs, targets: dataset to compute loss with
    :vs: list of vectors in the inverse-hvp, one per each parameter
    :batch_size: size of minibatch sample at each iteration
    :scale: the factor to scale down loss (to keep hessian <= I)
    :damping: lambda added to guarantee hessian be p.d.
    :num_repeats: hyperparameter 'r' in in the paper (to reduce variance)
    :recursion_depth: number of iterations for LiSSA algorithm
    :returns: list of inverse-hvps computed per each param
    """
    assert criterion is not None, "Provide the criterion used to train model"
    assert batch_size <= len(dataset), \
        "ERROR: Minibatch size for LiSSA should be less than dataset size"
    assert len(dataset) % batch_size == 0, \
        "ERROR: Dataset size for LiSSA should be a multiple of minibatch size"
    assert isinstance(dataset, Dataset), "ERROR: `dataset` must be PyTorch Dataset"

    params = [param for param in model.parameters() if param.requires_grad]

    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)
    inverse_hvp = None

    for rep in range(num_repeats):
        cur_estimate = vs
        data_iter = iter(data_loader)   # To allow for multiple cycles through data_loader
        for it in range(recursion_depth):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)

            if has_label:
                batch_inputs, batch_targets = batch
                batch_out = model(batch_inputs)
                assert batch_out.shape == batch_targets.shape, \
                    "Model output and label should have the same shape"
                loss = criterion(batch_out, batch_targets) / batch_size
            else:
                batch_inputs = batch[0]
                loss = criterion(model(batch_inputs)) / batch_size

            hvp = hessian_vector_product(loss, params, vs=cur_estimate)
            cur_estimate = [v + (1-damping) * ce - hv / scale \
                            for (v, ce, hv) in zip(vs, cur_estimate, hvp)]

            if verbose and (it+1) % 100 == 0:
                print(f">>> Completed iteration {it+1}")
            
        inverse_hvp = [hv1 + hv2 / scale for (hv1, hv2) in zip(inverse_hvp, cur_estimate)] \
                       if inverse_hvp is not None \
                       else [hv2 / scale for hv2 in cur_estimate]
    
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
                                  collate_fn=preproc_binary_mnist,
                                 )

    for item in inverse_hvp:
        print(item.shape)

    print(hvp)

