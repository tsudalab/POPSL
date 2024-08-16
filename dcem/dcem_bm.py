# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import datetime

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.dirichlet import Dirichlet
import torch.nn.functional as F

from lml import LML
from hydra.utils import to_absolute_path as abs_path
import matplotlib.pyplot as plt

def dcem(
    f,
    nx,
    gt=None,
    n_batch=100,
    n_sample=1000,
    n_elite=100,
    n_iter=10,
    temp=1.,
    lb=None,
    ub=None,
    init_mu=None,
    init_sigma=None,
    device=None,
    iter_cb=None,
    proj_iterate_cb=None,
    lml_verbose=0,
    lml_eps=1e-3,
    normalize=True,
    iter_eps=1e-4,
    epoch=1,
    y_gt=None,
):
    if init_mu is None:
        init_mu = 0.

    size = (n_batch, nx)

    if isinstance(init_mu, torch.Tensor):
        mu = init_mu.clone()
        mu_ii = mu
    elif isinstance(init_mu, float):
        mu = init_mu * torch.ones(size, requires_grad=True, device=device)
        mu_ii = mu
    else:
        assert False

    # TODO: Check if init_mu is in the domain

    if init_sigma is None:
        init_sigma = 1.

    if isinstance(init_sigma, torch.Tensor):
        sigma = init_sigma.clone()
        sigma_ii = sigma
    elif isinstance(init_sigma, float):
        sigma = init_sigma * torch.ones(
            size, requires_grad=True, device=device)
        sigma_ii = sigma
    else:
        assert False

    assert mu.size() == size
    assert sigma.size() == size

    if lb is not None:
        assert isinstance(lb, float)

    if ub is not None:
        assert isinstance(ub, float)
        assert ub > lb

    time_s = datetime.datetime.now()

    for i in range(n_iter):
       
        X = Normal(mu, sigma).rsample((n_sample,)).transpose(0, 1).to(device) 
        
        mu = mu.unsqueeze(1).repeat(1, X.shape[1], 1)
        sigma = sigma.unsqueeze(1).repeat(1, X.shape[1], 1)
        l_bound = mu - 3.5 * sigma
        r_bound = mu + 3.5 * sigma
        X = (X - l_bound) / (r_bound - l_bound)
        X = X / torch.sum(X, dim=-1).unsqueeze(-1)
        X = X + 0.5 * torch.randn(X.size()).to(device) 
        
        X = X.contiguous()
        if lb is not None or ub is not None:
            X = torch.clamp(X, lb, ub)

        if proj_iterate_cb is not None:
            X = proj_iterate_cb(X)

        fX, inputs, objs = f(X.squeeze(), y_gt)
        X, fX = X.view(n_batch, n_sample, -1), fX.view(n_batch, n_sample)

        if temp is not None and temp < np.infty:
            if normalize:
                fX_mu = fX.mean(dim=1).unsqueeze(1)
                fX_sigma = fX.std(dim=1).unsqueeze(1)
                _fX = (fX - fX_mu) / (fX_sigma + 1e-6)
            else:
                _fX = fX

            if n_elite == 1:
                # I = LML(N=n_elite, verbose=lml_verbose, eps=lml_eps)(-_fX*temp)
                I = torch.softmax(-_fX * temp, dim=1)
            else:
                I = LML(N=n_elite, verbose=lml_verbose,
                        eps=lml_eps)(-_fX * temp)
            I = I.unsqueeze(2)
        else:
            I_vals = fX.argsort(dim=1)[:, :n_elite]
            # TODO: A scatter would be more efficient here.
            I = torch.zeros(n_batch, n_sample, device=device)
            for j in range(n_batch):
                for v in I_vals[j]:
                    I[j, v] = 1.
            I = I.unsqueeze(2)

        assert I.shape[:2] == X.shape[:2]
        X_I = I * X
        
        
        objs_I = I * objs
        old_mu = mu.clone().mean(dim=1)
        mu = torch.sum(X_I, dim=1) / n_elite
        sigma = ((I * (X - mu.unsqueeze(1))**2).sum(dim=1) / n_elite).sqrt()
        # alpha = torch.exp(mu+sigma)
        
        I_vals = fX.unsqueeze(-1).argsort(dim=1)[:, :n_elite]
        # print(torch.sort(fX, dim=1)[0][:, :100], fX.shape)
        I_mask = torch.zeros(I.shape[0], n_elite, I.shape[1]).to(device)
        I_mask.scatter_(-1, I_vals, 1)
        eilites = torch.bmm(I_mask, torch.softmax(I, dim=-1)*X) 
        
        # print('old mu: ', old_mu.shape, mu.shape)
        if (mu - old_mu).norm() < iter_eps:
            break

        if iter_cb is not None:
            iter_cb(i, X, fX, I, X_I, mu, sigma)
        # print('res: ', i, mu)
    time_e = datetime.datetime.now()
    print('running time: ', (time_e - time_s).seconds)

    if lb is not None or ub is not None:
        mu = torch.clamp(mu, lb, ub)
    # print('final mu: ', eilites[0])
    return mu, eilites
