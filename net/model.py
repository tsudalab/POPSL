# 1. mapping: from simplex to manifold

import argparse
import typing

import copy

import numpy as np
import numpy.random as npr

from net.problem import get_problem

import torch
import gpytorch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pymoo.indicators.hv import Hypervolume as HV
import torch.nn.functional as F
import os
import sys

from dcem.dcem_bm import dcem


class DCEM(torch.nn.Module):
    def __init__(self, cfg, model_gp):
        super(DCEM, self).__init__()
        self.cfg = cfg
        # self.prob_size = cfg.data.prob_size
        self.in_dim = cfg.train.in_dim
        self.out_dim = cfg.train.out_dim
        self.problem = cfg.train.problem
        self.model_gp = model_gp
        
        self.rho = cfg.model.rho
        self.norm_per = cfg.output.norm_per
        

        self.exp_dir = os.getcwd()
        self.model_dir = os.path.join(self.exp_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        torch.manual_seed(1)
        npr.seed(1)

        self.device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        self.map_net = MappingNet(n_z_dim=self.in_dim, n_x_dim=self.out_dim).to(self.device)
    
    def re_func_chem(self, w, y_gt): 
        if len(y_gt.shape) <= 1:
            y_gt = y_gt.reshape(1, -1).repeat(w.shape[0], 1) 
        y_gt = y_gt.unsqueeze(1)
        if len(w.shape) > 2:
            y_gt = y_gt.repeat(1, w.shape[1], 1)
            
        x = self.map_net(w).to(torch.float32)
        y_pred, _ = self.model_gp.predict(x)
        y_pred = y_pred.to(torch.float32)

       
        ############### pbi
        w_norm = torch.linalg.norm(w, dim=-1)
        y_F = (y_pred.float() - y_gt)
        d1 = torch.sum(y_F * w, dim=-1) / w_norm
        y_dis = d1 + self.rho * torch.linalg.norm(y_F - (d1.unsqueeze(-1) * w) / w_norm.unsqueeze(-1), dim=-1) 
        
        return y_dis, x, y_pred 

    def forward(self, epoch, y_gt, mode):

        # print('weight: ', weight)
        self.n_iter = 5
        if mode in ['eval']:
            self.n_iter = 1
        z_pred, w_pred = dcem(
            self.re_func_chem,
            n_batch=y_gt.shape[0],
            nx=self.in_dim,
            n_sample=1000,
            n_elite=101,
            n_iter=self.n_iter,
            temp=10,
            device=self.device,
            normalize=True,
            gt=property,
            epoch=epoch,
            norm_per=self.norm_per,
            y_gt=y_gt,
        )
        _, x_pred, y_pred = self.re_func_chem(w_pred.unsqueeze(1), y_gt)

        return y_pred.squeeze(), w_pred, x_pred


class MappingNet(torch.nn.Module):
    def __init__(self, n_z_dim, n_x_dim):
        super(MappingNet, self).__init__()
        self.n_in = n_z_dim
        self.n_out = n_x_dim
       
        self.fc1 = nn.Linear(self.n_in, 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc2_ = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_out)
        
        self.elu = nn.ELU()
        
        # self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            print(m)
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
    def forward(self, z):
        x = self.elu(self.fc1(z))
        x = self.elu(self.fc2(x))
        x = self.fc3(x) 
        x = torch.sigmoid(x)
       
        return x.to(torch.float64)

class GPModel():
    def __init__(self, n_obj, device, lengthscale=1.0, variance=1.0, noise=1e-5):
        self.n_obj = n_obj
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale))
        self.variance = nn.Parameter(torch.tensor(variance))
        self.noise = noise
        self.device = device
    
    def matern52_kernel(self, x1, x2, lengthscale=1.0, variance=1.0):
        dist = torch.cdist(x1, x2)
        sqrt5_dist = torch.sqrt(torch.tensor(5.0)) * dist
        return variance * (1 + sqrt5_dist / lengthscale + 5 * (dist ** 2) / (3 * lengthscale ** 2)) * torch.exp(-sqrt5_dist / lengthscale)
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.L = []
        self.alpha = []

        for i in range(self.n_obj):
            K = self.matern52_kernel(X_train, X_train, self.lengthscale, self.variance) + (self.noise * torch.eye(X_train.size(0))).to(self.device)
            L = torch.linalg.cholesky(K)
            self.L.append(L)
            alpha = torch.linalg.solve_triangular(L, y_train[:, i].unsqueeze(1), upper=False)
            alpha = torch.linalg.solve_triangular(L.t(), alpha, upper=True)
            self.alpha.append(alpha)

    def predict(self, X_test):
        mu = []
        cov = []
        # print('X_test shape: ', X_test.shape)
        for i in range(self.n_obj):
            K_s = self.matern52_kernel(self.X_train, X_test, self.lengthscale, self.variance)
            K_ss = self.matern52_kernel(X_test, X_test, self.lengthscale, self.variance) + self.noise * torch.eye(X_test.size(1)).to(X_test.device)

            # mean
            mu_i = torch.matmul(K_s.transpose(-1, -2), self.alpha[i]).squeeze(-1)
            mu.append(mu_i)

            # covariance
            v = torch.linalg.solve_triangular(self.L[i], K_s, upper=False)
            cov_i = K_ss - torch.matmul(v.transpose(-1, -2), v)
            cov.append(cov_i)

        return torch.stack(mu, dim=-1), cov

