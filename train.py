import logging
import os
import sys
import random

import torch
import gpytorch
import torch.nn as nn
from torch import optim
# from tqdm import tqdm
from os.path import abspath
import numpy as np
import pymoo
import datetime
from pyDOE import lhs

# from dataset import load_dataset, MO_Dataset
from net.DCEM_dtlz import DCEM, SurModel, GPModel
# from net.DCEM_chem_net import DCEM
# from util.model_util import evaluate


from torch.autograd import gradcheck
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from dataset import load_chem_dataset
from geomloss import SamplesLoss
# from paretoset import paretoset

import hydra
from hydra.utils import to_absolute_path
from pymoo.indicators.hv import Hypervolume as HV
from scipy.spatial import ConvexHull

from torch.distributions.dirichlet import Dirichlet
from net.problem import get_problem
from pymoo.problems.many.dtlz import DTLZ2
from pymoo.decomposition.asf import ASF


# def pre_train(device, cfg, model, likelihood, mll, train_x, train_y):
#     num_steps = 10
#     # net = SurModel(cfg.train.in_dim, cfg.train.out_dim)
#     # model.to(device)
#     # likelihood.to(device)
#     # train_x.to(device)
#     # train_y.to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     for i in range(num_steps):
#         optimizer.zero_grad()
#         output = model(train_x)
#         print(likelihood(output).mean.shape, train_y.shape)
#         loss = -mll(output, train_y)
#         loss.backward(retain_graph=True)
#         print(f'Iter {i + 1}/{num_steps} - Loss: {loss.item():.3f}')
#         optimizer.step()
#     return model

def penalty_item(obj_mat, ref_vec):
    dot_product = torch.bmm(obj_mat, ref_vec.transpose(1, 2))
    norm_obj = torch.norm(obj_mat, dim=-1).unsqueeze(-1)
    norm_vec = torch.norm(ref_vec, dim=-1).unsqueeze(-1)
    
    cos_vec = dot_product / (norm_obj * norm_vec)
    # print(dot_product.shape, norm_obj.shape, norm_vec.shape, cos_vec.shape)
    
    return cos_vec

def train_net(device, cfg):

    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size
    
    # optimizer & loss
    num_iter = 4
    n_run = 1
    # n_sample = 10 
    # inner_steps = 5
    n_obj = cfg.train.in_dim
    n_var = cfg.train.out_dim
    num_bs = cfg.train.num_bs
    
    
  

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Checkpoints:     {cfg.output.save}
        Device:          {device.type}
    ''')

    # training 
    # file_path = to_absolute_path('data/gt_dtlz2_2.csv')
    # data_gt = np.genfromtxt(file_path, dtype='float', delimiter=',')[:, :2]
    torch.autograd.set_detect_anomaly(True)
    for run_num in range(n_run):
        setup_seed(10*run_num)

        # data_lits = data_list[np.random.choice(data_list.shape[0], 10), :]
        x_init = torch.from_numpy(np.random.uniform(0, 1, (50, n_var))).to(torch.float32).to(device)
        F1 = get_problem(cfg.train.problem)
        y_init = F1.evaluate(x_init).to(torch.float32).to(device)
        # data_lits = torch.cat((y_init, x_init), 1)
        x_new = x_init
        y_new = y_init
        
        net_pre = GPModel(n_obj, device)
        net_pre.fit(x_init, y_init)

        net = DCEM(cfg, net_pre)
        add_perturbation_to_parameters(net)
        # net = nn.DataParallel(net)
        net.to(device=device)
        optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
        # break
        # np.random.seed(10*run_num)
        time_list = []
        model_time_list = []
        for epoch in range(epochs):
            time_s = datetime.datetime.now()
            # for n_iter in range(num_iter):
            # d_item = data_gt[np.random.choice(data_gt.shape[0], batch_size), :]
            d_item = -0.1 * np.ones((batch_size*n_obj, n_obj)) # np.random.uniform(0, 1, (batch_size*n_obj, n_obj))
            for n_o in range(n_obj):
                d_item[(n_o * batch_size): ((n_o + 1) * batch_size), n_o] = np.random.uniform(-0.1, 1, batch_size) # 0
                # d_item[(n_o * batch_size): ((n_o + 1) * batch_size), 0] = np.random.uniform(0, 1, batch_size) # 0
                # d_item[(n_o * batch_size): ((n_o + 1) * batch_size), 1] = np.random.uniform(-0, 1, batch_size) # 0
            # d_item = 0 * np.ones((batch_size*n_obj, n_obj)) # np.random.uniform(0, 1, (batch_size*n_obj, n_obj))
            # for n_o in range(n_obj):
            #     d_item[(n_o * batch_size): ((n_o + 1) * batch_size), n_o] = np.random.uniform(0, 1, batch_size) # 0
            # d_item_1 = np.random.uniform(0.5, 1, (batch_size*n_obj - n_obj, n_obj))
            # for n_o in range(n_obj):
            #     d_item_1[(n_o * batch_size): ((n_o + 1) * batch_size), n_o] = 0
            # d_item_2 = np.random.uniform(0, 0.2, (n_obj, n_obj))
            # for n_o in range(n_obj):
            #     d_item_2[(n_o * batch_size): ((n_o + 1) * batch_size), n_o] = 0
            # d_item = np.vstack((d_item_1, d_item_2))
            # print(d_item.shape)
            d_item = torch.Tensor(d_item).to(device)
            y_pred, w_pred, _ = net(epoch, d_item, 'train')
            
            y_best = y_pred[:, 0, :].unsqueeze(1)
            y_pred = y_pred[:, 1:, :]
            w_pred = w_pred[:, 1:, :]
            # print(y_pred.shape)
            # y_pred.backward()
            # w_pred.retain_grad()
            
            
            # print(y_pred.shape, d_item.shape)
            # print(d_item)
            
            # d_item_ = d_item.unsqueeze(1).repeat(1, y_pred.shape[1], 1)
            # # print(d_item, d_item_)
            # w_norm = torch.linalg.norm(w_pred, dim=-1)
            # y_F = (y_pred.float() - d_item_)
            # d1 = torch.sum(y_F * w_pred, dim=-1) / w_norm
            # d2 = torch.linalg.norm(y_F - (d1.unsqueeze(-1) * w_pred) / w_norm.unsqueeze(-1), dim=-1)
            d1, d2 = PBI_cal(d_item, w_pred, y_pred)
            
            print((y_pred - y_best).shape, (d_item.unsqueeze(1) - y_best).shape)
            pty_item = penalty_item((y_pred - y_best + 1e-6), (d_item.unsqueeze(1) - y_best))
            constraint_1 = (torch.sqrt(torch.tensor(2.0)) / 2. - pty_item).squeeze() # (0 - pty_item).squeeze()
            constrain_2 = (pty_item - 1).squeeze()
            print(pty_item)
            
            loss = d1 + 5 * d2 + 1 * torch.exp(constraint_1 * constrain_2)  # + 0.1 * torch.sum(torch.mul((1 - w_pred.squeeze()), y_pred.float()), dim=-1) 
            
            # print('loss: ', epoch, loss.shape)
            
            optimizer.zero_grad()
            loss.backward(torch.ones((batch_size*n_obj, 100)).to(device), retain_graph=True)
            # loss.backward(2*torch.rand((batch_size*n_obj, 50)).to(device), retain_graph=True)
            # loss.backward(retain_graph=True) # loss.backward(torch.ones(batch_size), retain_graph=True)
            optimizer.step()
            
            print(w_pred.shape)
            
            # for name, param in net.named_parameters():
            #     # if param.grad is not None:
            #     print(f'Epoch {epoch}, {name} grad: {param.grad}')

            time_e = datetime.datetime.now()
            model_time_list.append((time_e - time_s).seconds)
            prefs, x_pred, y_pred = evaluate(net, device, cfg.train, epoch, run_num)
            
            # batch selection
            x_sel, y_sel = paretoset_sel(d_item, prefs, x_pred, y_pred, y_best, net, num_bs)
            x_new = torch.vstack((x_new, x_sel))
            y_new = torch.vstack((y_new, y_sel.to(device)))
            # # print('y_new: ', y_new)
            net_pre.fit(x_new, y_new)

            

            time_e = datetime.datetime.now()
            time_list.append((time_e - time_s).seconds)
            print('avg time: ', np.mean(time_list), np.mean(model_time_list), model_time_list)
            # break

            
def paretoset_hvi(d_cur, d_new, n_obj, n_sample):
    best_subset_list = []
    Y_p = d_cur[:, :n_obj] 
    Y_candidate_np = d_new[:, :n_obj] 
    for b in range(n_sample):
        hv = HV(ref_point=np.max(np.vstack([Y_p,Y_candidate_np]), axis = 0))
        best_hv_value = 0
        best_subset = None
        
        for k in range(len(Y_candidate_np)):
            Y_subset = Y_candidate_np[k]
            Y_comb = np.vstack([Y_p,Y_subset])
            hv_value_subset = hv._do(Y_comb)
            if hv_value_subset > best_hv_value:
                best_hv_value = hv_value_subset
                best_subset = [k]
                
        Y_p = np.vstack([Y_p,Y_candidate_np[best_subset]])
        best_subset_list.append(best_subset)  
    best_subset_list = np.array(best_subset_list).T[0]
    d_list = np.vstack((d_cur, d_new[best_subset_list, :]))
    return d_list

def PBI_cal(d_item, w_pred, y_pred):
    d_item_ = d_item.unsqueeze(1).repeat(1, y_pred.shape[1], 1)
    # print(d_item, d_item_)
    w_norm = torch.linalg.norm(w_pred, dim=-1)
    y_F = (y_pred.float() - d_item_)
    d1 = torch.sum(y_F * w_pred, dim=-1) / w_norm
    d2 = torch.linalg.norm(y_F - (d1.unsqueeze(-1) * w_pred) / w_norm.unsqueeze(-1), dim=-1)
    return d1, d2

def paretoset_sel(d_item, prefs, x_pred, y_pred, y_best, net, num_bs):
    net.eval()
    n_ref = d_item.shape[0]
    n_spl = int(np.ceil(num_bs/n_ref)) + 2
    with torch.no_grad():
        map_net = net.map_net
        # x_sel = map_net(prefs).unsqueeze(0).to(torch.float32)
        # print(x_sel.shape)
        # print(x_pred.shape, y_pred.shape)
        x_pred = x_pred.unsqueeze(0).to(torch.float32)
        y_sel, _ = net.model_gp.predict(x_pred)
        y_sel = y_sel.to(torch.float32)
        d1, d2 = PBI_cal(d_item, prefs, y_sel)
        pty_item = penalty_item((y_sel - y_best + 1e-6), (d_item.unsqueeze(1) - y_best))
        constraint_1 = (torch.sqrt(torch.tensor(2.0)) / 2. - pty_item).squeeze() # (0 - pty_item).squeeze()
        constrain_2 = (pty_item - 1).squeeze()
        scores = d1 + 5. * d2 + 1 * torch.exp(constraint_1 * constrain_2)
        idx_sub = torch.argsort(scores, dim=1)[:, :n_spl]
        idx_sub = idx_sub.cpu().detach().numpy()
        uq_idx, uq_iidx = np.unique(idx_sub, return_index=True)
        # uq_idx, uq_inv = torch.unique(idx_sub.flatten(), return_inverse=True, return_counts=False, sorted=True)
        # idx = torch.randperm(len(uq_idx))[:num_bs]
        idx = np.random.choice(len(uq_idx), num_bs, replace=False)
        # idx_sel_1 = uq_iidx[idx] // idx_sub.shape[1]
        # idx_sel_2 = uq_iidx[idx] % idx_sub.shape[1]
        x_sel = x_pred.squeeze()[idx, :]
        y_sel = y_pred[idx, :]
        # print(y_sel.shape, y_pred.shape)
        assert x_sel.shape[0] == y_sel.shape[0] and y_sel.shape[1] == d_item.shape[-1]
        # y_sel = y_sel[idx_sel_1, idx_sel_2]
        # print(n_spl, idx_sub[idx_sel_1, idx_sel_2], uq_idx[idx], uq_iidx, scores.shape)
    return x_sel, torch.Tensor(y_sel)          

def add_perturbation_to_parameters(model, perturbation_std=0.15):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * perturbation_std)
            # print(param)

def evaluate(net, device, params, epoch, run_num):
#     x_pred
    m_obj = params.in_dim
    m_vars = params.out_dim
    print('evaluate------')
    net.eval()
    with torch.no_grad():
        map_net = net.map_net
        pref = np.random.dirichlet(np.ones(m_obj), 100)
        pref = torch.Tensor(pref).to(device)
        x_pred = map_net(pref)
        F1 = get_problem(params.problem) # problem selection
        # F1 = DTLZ2(n_var=m_vars, n_obj=m_obj)
        y_pred = F1.evaluate(x_pred).cpu().detach().numpy()
        file_path = to_absolute_path('result/y-'+str(epoch)+'-'+str(run_num)+'.csv')
        np.savetxt(file_path, y_pred, fmt='%.8f', delimiter=',', newline='\n', header=str(epoch))
    return pref, x_pred, y_pred

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True 


@hydra.main(config_path='config', config_name='mo_train')
def main(cfg):
    torch.multiprocessing.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # setup_seed(1000)
    train_net(device=device, cfg=cfg)


if __name__ == '__main__':
    main()
