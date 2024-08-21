import logging
import os
import random

import torch
from torch import optim
import numpy as np
import datetime

# from dataset import load_dataset, MO_Dataset
from net.model import DCEM, GPModel

import hydra
from hydra.utils import to_absolute_path
from pymoo.indicators.hv import Hypervolume as HV

from net.problem import get_problem



def penalty_item(obj_mat, ref_vec):
    dot_product = torch.bmm(obj_mat, ref_vec.transpose(1, 2))
    norm_obj = torch.norm(obj_mat, dim=-1).unsqueeze(-1)
    norm_vec = torch.norm(ref_vec, dim=-1).unsqueeze(-1)
    
    cos_vec = dot_product / (norm_obj * norm_vec)
    
    return cos_vec

def train_net(device, cfg):

    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size
    
    num_iter = cfg.train.num_iter
    n_run = cfg.train.n_run
    n_obj = cfg.train.in_dim
    n_var = cfg.train.out_dim
    num_bs = cfg.train.num_bs
    num_init = cfg.train.num_init
    
    rho = cfg.model.rho
    lmd = cfg.model.lmd
    ri = cfg.model.r_ideal
    lb = cfg.model.lb_ideal
    rb = cfg.model.rb_ideal
    
  

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Checkpoints:     {cfg.output.save}
        Device:          {device.type}
    ''')

    # training 
    torch.autograd.set_detect_anomaly(True)
    for run_num in range(n_run):
        setup_seed(10*run_num)

        x_init = torch.from_numpy(np.random.uniform(0, 1, (num_init, n_var))).to(torch.float32).to(device)
        F1 = get_problem(cfg.train.problem)
        y_init = F1.evaluate(x_init).to(torch.float32).to(device)
        x_new = x_init
        y_new = y_init
        
        net_pre = GPModel(n_obj, device)
        net_pre.fit(x_init, y_init)

        net = DCEM(cfg, net_pre)
        add_perturbation_to_parameters(net)
        net.to(device=device)
        optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
        time_list = []
        model_time_list = []
        for epoch in range(epochs):
            time_s = datetime.datetime.now()
            d_item = ri * np.ones((batch_size*n_obj, n_obj)) # np.random.uniform(0, 1, (batch_size*n_obj, n_obj))
            for n_o in range(n_obj):
                d_item[(n_o * batch_size): ((n_o + 1) * batch_size), n_o] = np.random.uniform(lb, rb, batch_size) # 0
            d_item = torch.Tensor(d_item).to(device)
            y_pred, w_pred, _ = net(epoch, d_item, 'train')
            
            y_best = y_pred[:, 0, :].unsqueeze(1)
            y_pred = y_pred[:, 1:, :]
            w_pred = w_pred[:, 1:, :]
            
            d1, d2 = PBI_cal(d_item, w_pred, y_pred)
            
            pty_item = penalty_item((y_pred - y_best + 1e-6), (d_item.unsqueeze(1) - y_best))
            constraint_1 = (-pty_item).squeeze() # (torch.sqrt(torch.tensor(2.0)) / 2. - pty_item).squeeze()
            constrain_2 = (pty_item - 1).squeeze()
            
            loss = d1 + rho * d2 - lmd * torch.exp(constraint_1 * constrain_2) # + 1 * torch.norm(y_pred, p=2)

            
            optimizer.zero_grad()
            loss.backward(torch.ones((batch_size*n_obj, w_pred.shape[1])).to(device), retain_graph=True)
            optimizer.step()
            

            time_e = datetime.datetime.now()
            model_time_list.append((time_e - time_s).microseconds)
            
            if epoch % num_iter != 0:
                continue
            prefs, x_pred, y_pred = evaluate(net, device, cfg.train, (epoch // num_iter), run_num)
            
            # batch selection
            x_sel, y_sel = paretoset_sel(d_item, prefs, x_pred, y_pred, y_best, net, num_bs, rho, lmd)
            x_new = torch.vstack((x_new, x_sel))
            y_new = torch.vstack((y_new, y_sel.to(device)))
            net_pre.fit(x_new, y_new)

            

            time_e = datetime.datetime.now()
            time_list.append((time_e - time_s).microseconds)
            print('avg time: ', np.mean(time_list), np.mean(model_time_list), model_time_list)
            # break
        # 


def PBI_cal(d_item, w_pred, y_pred):
    d_item_ = d_item.unsqueeze(1).repeat(1, y_pred.shape[1], 1)
    w_norm = torch.linalg.norm(w_pred, dim=-1)
    y_F = (y_pred.float() - d_item_)
    d1 = torch.sum(y_F * w_pred, dim=-1) / w_norm
    d2 = torch.linalg.norm(y_F - (d1.unsqueeze(-1) * w_pred) / w_norm.unsqueeze(-1), dim=-1)
    return d1, d2

def paretoset_sel(d_item, prefs, x_pred, y_pred, y_best, net, num_bs, rho, lmd):
    net.eval()
    n_ref = d_item.shape[0]
    n_spl = int(np.ceil(num_bs/n_ref)) + 2
    with torch.no_grad():
        x_pred = x_pred.unsqueeze(0).to(torch.float32)
        y_sel, _ = net.model_gp.predict(x_pred)
        y_sel = y_sel.to(torch.float32)
        d1, d2 = PBI_cal(d_item, prefs, y_sel)
        pty_item = penalty_item((y_sel - y_best + 1e-6), (d_item.unsqueeze(1) - y_best))
        constraint_1 = (torch.sqrt(torch.tensor(2.0)) / 2. - pty_item).squeeze() # (0 - pty_item).squeeze()
        constrain_2 = (pty_item - 1).squeeze()
        scores = d1 + rho * d2 - lmd * torch.exp(constraint_1 * constrain_2)
        idx_sub = torch.argsort(scores, dim=1)[:, :n_spl]
        idx_sub = idx_sub.cpu().detach().numpy()
        uq_idx, uq_iidx = np.unique(idx_sub, return_index=True)
        idx = np.random.choice(len(uq_idx), num_bs, replace=False)
        x_sel = x_pred.squeeze()[idx, :]
        y_sel = y_pred[idx, :]
        assert x_sel.shape[0] == y_sel.shape[0] and y_sel.shape[1] == d_item.shape[-1]

    return x_sel, torch.Tensor(y_sel)          

def add_perturbation_to_parameters(model, perturbation_std=0.15):
    with torch.no_grad():
        for param in model.parameters():
            # param.add_(torch.randn_like(param) * perturbation_std)
            param.add_(perturbation_std * torch.normal(0, 1, size=param.size(), device=param.device))
            

def evaluate(net, device, params, epoch, run_num):
#     x_pred
    m_obj = params.in_dim
    m_vars = params.out_dim
    save_path = params.save_path
    print('evaluate------')
    net.eval()
    with torch.no_grad():
        map_net = net.map_net
        pref = np.random.dirichlet(np.ones(m_obj), 200)
        pref = torch.Tensor(pref).to(device)
        x_pred = map_net(pref)
        F1 = get_problem(params.problem) # problem selection
        # F1 = DTLZ2(n_var=m_vars, n_obj=m_obj)
        y_pred = F1.evaluate(x_pred).cpu().detach().numpy()
        file_path = to_absolute_path(save_path + '/y-'+str(epoch)+'-'+str(run_num)+'.csv')
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
    os.makedirs(to_absolute_path(cfg.train.save_path), exist_ok=True)
    train_net(device=device, cfg=cfg)


if __name__ == '__main__':
    main()
