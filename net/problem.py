import torch
import numpy as np
from pymoo.problems.many.wfg import WFG1


def get_problem(name, *args, **kwargs):
    name = name.lower()
    
    PROBLEM = {
        'dtlz2_4': DTLZ2_4,
        'dtlz5': DTLZ5,
        'zdt3': ZDT3,
        're5': RE5,
        'custom': Custom,
 }

    if name not in PROBLEM:
        raise Exception("Problem not found.")
    
    return PROBLEM[name](*args, **kwargs)

    
class RE5():
    def __init__(self, n_dim = 4):
        
      
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0, 0, 0, 0]).float()
        self.ubound = torch.tensor([1, 1, 1, 1]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]
        
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
      
        xAlpha = x[:,0]
        xHA = x[:,1]
        xOA = x[:,2]
        xOPTT = x[:,3]
 
        # f1 (TF_max)
        f1 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f2 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f3 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)
 
         
        objs = torch.stack([f1,f2,f3]).T
        
        return objs

    
class DTLZ2_4():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 4
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
       
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = torch.sum(torch.stack([torch.pow(x[:,i]-0.5,2) for i in range(self.n_obj-1,n)]), axis = 0)
        g = sum1
        
        f1 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) 
        f2 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.sin(x[:,2]*np.pi/2)
        f3 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.sin(x[:,1]*np.pi/2)
        f4 = (1 + g) * torch.sin(x[:,0]*np.pi/2)
        
        objs = torch.stack([f1,f2, f3, f4]).T
        
        return objs
    
    
class DTLZ5():
    def __init__(self, n_dim = 10, n_obj=3):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.k = n_dim - n_obj + 1
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()

    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = torch.sum(torch.stack([torch.pow(x[:,i]-0.5,2) for i in range(self.n_obj-1,n)]), axis = 0)
        g = sum1
        theta = 1. / (2 * (1 + g))
        f1 = (1 + g) * torch.cos(x[:,0] * np.pi/2) * torch.cos(theta * (1 + 2 * g * x[:,1]) * np.pi/2)
        f2 = (1 + g) * torch.cos(x[:,0] * np.pi/2) * torch.sin(theta * (1 + 2 * g * x[:,1]) * np.pi/2)
        f3 = (1 + g) * torch.sin(x[:,0] * np.pi/2)

        objs = torch.stack([f1,f2, f3]).T

        return objs
    
    
class ZDT3():
    def __init__(self, n_dim=6, n_obj=2):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()

    def evaluate(self, x):
        f_1 = x[:, 0]
        g = 1.0 + 9.0 * torch.sum(x[:, 1:], axis=1) / (self.n_dim - 1)
        h = 1.0 - torch.sqrt(f_1 / g) - f_1 / g * torch.sin(10.0 * np.pi * f_1)
        # h = 1 - (f_1 / g)**2
        f_2 = g * h

        objs = torch.stack([f_1,f_2]).T

        return objs
    
class Custom():
    def __init__(self, n_dim=1, n_obj=1):
        '''
        provide the dimension of variable and objective
        '''
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()

    def evaluate(self, x):
        '''
        set your problem here and returh the objectives
        return: objs -[n_samples, n_obj]
        '''

        return 
    

