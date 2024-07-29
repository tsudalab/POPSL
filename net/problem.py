import torch
import numpy as np
from pymoo.problems.many.wfg import WFG1


def get_problem(name, *args, **kwargs):
    name = name.lower()
    
    PROBLEM = {
        'f1': F1,
        'f2': F2,
        'f3': F3,
        'f4': F4,
        'f5': F5,
        'f6': F6,
        'vlmop1': VLMOP1,
        'vlmop2': VLMOP2,
        'vlmop3': VLMOP3,
        'dtlz1': DTLZ1,
        'dtlz2': DTLZ2,
        'wfg1': WFG1,
        'dtlz2_5': DTLZ2_5,
        'dtlz2_2': DTLZ2_2,
        'dtlz2_10': DTLZ2_10,
        'dtlz2_8': DTLZ2_8,
        'dtlz7': DTLZ7,
        'zdt3': ZDT3,
        'zdt2': ZDT2,
        'zdt1':ZDT1,
        're5': RE5,
 }

    if name not in PROBLEM:
        raise Exception("Problem not found.")
    
    return PROBLEM[name](*args, **kwargs)

class F1():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 =  0.0
        count1 = count2 =  0.0
            
        for i in range(2,n+1):
            yi    = x[:,i-1] - torch.pow(2 * x[:,0] - 1, 2)
            yi    = yi * yi
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
        

class F2():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 =  0.0
        count1 = count2 =  0.0
            
        for i in range(2,n+1):
            theta = 1.0 + 3.0*(i-2)/(n - 2)
            yi    = x[:,i-1] - torch.pow(x[:,0], 0.5*theta)
            yi    = yi * yi
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0/count1 * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    
class F3():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0.0
        count1 = count2 = 0.0
        
        for i in range(2,n+1):
            xi = x[:,i-1]
            yi = xi - (torch.sin(4.0*np.pi* x[:,0]  + i*np.pi/n) + 1) / 2
            yi = yi * yi 
            
            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0])) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
    
class F4():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - 0.8 * x[:,0] * torch.sin(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8* x[:,0] * torch.cos(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
class F5():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - 0.8 * x[:,0] * torch.sin(4.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:,0] * torch.cos((4.0*np.pi*x[:,0] + i*np.pi/n)/3)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
class F6():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
      
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = sum2 = 0
        count1 = count2 = 0
        
        for i in range(2,n+1):
            xi = -1.0 + 2.0*x[:,i-1]
 
            if i % 2 == 0:
                yi = xi - (0.3 * x[:,0] ** 2 * torch.cos(12.0*np.pi*x[:,0] + 4 *i*np.pi/n) + 0.6 * x[:,0]) * torch.sin(6.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - (0.3 * x[:,0] ** 2 * torch.cos(12.0*np.pi*x[:,0] + 4 *i*np.pi/n) + 0.6 * x[:,0]) * torch.cos(6.0*np.pi*x[:,0] + i*np.pi/n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0
       
        f1 = (1 + 1.0/count1  * sum1 ) * x[:,0]  
        f2 = (1 + 1.0/count2 * sum2 ) * (1.0 - torch.sqrt(x[:,0] / (1 + 1.0/count2 * sum2 ))) 
        
        objs = torch.stack([f1,f2]).T
        
        return objs
    
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



class VLMOP1():
    def __init__(self, n_dim = 1):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([-2.0]).float()
        self.ubound = torch.tensor([4.0]).float()
        self.nadir_point = [4, 4]
       
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        f1 = torch.pow(x[:,0], 2)
        f2 = torch.pow(x[:,0] - 2, 2)
     
        objs = torch.stack([f1,f2]).T
        
        return objs
    

class VLMOP2():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]).float()
        self.ubound = torch.tensor([2.0, 2.0,2.0, 2.0,2.0, 2.0]).float()
        self.nadir_point = [1, 1]
       
    def evaluate(self, x):
        
        n = x.shape[1]
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        f1 = 1 - torch.exp(-torch.sum((x - 1 / np.sqrt(n))**2, axis = 1))
        f2 = 1 - torch.exp(-torch.sum((x + 1 / np.sqrt(n))**2, axis = 1))
     
        objs = torch.stack([f1,f2]).T
        
        return objs
    

class VLMOP3():
    def __init__(self, n_dim = 2):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([-3.0, -3.0]).float()
        self.ubound = torch.tensor([3.0, 3.0]).float()
        self.nadir_point = [10,60,1]
       
    def evaluate(self, x):
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        x1, x2 = x[:, 0], x[:, 1]
    
        f1 = 0.5 * (x1 ** 2 + x2 ** 2) + torch.sin(x1 ** 2 + x2 ** 2)
        f2 = (3 * x1 - 2 * x2 + 4) ** 2 / 8 + (x1 - x2 + 1) ** 2 / 27 + 15
        f3 = 1 / (x1 ** 2 + x2 ** 2 + 1) - 1.1 * torch.exp(-x1 ** 2 - x2 ** 2)
     
        objs = torch.stack([f1,f2,f3]).T
        
        return objs
    
class DTLZ1():
    def __init__(self, n_dim = 4):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
       
        
    def evaluate(self, x):
        n = x.shape[1]
        k = self.n_dim - self.n_obj + 1
        # sum1 = torch.sum(torch.stack([(torch.pow(x[:,i]-0.5,2) - torch.cos(20 * np.pi * (x[:,i]-0.5))) for i in range(2,n)]), axis = 0)
        # g = 100 * (k + sum1)

        X_, X_M = x[:, :(self.n_obj - 1)], x[:, (self.n_obj - 1):]
        # print(X_, X_M)
        g = 100 * (k + torch.sum(torch.pow(X_M - 0.5, 2) - torch.cos(20 * np.pi * (X_M - 0.5)), axis=1))

        # f = []
        # for i in range(0, self.n_obj):
        #     _f = 0.5 * (1 + g)
        #     _f *= anp.prod(X_[:, :X_.shape[1] - i], axis=1)
        #     if i > 0:
        #         _f *= 1 - X_[:, X_.shape[1] - i]
        #     f.append(_f)

        f1 = 0.5 * (1 + g) * torch.prod(X_[:, :X_.shape[1]], axis=1)
        f2 = 0.5 * (1 + g) * (1 - X_[:, X_.shape[1] - 1])
        
        # f1 = 0.5 * (1 + g) * x[:,0] * x[:,1] * x[:,2]
        # # f2 = 0.5 * (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.sin(x[:,1]*np.pi/2)
        # f2 = 0.5 * (1 + g) * (1 - x[:,0])
        print(x)
        
        objs = torch.stack([f1,f2]).T
        
        return objs

class DTLZ2():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1, 1]
       
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = torch.sum(torch.stack([torch.pow(x[:,i]-0.5,2) for i in range(2,n)]), axis = 0)
        g = sum1
        
        f1 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2)
        f2 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.sin(x[:,1]*np.pi/2)
        f3 = (1 + g) * torch.sin(x[:,0]*np.pi/2)
        
        objs = torch.stack([f1,f2, f3]).T
        
        return objs

class DTLZ2_2():
    def __init__(self, n_dim = 4):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]
       
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = torch.sum(torch.stack([torch.pow(x[:,i]-0.5,2) for i in range(2,n)]), axis = 0)
        g = sum1
        
        f1 = (1 + g) * torch.cos(x[:,0]*np.pi/2) # * torch.cos(x[:,1]*np.pi/2)
        # f2 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.sin(x[:,1]*np.pi/2)
        f2 = (1 + g) * torch.sin(x[:,0]*np.pi/2)
        
        objs = torch.stack([f1, f2]).T
        # print(objs)
        
        return objs
    
class DTLZ2_5():
    def __init__(self, n_dim = 6):
        self.n_dim = n_dim
        self.n_obj = 4
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1, 1, 1]
       
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = torch.sum(torch.stack([torch.pow(x[:,i]-0.5,2) for i in range(2,n)]), axis = 0)
        g = sum1
        
        # f1 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2)
        # f2 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.sin(x[:,3]*np.pi/2)
        # f3 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.sin(x[:,2]*np.pi/2)
        # f4 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.sin(x[:,1]*np.pi/2)
        # f5 = (1 + g) * torch.sin(x[:,0]*np.pi/2)
        f1 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) 
        f2 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.sin(x[:,2]*np.pi/2)
        f3 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.sin(x[:,1]*np.pi/2)
        f4 = (1 + g) * torch.sin(x[:,0]*np.pi/2)
        # f5 = (1 + g) * torch.sin(x[:,0]*np.pi/2)
        
        objs = torch.stack([f1,f2, f3, f4]).T
        
        return objs
    
class DTLZ2_10():
    def __init__(self, n_dim = 20):
        self.n_dim = n_dim
        self.n_obj = 10
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = torch.ones(self.n_obj)
       
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = torch.sum(torch.stack([torch.pow(x[:,i]-0.5,2) for i in range(2,n)]), axis = 0)
        g = sum1
        
        f1 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.cos(x[:,4]*np.pi/2) * torch.cos(x[:,5]*np.pi/2) * torch.cos(x[:,6]*np.pi/2) * torch.cos(x[:,7]*np.pi/2) * torch.cos(x[:,8]*np.pi/2)
        f2 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.cos(x[:,4]*np.pi/2) * torch.cos(x[:,5]*np.pi/2) * torch.cos(x[:,6]*np.pi/2) * torch.cos(x[:,7]*np.pi/2) * torch.sin(x[:,8]*np.pi/2)
        f3 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.cos(x[:,4]*np.pi/2) * torch.cos(x[:,5]*np.pi/2) * torch.cos(x[:,6]*np.pi/2) * torch.sin(x[:,7]*np.pi/2)
        f4 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.cos(x[:,4]*np.pi/2) * torch.cos(x[:,5]*np.pi/2) * torch.sin(x[:,6]*np.pi/2)
        f5 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.cos(x[:,4]*np.pi/2) * torch.sin(x[:,5]*np.pi/2)
        f6 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.sin(x[:,4]*np.pi/2)
        f7 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.sin(x[:,3]*np.pi/2)
        f8 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.sin(x[:,2]*np.pi/2)
        f9 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.sin(x[:,1]*np.pi/2)
        f10 = (1 + g) * torch.sin(x[:,0]*np.pi/2)
        
        objs = torch.stack([f1,f2, f3, f4, f5, f6, f7, f8, f9, f10]).T
        
        return objs
    
class DTLZ2_8():
    def __init__(self, n_dim = 20):
        self.n_dim = n_dim
        self.n_obj = 10
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = torch.ones(self.n_obj)
       
        
    def evaluate(self, x):
        n = x.shape[1]
       
        sum1 = torch.sum(torch.stack([torch.pow(x[:,i]-0.5,2) for i in range(2,n)]), axis = 0)
        g = sum1
        
        f1 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.cos(x[:,4]*np.pi/2) * torch.cos(x[:,5]*np.pi/2) * torch.cos(x[:,6]*np.pi/2)
        f2 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.cos(x[:,4]*np.pi/2) * torch.cos(x[:,5]*np.pi/2) * torch.sin(x[:,6]*np.pi/2)
        f3 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.cos(x[:,4]*np.pi/2) * torch.sin(x[:,5]*np.pi/2)
        f4 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.cos(x[:,3]*np.pi/2) * torch.sin(x[:,4]*np.pi/2)
        f5 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.cos(x[:,2]*np.pi/2) * torch.sin(x[:,3]*np.pi/2)
        f6 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.cos(x[:,1]*np.pi/2) * torch.sin(x[:,2]*np.pi/2)
        f7 = (1 + g) * torch.cos(x[:,0]*np.pi/2) * torch.sin(x[:,1]*np.pi/2)
        f8 = (1 + g) * torch.sin(x[:,0]*np.pi/2)
        
        objs = torch.stack([f1,f2, f3, f4, f5, f6, f7, f8]).T
        
        return objs
    
class DTLZ7():
    def __init__(self, n_dim = 10, n_obj=3):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.k = n_dim - n_obj + 1
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = torch.ones(self.n_obj)

    def evaluate(self, x):
        # f = []
        # for i in range(0, self.n_obj - 1):
        #     f.append(x[:, i])
        # f = anp.column_stack(f)
        f1 = x[:, 0]

        f2 = x[:, 1]

        g = 1 + 9 / self.k * torch.sum(x[:, -self.k:], axis=1)
        # print(g.shape)
        h = self.n_obj - (f1 / (1 + g) * (1 + torch.sin(3 * np.pi * f1))) - (f2 / (1 + g) * (1 + torch.sin(3 * np.pi * f2)))
        f3 = (1 + g) * h

        objs = torch.stack([f1,f2, f3]).T

        return objs
    
class ZDT1():
    def __init__(self, n_dim=4, n_obj=2):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        f_1 = x[:, 0]
        g = 1 + 9.0 / (self.n_dim - 1) * torch.sum(x[:, 1:], axis=1)
        f_2 = g * (1 - torch.pow((f_1 / g), 0.5))
        # f_1 = x[:, 0]
        # g = 1.0 + 9.0 * torch.sum(x[:, 1:], axis=1) / (self.n_dim - 1)
        # h = 1.0 - torch.sqrt(f_1 / g) - f_1 / g * torch.sin(10.0 * np.pi * f_1)
        # h = 1 - (f_1 / g)**2
        # f_2 = g * h

        objs = torch.stack([f_1,f_2]).T

        return objs
    
class ZDT2():
    def __init__(self, n_dim=30, n_obj=2):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        f_1 = x[:, 0]
        g = 1.0 + 9.0 * torch.sum(x[:, 1:], axis=1) / (self.n_dim - 1)
        # h = 1.0 - torch.sqrt(f_1 / g) - f_1 / g * torch.sin(10.0 * np.pi * f_1)
        h = 1 - (f_1 / g)**2
        f_2 = g * h

        objs = torch.stack([f_1,f_2]).T

        return objs
    
class ZDT3():
    def __init__(self, n_dim=6, n_obj=2):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        f_1 = x[:, 0]
        g = 1.0 + 9.0 * torch.sum(x[:, 1:], axis=1) / (self.n_dim - 1)
        h = 1.0 - torch.sqrt(f_1 / g) - f_1 / g * torch.sin(10.0 * np.pi * f_1)
        # h = 1 - (f_1 / g)**2
        f_2 = g * h

        objs = torch.stack([f_1,f_2]).T

        return objs
    
class WFG1():
    def __init__(self, n_dim=20, n_obj=5, k=None):
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1, 1, 1, 1]
        self.xu = 2 * torch.arange(1, n_dim + 1).float()
        if k:
            self.k = k
        else:
            if self.n_obj == 2:
                self.k = 4
            else:
                self.k = 2 * (n_obj - 1)

    def t1(self, x, n, k):
        x[:, k:n] = _transformation_shift_linear(x[:, k:n], torch.Tensor([0.35]))
        return x

    def t2(self, x, n, k):
        x[:, k:n] = _transformation_bias_flat(x[:, k:n], torch.Tensor([0.8]), torch.Tensor([0.75]), torch.Tensor([0.85]))
        return x

    def t3(self, x, n):
        x[:, :n] = _transformation_bias_poly(x[:, :n], 0.02)
        return x

    def t4(self, x, m, n, k):
        w = torch.arange(2, 2 * n + 1, 2).float()
        gap = k // (m - 1)
        t = []
        for m in range(1, m):
            _y = x[:, (m - 1) * gap: (m * gap)]
            _w = w[(m - 1) * gap: (m * gap)]
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(x[:, k:n], w[k:n]))
        return torch.column_stack(t).float()

    def evaluate(self, x):
        if x.device.type == 'cuda': 
            self.xu = self.xu.to(torch.device("cuda"))
        y = x / self.xu
        y = self.t1(y, self.n_dim, self.k)
        # print('========y t1: ', y)
        y = self.t2(y, self.n_dim, self.k)
        # print('========y t2: ', y)
        y = self.t3(y, self.n_dim)
        # print('========y t3: ', y)
        y = self.t4(y, self.n_obj, self.n_dim, self.k)
        # print('========y t4: ', y)
        # print('=======WFG1=========', y)
        return y


########################## WFG TRANSFORMATION ###########################
def _transformation_shift_linear(value, shift=torch.Tensor([0.35])):
    if value.device.type == 'cuda': 
        shift = shift.float().to(torch.device("cuda"))
    return correct_to_01(torch.abs(value - shift) / torch.abs(torch.floor(shift - value) + shift))

def _transformation_bias_flat(y, a, b, c):
    if y.device.type == 'cuda': 
        a = a.to(torch.device("cuda"))
        b = b.to(torch.device("cuda"))
        c = c.to(torch.device("cuda"))
        o = torch.ones(1).float().to(torch.device("cuda"))
        z = torch.zeros(1).float().to(torch.device("cuda"))
    ret = a + torch.minimum(z, torch.floor(y - b)) * (a * (b - y) / b) \
          - torch.minimum(z, torch.floor(c - y)) * ((o - a) * (y - c) / (o - c))
    return correct_to_01(ret)

def _transformation_bias_poly(y, alpha):
    # alpha = alpha.to(torch.device("cuda"))
    # y = y.pow(alpha)
    return correct_to_01(torch.pow(y.detach(), alpha))
    # return correct_to_01(y)

########################## WFG TRANSFORMATION ###########################
def _reduction_weighted_sum(y, w):
    if y.device.type == 'cuda': 
        w = w.float().to(torch.device("cuda"))
    # return correct_to_01(torch.dot(y, w) / w.sum())
    sum_dot = torch.sum(y.mul(w), dim=1).float()
    # print('shape: ', y.shape, w.shape, sum_dot.shape)
    return correct_to_01(sum_dot / w.sum())
    

########################## WFG UTIL ##############################    
def correct_to_01(X, epsilon=None):
    if epsilon is None:
        epsilon = torch.Tensor([1.0e-10])
    if X.device.type == 'cuda': 
        epsilon = epsilon.float().to(torch.device("cuda"))
    X[torch.logical_and(X < 0, X >= 0 - epsilon)] = 0
    X[torch.logical_and(X > 1, X <= 1 + epsilon)] = 1
    return X
