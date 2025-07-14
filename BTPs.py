# Abstract BTP class
class BTP:
    def __init__(self):
        pass
    
    def enroll(self, x):
        NotImplemented
    
    def auth(self, y, t):
        NotImplemented
        

# Implementation of [CVPR21] IronMask: Modular Architecture for Protecting Deep Face Template
# Abstract BTP class
class BTP:
    def __init__(self):
        pass
    
    def enroll(self, x):
        NotImplemented
    
    def auth(self, y, t):
        NotImplemented
        

# Implementation of [CVPR21] IronMask: Modular Architecture for Protecting Deep Face Template
import torch
from torch import nn
import torch.nn.functional as F
import math

@torch.no_grad()
def ironmask_decode(x, nonzeros):
    _, ind = x.abs().topk(nonzeros)
    data = x.gather(1, ind)
    data = (data > 0) * 2 - 1
    ret = torch.zeros_like(x, device = x.device)
    ret.scatter_(1, ind, data.to(ret.dtype))
    return ret

@torch.no_grad()
def GenC(dim, nonzeros):
    pos = torch.randperm(dim)[:nonzeros]
    coin = (torch.rand(1, nonzeros) > .5) * 2 - 1
    ret = torch.zeros(1, dim)
    ret[:, pos] = coin.to(ret.dtype)
    return ret

# It samples a random orthogonal matrix.
@torch.no_grad()
def GenP(dim):
    P = torch.empty((dim, dim))
    nn.init.orthogonal_(P)
    return P

# HRMG algorithm 
@torch.no_grad()
def HRMG(x,y,n=512):
    x = x.T
    y = y.T
    
    x = x/x.norm()
    y = y/y.norm()
    
    w=y-torch.mm(x.t(),y)*x
    w=w/(w.norm())
    cost=torch.mm(x.t(),y)
    sint=math.sqrt(1-cost**2)
    xw  =torch.cat((x,w),1)
    rot_2dim  =torch.FloatTensor([[cost,-sint],[sint,cost]])
    R   =torch.eye(n)-torch.mm(x,x.t())-torch.mm(w,w.t())+torch.mm(torch.mm(xw,rot_2dim),xw.t())
    return R

# IronMask 
class IronMask(BTP):
    def __init__(self, dim, nonzeros):
        super().__init__()
        self.dim = dim
        self.nonzeros = nonzeros
    
    def enroll(self, x):
        c = GenC(self.dim, self.nonzeros)
        P = GenP(self.dim)
        t = torch.mm(x, P.to(x.device).T)
        Q = HRMG(t, c.to(x.device))
        R = torch.mm(P.to(x.device).T, Q.to(x.device).T)
        return (R, c)
    
    def auth(self, y, t):
        R, c  = t
        cp = torch.mm(y, R).to(c.device)
        cp = ironmask_decode(cp, self.nonzeros)
        return ((c == cp).sum(dim = 1) == self.dim)


# Implementation of [CVPRW19] Significant Feature Based Representation for Template Protection
import numpy as np
import galois

# Table
# n = 127, k = 64, t = 10
# n = 127, k = 113, t = 2
# n = 255, k = 115, t = 21
# n = 512, k = 112, t = 59 

def enroll_batch_SF(t, rs, n=255,k=115):
#     rs = galois.ReedSolomon(n,k)
    GF = rs.field    
    coin = (torch.randn(1) > 0)
    dat1, idx1 = torch.topk(t,n//2+coin, largest = True, sorted = False)
    dat2, idx2 = torch.topk(t,n//2 + (coin^1), largest = False, sorted = False)

    dat1[:,:] = 1
    dat2[:,:] = 0

    dat = torch.cat((dat1,dat2),1)
    idx = torch.sort(torch.cat((idx1,idx2),1),1)

    c = dat.gather(1,idx[1]).to(dtype = torch.int16)    
    d = torch.tensor(rs.encode(np.array(GF.Random((t.shape[0],k))) ).astype(np.int16))

    return d, c^d, idx[0]


def auth_batch_SF(t, rs, tmp,n=255,k=115):
#     rs = galois.ReedSolomon(n,k)
    code, helper, h_i = tmp
    z = torch.zeros(t.shape)
    k = torch.topk(t,t.shape[1]//2,1,largest = True)[0][:,-1].reshape(t.shape[0],1)
    z[torch.where(t>=k)] = 1
    tmp = z.gather(1,h_i.expand_as(torch.zeros(z.shape[0],n))).to(torch.int16)
    right = torch.tensor(rs.encode(rs.decode(np.array(tmp^helper))).astype(np.int16))
    return right
    
    
class SigFeat(BTP):
    def __init__(self, dim, n,k, mode = "BCH"):
        super().__init__()
        self.dim = dim
        self.n = n 
        self.k = k
        self.ECC = None
        if mode == "BCH":
            self.ECC = galois.BCH(n, k)
        elif mode == "RS":
            self.ECC = galois.ReedSolomon(n, k)
        else:
            raise ValueError(f"Invalid Mode: {mode}")
    
    def enroll(self, x):
        x = F.normalize(x)
        tmp = enroll_batch_SF(x, self.ECC, self.n, self.k)
        return x, tmp
            
    def auth(self, x, tmp):
        return auth_batch_SF(x, self.ECC, tmp, self.n, self.k)   