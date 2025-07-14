import torch
from torch import nn
import torch.nn.functional as F

import math
from scipy.special import betainc
import numpy as np

# Safe Logarithm funtion of base 2
def safe_log2(x):
    if x == 0:
        return -torch.inf
    else:
        return torch.log2(x)
    
def safe_log2_np(x):
    if x == 0:
        return -np.inf
    else:
        return np.log2(x)    
    
# This calculates the logarithm of surface area of a hyperspherical cap    
def calculate_logarea(d, theta):
    return safe_log2_np(betainc((d-1)/2, 1/2, np.sin(theta * np.pi / 180) ** 2)) - 1
    
# Simple binary search
def find_theta(sec_level, dim):
    lb, ub = 0, 90
    
    for i in range(100):
        mid = (lb + ub) / 2
        curr = -calculate_logarea(dim, mid)
        # Smaller Theta
        if curr < sec_level:
            ub = mid
        # Larger Theta
        else:
            lb = mid
    
    return mid

# Make Tables for the upper bound of theta.
def make_table(sec_levels, dims):
    tab = dict()
    for dim in dims:
        tab[dim] = dict()
        for sec in sec_levels:
            theta = find_theta(sec, dim)
            tab[dim][sec] = theta
            
    return tab

import matplotlib.pyplot as plt

def plot_table(tab, sec_levels):
    symb = {
        128: "^",
        192: "o",
        256: "s",
    }
    dims = [dim for dim in tab]
    plt.figure(figsize = (8, 6))
    for level in sec_levels:
        data = [tab[dim][level] for dim in tab]
        plt.plot(dims, data, "--"+symb[level], linewidth = 2.5, markersize = 15, label ="%d-bit"%level)
        plt.xscale("log", base = 2)
        plt.xlabel("Dimension", fontsize = 25)
        plt.ylabel(r"$\theta(^{\circ})$", fontsize = 25)
        plt.xticks(fontsize = 25)
        plt.yticks(fontsize = 25)
        plt.grid("on", linewidth = 2)
    plt.legend(fontsize = 20)
    plt.show()
    
    
# Estimator     
        
# Sample over the boundary of the given strip.
@torch.no_grad()
def eqangle(vec, angle, num = 100):
    radius = math.sin(angle * math.pi / 180)
    height = math.cos(angle * math.pi / 180)
    
    r = F.normalize(torch.randn((num, len(vec.T)), device = vec.device))
    proj_r = r - vec.expand((num, len(vec.T))) * torch.matmul(vec, r.T).T / torch.matmul(vec, vec.T).item()
    
    return vec*height + radius * F.normalize(proj_r)
    
    
    
    
### ESTIMATOR ALGORTIHMS
# Useful arithmetics
def log_exp_add(a, b):
    if a < b:
        a, b = b, a
    if b == -np.inf:
        return a
    
    return a + safe_log2(1 + 2 ** (b-a))

def log_exp_sub(a, b):
    if a < b:
        a, b = b, a
        
    if b == -np.inf:
        return a
    else:
        return b + np.log2(2 ** (a-b) - 1)

    
# f/g functions    
f_new = lambda d, theta:  safe_log2(betainc((d-1)/2, 1/2, np.sin(theta * np.pi / 180) ** 2)) - 1    
g_new = lambda d, angle, eps: log_exp_sub(f_new(d, angle + eps), f_new(d, angle))

def safe_add(x, y):
    if x == -torch.inf or y == -torch.inf:
        return -torch.inf
    else:
        return x + y

# Transition Operator
# Move vec towards vec2 by "to_angle"
@torch.no_grad()
def transition(vec, vec2, to_angle):
    radius = math.sin(to_angle * math.pi / 180)
    height = math.cos(to_angle * math.pi / 180)
    
    proj_r = vec2 - vec.expand((vec2.size(0)), len(vec.T)) * torch.matmul(vec, vec2.T).T / torch.matmul(vec, vec.T).item()
    
    return vec * height + radius * F.normalize(proj_r)
        
    
# Reject Sampling for IM
# TODO: Generalization
def reject_sample(code, dim, alpha, angle, tg_num, N = 1<<12):
    mem = []
    cnt = 0
    while cnt < tg_num:
        x = eqangle(F.normalize(code), angle, N)
        idx = ((ironmask_decode(x, alpha) == code).sum(dim=1) == dim)
        cnt += idx.sum()
        mem.append(x[idx, :])
    return torch.cat(mem, dim = 0)[:tg_num]
    

# Estimator for IronMask
from BTPs import BTP, GenC, ironmask_decode
from tqdm.auto import tqdm

def estimator_IM(dim, alpha, n_divs = 180, M = 1<<12, n_iter = 5):
    divs = torch.linspace(0, 90, n_divs+1)[1:]
#     divs = torch.linspace(1, 0, n_divs+1)[1:].acos() * (180/math.pi)
    log_probs = []
    for _ in range(n_iter):
        ret_l = -torch.inf
        ret_u = -torch.inf

        code = GenC(dim, alpha).cuda()
        prev = divs[0]
        vec = eqangle(F.normalize(code), divs[0], M)
        prob = ((ironmask_decode(vec, alpha) == code).sum(dim=1) == dim).sum() / M
        log_prob = safe_log2(prob)


        prev = divs[0]
        
        tmpp = [log_prob]


        for i in tqdm(range(1, n_divs-1), "Security Estimator"):
            curr = divs[i]
            new = divs[i+1]
            eps1 = curr-prev
            eps2 = new-curr

            tran_vec = transition(F.normalize(code), vec, curr)
            tran_idx = ((ironmask_decode(tran_vec, alpha) == code).sum(dim=1) == dim)

            # Pr[E_i | E_i-1]
            tran_prob = tran_idx.sum() / (vec.size(0) + 1e-12)

            log_prob = safe_add(log_prob, safe_log2(tran_prob))


    #         vec[tran_idx] = tran_vec[tran_idx]
            if (~tran_idx).sum() > 0 and log_prob > -12:
                vec[tran_idx] = tran_vec[tran_idx]
                tmp = reject_sample(code, dim, alpha, curr, (~tran_idx).sum())
                vec[~tran_idx] = reject_sample(code, dim, alpha, curr, (~tran_idx).sum())
            else:
                vec = tran_vec[tran_idx]

            lower = g_new(512, prev, eps1)
            upper = g_new(512, new, eps2)
            lower = safe_add(lower, log_prob)
            upper = safe_add(upper, log_prob)

            ret_u = log_exp_add(ret_u, upper)
            ret_l = log_exp_add(ret_l, lower)

            prev = curr
            tmpp.append(log_prob)
            
        tmpp = torch.Tensor(tmpp)
                        
        log_probs.append(tmpp)
                
    log_probs = torch.cat(log_probs, dim = 0).reshape(n_iter, -1)
    log_probs = torch.log2(torch.mean(2**log_probs, dim = 0))
        
    ret_l = -torch.inf
    ret_u = -torch.inf
    prev = 0
    
    for i, log_prob in enumerate(log_probs):
        if i == 0:
            continue
        curr = divs[i]
        new = divs[i+1]
        eps1 = curr-prev
        eps2 = new-curr        
        lower = g_new(512, prev, eps1)
        upper = g_new(512, new, eps2)
        lower = safe_add(lower, log_prob)
        upper = safe_add(upper, log_prob)

        ret_u = log_exp_add(ret_u, upper)
        ret_l = log_exp_add(ret_l, lower)       
        prev = curr
        
    return (ret_l, ret_u)

from BTPs import BTP, SigFeat

# Estimator for SigFeat
def reject_sample_SF(BTP, t, tmp, angle, tg_num, N = 1<<10):
    mem = []
    cnt = 0
    while cnt < tg_num:
        x = eqangle(F.normalize(t), angle, N)
        c_x = BTP.auth(x, tmp)
        idx = ((c_x == tmp[0]).sum(dim=1) == BTP.n)
        cnt += idx.sum()
        mem.append(x[idx, :])
    return torch.cat(mem, dim = 0)[:tg_num]

def estimator_SF(BTP, n_divs = 270, M = 1<<10):
    divs = torch.linspace(0, 90, n_divs+1)[1:]
    ret_l = -np.inf
    ret_u = -np.inf
    
    print("[LOG] GENERATE Target Template...")
    t = F.normalize(torch.randn((1,512)))
    _, tmp = BTP.enroll(t)
    
    prev = divs[0]
    vec = eqangle(F.normalize(t), divs[0], M)
    
    print("[LOG] Initial Setup...")
    c_x = BTP.auth(vec, tmp)
    prob = ((tmp[0] == c_x).sum(dim = 1) == BTP.n).sum() / M
    log_prob = safe_log2(prob)
    
    prev = divs[0]
    
    for i in tqdm(range(1, n_divs-1), "Security Estimator"):
        curr = divs[i]
        new = divs[i+1]
        eps1 = curr-prev
        eps2 = new-curr
        
        tran_vec = transition(F.normalize(t), vec, curr)
        c_x = BTP.auth(tran_vec, tmp)
        tran_idx = ((tmp[0] == c_x).sum(dim=1) == BTP.n)
        
        # Pr[E_i | E_i-1]
        tran_prob = tran_idx.sum() / (vec.size(0) + 1e-12)
        
        log_prob = safe_add(log_prob, safe_log2(tran_prob))
                
        if (~tran_idx).sum() > 0 and log_prob > -12:
            vec[tran_idx] = tran_vec[tran_idx]
            vec[~tran_idx] = reject_sample_SF(BTP, t, tmp, curr, (~tran_idx).sum())
        else:
            vec = tran_vec[tran_idx]
        
        lower = g_new(512, prev, eps1)
        upper = g_new(512, new, eps2)
        lower = safe_add(lower, log_prob)
        upper = safe_add(upper, log_prob)

        ret_u = log_exp_add(ret_u, upper)
        ret_l = log_exp_add(ret_l, lower)
        
        prev = curr
        
    return ret_l, ret_u
    
    
    
    
    
    
    
    
    
    
    
    