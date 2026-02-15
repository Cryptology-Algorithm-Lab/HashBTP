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
    
### ESTIMATOR ALGORTIHMS

# Safe Functions
def safe_log2(x: torch.Tensor) -> torch.Tensor:
    # x can be on CPU or CUDA
    neg_inf = torch.tensor(-torch.inf, device=x.device, dtype=x.dtype)
    return torch.where(x == 0, neg_inf, torch.log2(x))
    
def safe_add(x, y):
    if x == -torch.inf or y == -torch.inf:
        return -torch.inf
    else:
        return x + y

# Useful Arithmetics
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
        return b + safe_log2(2**(a-b) - 1)
    
# f/g functions    
def f_new(d, theta):
    return safe_log2_np(betainc((d-1)/2, 1/2, np.sin(theta * np.pi / 180) ** 2)) -1

def g_new(d, theta_start, theta_end):
    return log_exp_sub(f_new(d, theta_end), f_new(d, theta_start))

# Vector Sampling & Transition
@torch.no_grad()
def eqangle(vec, angle, num = 100):
    radius = math.sin(angle * math.pi / 180)
    height = math.cos(angle * math.pi / 180)
    r = F.normalize(torch.randn((num, len(vec.T)), device = vec.device))
    proj_r = r - vec.expand((num, len(vec.T))) * torch.matmul(vec, r.T).T / torch.matmul(vec, vec.T).item()
    return F.normalize(vec*height + radius * F.normalize(proj_r))
    
@torch.no_grad()
def transition(vec, vec2, to_angle):
    radius = math.sin(to_angle * math.pi / 180)
    height = math.cos(to_angle * math.pi / 180)
    
    proj_r = vec2 - vec.expand((vec2.size(0)), len(vec.T)) * torch.matmul(vec, vec2.T).T / torch.matmul(vec, vec.T).item()
    
    return F.normalize(vec * height + radius * F.normalize(proj_r))
        
        
# Rejection Sampling
@torch.no_grad()
def reject_sample(BTP, t, tmp, angle, tg_num, N=1<<12):
    mem = []
    cnt = 0
    while cnt < tg_num:
        x = eqangle(t, angle, N)
        idx = BTP.auth(x, tmp)
        cnt += idx.sum()
        mem.append(x[idx, :])
    return torch.cat(mem, dim = 0)[:tg_num]    
    
# Estimator
@torch.no_grad()
def estimator(BTP, device, n_divs=720, M=4096, n_iter=5, verbose = False):
    divs = torch.linspace(0, 1, n_divs+1)[1:]
    divs = divs.asin() * 180 / torch.pi
    ret_l = -np.inf
    ret_u = -np.inf
    
    if verbose: print("[LOG] Security Estimator Start...")    
    for idx in range(n_iter):
        # Initialize
        t = F.normalize(torch.randn((1,512)))
        tmp = BTP.enroll(t)    
        prob_mem = []
        t = t.to(device)
        tmp = [z.to(device) for z in tmp]        
        
        prev = divs[0]                
        vec = eqangle(t, divs[0], M)
        ret = BTP.auth(vec, tmp)
        prob = ret.sum() / M
        log_prob = torch.tensor(0., device=device)
        curr_prob_mem = [log_prob]
            
        for i in range(1, n_divs-1):  
            # Step 1: Make Transition
            tran_vec = transition(t, vec, divs[i])
            
            # Step 2: Compute Markov Prob            
            tran_idx = BTP.auth(tran_vec, tmp)            
            
            # Pr[E_i | E_i-1]
            tran_prob = tran_idx.sum() / vec.size(0)
            
            # Step 3: Update log_prob and record it.
            log_prob = safe_add(log_prob, safe_log2(tran_prob))
            curr_prob_mem.append(log_prob)

            # Step 4: If needed, do a rejection sampling for missing indices
            if (~tran_idx).sum() > 0 and log_prob > -15:
                vec[tran_idx] = tran_vec[tran_idx]
                vec[~tran_idx] = reject_sample(BTP, t, tmp, divs[i], (~tran_idx).sum(), N = M)
            else:
                vec = tran_vec[tran_idx]

        prob_mem.append(curr_prob_mem)
            
    if verbose: print("[LOG] Prob Collection Done. Averaging Results...")
    prob_mem = torch.FloatTensor(prob_mem)
    prob_mem = torch.log2(torch.mean(2**prob_mem, dim = 0))
    
    ret_l = -torch.inf
    ret_u = -torch.inf
    full_divs = torch.cat([torch.tensor([0.0]), divs])

    prev = divs[0]
    
    for i, log_prob in enumerate(prob_mem):
        if i ==0:
            continue
        curr = divs[i]
        new = divs[i+1]
        eps1 = curr-prev
        eps2 = new-curr        
        lower = g_new(512, prev, prev+eps1)
        upper = g_new(512, new, new+eps2)
        lower = safe_add(lower, log_prob)
        upper = safe_add(upper, log_prob)

        ret_u = log_exp_add(ret_u, upper)
        ret_l = log_exp_add(ret_l, lower)       
        prev = curr
        
    return (ret_l, ret_u)    
    
   
