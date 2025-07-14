import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from tqdm.auto import tqdm

@torch.no_grad()
def benchmark_plain(feats, target_far = None):
    left, right = feats[::2], feats[1::2]
    n_pairs, dim = left.size(0), left.size(1)
    score = (F.cosine_similarity(left, right).acos() * (180 / torch.pi)).reshape(20, -1)
    true_pair, false_pair = score[::2].reshape(-1), score[1::2].reshape(-1)
    
    stepsize = n_pairs // 20
    issame = []
    
    for i in range(n_pairs):
        if (n_pairs // stepsize) % 2 == 0:
            issame += [True]
        else:
            issame += [False]
    
    
    if target_far == None:
        thxs = torch.linspace(0, 90, 9001)
        best_thx = 0
        best_ACC = 0
        best_TAR = 0
        best_FAR = 0

        for thx in thxs:
            pred = score < thx
            TA = (true_pair < thx).sum()
            TR = (false_pair > thx).sum()
            FA = (false_pair < thx).sum()
            FR = (true_pair > thx).sum()
            ACC = (TA + TR) / n_pairs

            TAR = TA / (n_pairs / 2)
            FAR = FA / (n_pairs / 2)

            if best_ACC <= ACC:
                best_ACC = ACC
                best_TAR = TAR
                best_FAR = FAR
                best_thx = thx


        return best_thx, best_ACC, best_TAR, best_FAR
    
    else:
        thxs = torch.linspace(0, 90, 9001)
        
        p_TAR, p_FAR, p_thx, p_ACC = 0, 0, 0, 0
        
        for thx in thxs:
            pred = score < thx
            TA = (true_pair < thx).sum()
            TR = (false_pair > thx).sum()
            FA = (false_pair < thx).sum()
            FR = (true_pair > thx).sum()
            ACC = (TA + TR) / n_pairs

            TAR = TA / (n_pairs / 2)
            FAR = FA / (n_pairs / 2)
            
            if FAR > target_far:
                return p_thx, p_ACC, p_TAR, p_FAR
            else:
                p_TAR, p_FAR, p_thx, p_ACC = TAR, FAR, thx, ACC
            
@torch.no_grad()
def benchmark_IM(feats, BTP, batch_size):
    left, right = feats[::2], feats[1::2]
    n_pairs = left.size(0)
    stepsize = n_pairs // 20
    issame = []
    
    for i in range(n_pairs):
        if (n_pairs // stepsize) % 2 == 0:
            issame += [True]
        else:
            issame += [False]
    
    issame = torch.BoolTensor(issame)
    
    pred1, pred2 = [], []
    ba = 0
    for i in tqdm(range(n_pairs), "IronMask"):        
        # Left: Enroll, Right: Auth
        blk = left[i:i+1]
        tmps = BTP.enroll(blk)
        ret1 = BTP.auth(right[i:i+1], tmps)
        pred1.append(ret1)
        
        # Left: Auth, Right: Enroll
        blk = right[i:i+1]
        tmps = BTP.enroll(blk)
        ret2 = BTP.auth(left[i:i+1], tmps)
        pred2.append(ret2)
        
    pred1 = torch.cat(pred1)
    pred2 = torch.cat(pred2)
    
    TA = (pred1 & issame).sum() + (pred2 & issame).sum()
    TR = (~pred1 & ~issame).sum() + (~pred2 & ~issame).sum()
    FA = (pred1 & ~issame).sum() + (pred2 & ~issame).sum()
    FR = (~pred1 & issame).sum() + (~pred2 & issame).sum()
    
    TAR = TA / n_pairs
    FAR = FA / n_pairs
    ACC = (TA + TR) / (2 * n_pairs)
    
    return TAR, FAR, ACC

@torch.no_grad()
def benchmark_SF(dataset, BTP, batch_size=500,n = 255, k = 115):
#     rs = galois.ReedSolomon(n,k)
    print("DECODE TEST START")
    # Generate issame list
    left, right = dataset[::2], dataset[1::2]
    len_data = left.size(0)
    issame = []
    n, k = BTP.n, BTP.k
    
    for idx in range(len_data):
        if (idx // (len_data // 20)) % 2 == 0:
            issame += [True]
        else:
            issame += [False]
            
    issame = np.array(issame)
    # Benchmark start
    predict = []
    
    for idx in tqdm(range(len_data//batch_size)):
        # encoding
        left_data = left[batch_size*idx:batch_size*(idx+1)]
        _, tmp = BTP.enroll(left_data)
        code_left = tmp[0]

        # decoding
        right_data = right[batch_size*idx:batch_size*(idx+1)]
        code_right = BTP.auth(right_data, tmp)
        
        # Scoring
        predict += list((code_left == code_right).all(axis = 1))

    predict = np.array(predict)
    
    
    # Computing Score
    A = (issame & predict).sum()
    C = (~issame & predict).sum()
    D = (issame & ~predict).sum()
    B = (~issame & ~predict).sum()
    
    
    TAR = A / (A+D)
    FAR = C / (B+C)  
    ACC = (issame == predict).sum() / len_data
    
    return TAR, FAR, ACC

@torch.no_grad()
def estimator_ideal(dim, thx, n_divs = 3000):
    eps = thx / n_divs
    divs = (1 + torch.arange(n_divs - 1)) * eps
    
    for i in divs:
        ret_u = None
        ret_r = None
        
    orig = None
        
    return -let_u, -let_r, -orig


@torch.no_grad()
def upper_bound_estim_tab(feats, sec_level, table):
    n_pairs, dim = feats.size(0) // 2, feats.size(1)
    thx = table[dim][sec_level]
    
    left, right = feats[::2], feats[1::2]
    score = (F.cosine_similarity(left, right).acos() * (180 / torch.pi)).reshape(20, -1)
    true_pair, false_pair = score[::2].reshape(-1), score[1::2].reshape(-1)
    
    TA = (true_pair < thx).sum()
    TR = (false_pair > thx).sum()
    FA = (false_pair < thx).sum()
    FR = (true_pair > thx).sum()
    ACC = (TA + TR) / n_pairs
    
    TAR = TA / (n_pairs / 2)
    FAR = FA / (n_pairs / 2)
    
    return TAR, FAR, ACC

@torch.no_grad()
def upper_bound_estim_thx(feats, thx):
    n_pairs, dim = feats.size(0) // 2, feats.size(1)
    
    left, right = feats[::2], feats[1::2]
    score = (F.cosine_similarity(left, right).acos() * (180 / torch.pi)).reshape(20, -1)
    true_pair, false_pair = score[::2].reshape(-1), score[1::2].reshape(-1)
    
    TA = (true_pair < thx).sum()
    TR = (false_pair > thx).sum()
    FA = (false_pair < thx).sum()
    FR = (true_pair > thx).sum()
    ACC = (TA + TR) / n_pairs
    
    TAR = TA / (n_pairs / 2)
    FAR = FA / (n_pairs / 2)
    
    return TAR, FAR, ACC

from feat_tools import feat_ext


@torch.no_grad()
def end_to_end_benchmark(dataset, backbone, batch_size, device, table, target_far = None, suffix = None):
    # Step 1. Feature Extraction
    feats = feat_ext(dataset, backbone, batch_size, device)
    
    # Step 2. Benchmark
    thx, ACC, TAR, FAR = benchmark_plain(feats, target_far)
    
    
    print("Benchmark, Plain")
    print(f"TAR: {TAR.item() * 100:.2f}%")
    print(f"FAR: {FAR.item() * 100:.2f}%")
    print(f"ACC: {ACC.item() * 100:.2f}%")
    print(f"THX: {thx.item():.4f}")
    
    # Step 3. upper_bind_estim
    sec_levels = [128, 192, 256]
    for level in sec_levels:
        print(f"{level}-bit security level...")
        TAR, FAR, ACC = upper_bound_estim_tab(feats, level, table)
        print(f"TAR: {TAR.item() * 100:.2f}%")
        print(f"FAR: {FAR.item() * 100:.2f}%")
        print(f"ACC: {ACC.item() * 100:.2f}%")    

    plot_true_false_scores(feats, thx.item(), sec_levels, table, suffix)

import matplotlib.pyplot as plt

@torch.no_grad()
def plot_true_false_scores(feats, thx_orig, sec_levels, table, suffix = None):
    dim = feats.size(1)
    left, right = feats[::2], feats[1::2]
    score = (F.cosine_similarity(left, right).acos() * (180 / torch.pi)).reshape(20, -1).cpu()
    pos, neg = score[::2].reshape(-1), score[1::2].reshape(-1)
    
    
    
    plt.figure(figsize = (8, 4))
    plt.title(suffix, fontsize = 25, weight = "bold")
    plt.hist(pos.numpy(), bins = 40, alpha = .25, color = 'b', label = "Pos")
    plt.hist(neg.numpy(), bins = 40, alpha = .25, color = 'r', label = "Neg")
    plt.vlines(thx_orig, 0, 400, linewidth = 3, colors = "r", linestyles = 'solid')
    
    ctable = {128:"tab:blue", 192: "tab:orange", 256:"tab:green"}
    for level in sec_levels:
        theta = table[dim][level]
        plt.vlines(theta, 0, 400, linewidth = 3, colors = ctable[level], linestyles = 'dashed')
    plt.xlim(20, 100)
    plt.ylim(0, 350)
    plt.xticks(np.linspace(20, 100, 5), fontsize = 20)
    plt.yticks([0, 100, 200, 300, 400], fontsize = 20)
    plt.legend(fontsize = 20, loc = 'upper right')
    plt.show()