import numpy as np
import torch
import torch.nn as nn

def weight_comp(y):
    cw = np.sum(y, axis = 0)
    cw = cw / y.shape[0]
    return cw

def integrated_loss_weight(wt):
    w = wt / np.amax(wt)
    return w

def loss_weight(yt, ysb):
    wt = weight_comp(yt)
    w_class = integrated_loss_weight(wt)
    w_sample_class = np.array(([0.] * 2 * ysb.shape[0]))
    w_sample_adv = np.array(([1.] * 2 * ysb.shape[0]))

    ysbi = ysb.argmax(1)
    w_sample_class[0:ysb.shape[0]] = w_class[ysbi]
    w_sample_adv[0:ysb.shape[0]] = w_class[ysbi]

    return w_sample_class, w_sample_adv

def MSE(pred, real):
    diffs = torch.add(real, -pred)
    n = torch.numel(diffs.data)
    mse = torch.sum(diffs.pow(2)) / n
    
    return mse

def SIMSE(pred, real):
    diffs = torch.add(real, - pred)
    n = torch.numel(diffs.data)
    simse = torch.sum(diffs).pow(2) / (n ** 2)
    
    return simse

def DiffLoss(x, recon_x):
    
    batch_size = x.size(0)
    x = x.view(batch_size, -1)
    recon_x = recon_x.view(batch_size, -1)
    
    x_l2_norm = torch.norm(x, p=2, dim=1, keepdim=True).detach()
    x_l2 = x.div(x_l2_norm.expand_as(x) + 1e-6)
    
    recon_x_l2_norm = torch.norm(recon_x, p=2, dim=1, keepdim=True).detach()
    recon_x_l2 = recon_x.div(recon_x_l2_norm.expand_as(recon_x) + 1e-6)
    
    diff_loss = torch.mean((x_l2.t().mm(recon_x_l2)).pow(2))
    
    return diff_loss
