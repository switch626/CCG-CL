import math
import torch
import torch.nn.functional as F
import numpy as np

def pyramid(I, levels):

    pym = []
    for i in range(levels-1):
        pym.append(I)
        I=gaussian_reduce(I,sigma=2.0)

    pym.append(I)

    return pym


def pyramid_transform(T,H,W, size, level):
    i = level
    scale = torch.tensor([
        size / W, size / W, 1, size / H, size / H, 1], dtype=torch.float32,
        device=T.device).reshape(2, 3)*torch.tensor([2 ** i, 2 ** i, 1], device=T.device)
    T = T*scale
    M = T[:,:2]
    Minv = M.inverse()
    t = T[:,[2]]
    T = torch.cat(
        (Minv,torch.mm(-Minv,t)),1
    )
    return T

def stack(pym, size, T, augment=False):
    N = pym[0].shape[0]
    C = T.shape[1]
    H = pym[0].shape[2]
    W = pym[0].shape[3]

    stacked = torch.zeros(N,C,len(pym),size,size,device=pym[0].device)

    sample = torch.arange(-(size-1)/size,1,2/size,device=pym[0].device)
    gy,gx = torch.meshgrid(sample,sample)
    grid = torch.stack((gx,gy,torch.ones((size,size),device=pym[0].device)),2).expand(N,C,size,size,3)

    scale = torch.tensor([size / W, size / W, 1, size / H, size / H, 1], dtype=torch.float32,
                         device=pym[0].device).reshape(1, 1, 2, 3)
    for i in range(len(pym)):
        Tl = T * torch.tensor([2 ** i, 2 ** i, 1], device=pym[0].device) * scale  # 2 1 2 3

        g = torch.matmul(grid.view(N,C,size*size,3), Tl.transpose(2,3))\
            .view(N,size*C,size,2)  # 2 64 64 2  (N, H, W, 2)
        stacked[:, :, i, :, :] = F.grid_sample(pym[i],g,align_corners=False).view(N,C,size,size)

    if augment:
        stacked+=torch.randn(stacked.shape,device=pym[0].device)*0.01
    return stacked


def box_reduce(I):
    return (I[:,:, ::2, ::2] + I[:,:, 1::2, ::2] + I[:,:, ::2, 1::2] + I[:,:, 1::2, 1::2]) * 0.25

def gaussian_reduce(input, sigma=2.0/6.0, truncate=3.0):
    reduction = 0.5
    sd = math.sqrt(2 * math.log(1 / 0.5)) / (math.pi * reduction)
    size = input.shape
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd

    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    wH = torch.tensor(weights).reshape(1,1,1,2*lw+1).to(input.device)
    wV = torch.tensor(weights).reshape(1,1,2*lw+1,1).to(input.device)

    with torch.set_grad_enabled(False):
        out = F.conv2d(input,wH,padding=(0,lw))
        out = F.conv2d(out,wV,padding=(lw,0),stride=(2,2))
        return out
