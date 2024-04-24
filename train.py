# 01. import packages
import os
import sys
import copy
import random
import XrayData19 as XrayData
from pyramid import pyramid, stack, pyramid_transform

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model.model_19 as m
import math
from math import pi
from time import time
from random import randint
from thop import profile


np.set_printoptions(suppress=True, precision=10, threshold=2000, linewidth=150) 
IMG_SIZE_ORIGINAL = {'width': 1935, 'height': 2400}
IMG_SIZE_ROUNDED_TO_64 = {'width': 1920, 'height': 2432}
IMG_TRANSFORM_PADDING = {'width': IMG_SIZE_ROUNDED_TO_64['width'] - IMG_SIZE_ORIGINAL['width'],
                        'height': IMG_SIZE_ROUNDED_TO_64['height']- IMG_SIZE_ORIGINAL['height']}


# 02.  define functions
def Calculate_Flops_Params(size, poes, model):
    x = size
    rate = 1 
    model.cuda()
    macs, params = profile(model, inputs=(x, poes))
    start = time()
    y = model(x, poes, train=False)
    end = time()
    Params = params / 1e6 # (M)
    FLOPs = 2 * macs / 1e9 / rate # (G)
    print('FLOPs:', FLOPs, ', Params:', Params, 'Running time: %s Seconds' % (end - start))

def rescale_point_to_original_size(point):
    middle = np.array([IMG_SIZE_ROUNDED_TO_64['width'], IMG_SIZE_ROUNDED_TO_64['height']]) / 2
    return ((point*IMG_SIZE_ROUNDED_TO_64['width'])/2) + middle

# 03. edit main function

















