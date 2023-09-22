# 1. Load packages
import os
import torch
import pyramid
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models.resnet as res


from math import pi
from functools import partial
from yacs.config import CfgNode as CN
from global_model import global_module
from torch.hub import load_state_dict_from_url

# 2. Define the functions

# 3. Test the model.
