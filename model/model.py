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
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
  
    model = ReSpaceNet(block, layers, **kwargs)
    if pretrained:

        state_dict = load_state_dict_from_url(res.model_urls[arch],
                                                  progress=progress)
        model.load_state_dict(state_dict)

    return model


class ReSpaceNet(res.ResNet):
    def __init__(self, block, layers,**kwargs):
        super().__init__(block,layers,**kwargs)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def patch(self):
        cw = self.conv1.weight

        self.conv1.stride = (1,1)
        self.conv1.weight = nn.Parameter(cw[:,[1],:,:])

        self.layer4 = None
        self.layer3[1].conv2.bias = nn.Parameter(torch.zeros(256))  
        self.layer3[1].bn2 = nn.Sequential() 

        self.fc = None

class Regressor(nn.Module):
    def __init__(self,levels):
        super().__init__()
        # self.fc1 = nn.Linear(4860, 512) 
        self.fc1 = nn.Linear(256*3 * levels, 512)  


        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(512, 128)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.regressor = nn.Linear(128, 2)
        nn.init.orthogonal_(self.regressor.weight)
        nn.init.zeros_(self.regressor.bias)

    def forward(self,x):
        out = self.fc1(x)
        out = F.gelu(out)  # relu 

        out = self.fc2(out)
        out = F.gelu(out)  # relu 

        out = self.regressor(out)
        return out


# 3. Test the model.
