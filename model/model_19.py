# 1. Load packages
import os
import torch
import model.pyramid
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models.resnet as res


from math import pi
from functools import partial
from yacs.config import CfgNode as CN
from model.global_model import global_module
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


class Cyclic_Coordinate_Guided(nn.Module):
    def __init__(self,levels):

        super().__init__()
        self.levels = levels
        blocks = [2,2,2,2]
        names = 'resnet18'
      
        self.resnet = nn.Sequential(*[
            _resnet(names, res.BasicBlock,
                              blocks,
                              True, True)
            for i in range(19) # or 37
        ])
        self.dim_trans = nn.Linear(16384, 768)

        for i in range(19): # or 37
            self.resnet[i].eval()
            self.resnet[i].patch()

        self.regression_head = Regressor(levels)
        self.global_block = global_module()
    

    def forward(self, x, pos, train=False, ep=40, j=0, i=0):

        bs = pos.shape[0]
        multi = pos.shape[1]
        device = x[0].device
        H = x[0].shape[2]
        W = x[0].shape[3]
        pos_fix = pos.clone() 
        pos_fix[:, :, 1] = pos[:, :, 1] * (W / H)
        s = 64

        results = torch.rand((bs,multi,2), device=device)

        tmpout = torch.ones((bs, multi, 4608), device=device)

        for lk in range(multi):
            ls = 1
            theta = (torch.rand((bs, ls, 1, 1), device=device) * 2 - 1) * pi / 12
            scale = torch.exp((torch.rand((bs, ls, 1, 1), device=device) * 2 - 1)*0.05)
          
            if not train:
                theta = theta*0
                scale = torch.ones((bs, ls, 1, 1), device=device)
            rsin = theta.sin()  
            rcos = theta.cos()

            R = torch.cat((rcos, -rsin, rsin, rcos), 3).view(bs, ls, 2, 2)  

            T = torch.cat((R*scale, pos_fix[:,lk,:].unsqueeze(1).unsqueeze(3)), 3)
        
            stacked = pyramid.stack(x,s,T,augment=train) 

            self.stack_vis = stacked.detach()

            N = stacked.shape[0]
        
            multi_model = False
          
            if multi_model:
                tmp = torch.ones((bs, pos.shape[1], 2), device=device)  
                for ib in range(multi):
                    batched = stacked.view(N*self.levels,multi,1,s,s)   
                    out = self.resnet(batched[:,ib,...])
                    out, self.heat_vis = mass2d(out)  
                    out = torch.flatten(out,1) 
                    out = out.view(N, 1, -1)   
                    tmp[:,ib,:] = self.regression_head(out[:,0,:])   
            else:
                batched = stacked.view(N*ls*self.levels,1,s,s)  
                out = self.resnet[lk](batched)

                out = out.permute(0,2,3,1) 
                out = self.global_block(out)
                out = out.permute(0,3,1,2)

                out = torch.flatten(out,1)
                out = self.dim_trans(out)
                
                out = out.view(N, ls, -1)  

                tmpout[:,lk,:] = out.squeeze(1)

        tmpout = tmpout.view(bs, multi, 4608) 

        tmp = self.regression_head(tmpout)

        out = tmp.view(N,multi,1,2)

        out = torch.matmul(out,R.transpose(2,3)/scale)

        return  pos + out.view(N,multi,2) 


def load_model(levels,name,load=False):
    model = Cyclic_Coordinate_Guided(levels)

    device = 'cuda'

    if load:
        if device=='cuda':
            model.load_state_dict(torch.load(f"Models/{name}.pt"))
        elif device=='cpu':
            model.load_state_dict(torch.load(f"Models/{name}.pt", map_location=torch.device('cpu')))

    model.to(device)

    return model


def save_model(model, name):
    if not os.path.exists("Models"):
        os.mkdir("Models")
    # torch.save(model.state_dict(), f"Models/{name}.pt")
    torch.save(model.state_dict(), f"{name}.pt")

# 3. Test the model.
