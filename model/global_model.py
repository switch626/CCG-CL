# optimized global model

import torch
import torch.nn as nn
import torch.nn.functional as F

class global_model(nn.Module):
    def __init__(self, input_channels=256, output_channels=512):
        super(global_model, self).__init__()

        self.expand = nn.Linear(input_channels, output_channels)
      
        self.norm = nn.LayerNorm(output_channels // 2)

        self.projection = nn.Linear(output_channels, input_channels)
    
    def forward(self, x):

        batch_size, height, width, channels = x.shape
        x = x.view(batch_size, height * width, channels)
        
        x = self.expand(x)
        
        u, v = torch.split(x, x.shape[-1] // 2, dim=-1)
        
        v = self.norm(v)
        gated_u = u * torch.sigmoid(v)
        
        combined_features = torch.cat([gated_u, u], dim=-1)
        output = self.projection(combined_features)
        
        output = output.view(batch_size, height, width, channels)
        
        return output

  
