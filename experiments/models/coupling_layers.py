import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, sqrt
from .nets_and_flows import OrthogonalLayer

def no_shuffle(x,y):
    return x, y

def shuffle(x,y):
    return y, x

def norm(x):
    return torch.linalg.norm(x,dim=-1,keepdim=True)

    
class AdditiveCouplingLayer(nn.Module):
    def __init__(self, model, * , initial_step_size = None, orientation = 'normal'):
        super().__init__()

        self.sub_model = model
        if initial_step_size is None:
            self.register_buffer('initial_step_size', 0.1*torch.ones(1))
        elif torch.is_tensor(initial_step_size) or isinstance(initial_step_size, float):
            self.initial_step_size = nn.Parameter(torch.ones(1)*initial_step_size)

        if orientation == 'normal':
            self.shuffle = no_shuffle
        elif orientation == 'skew':
            self.shuffle = shuffle

    def forward(self, x):
        x_1, x_2 = x.chunk(2,dim=-1)
        x_1, x_2 = self.shuffle(x_1,x_2)
        
        delta = self.sub_model(x_2)*self.initial_step_size

        x_1 = x_1 + delta
        x_1, x_2 = self.shuffle(x_1,x_2)
        return torch.cat((x_1,x_2),dim=-1)
    
    def inverse(self, x):
        x_1, x_2 = x.chunk(2,dim=-1)
        x_1, x_2 = self.shuffle(x_1,x_2)
        
        delta = self.sub_model(x_2)*self.initial_step_size
        x_1 = x_1 - delta
        
        x_1, x_2 = self.shuffle(x_1,x_2)
        return torch.cat((x_1,x_2),dim=-1)
    

class DoubleAdditiveCouplingLayer(nn.Module):
    def __init__(self, models, * , orientation = 'normal', gamma=1.0):
        super().__init__()

        self.models = models

        if orientation == 'normal':
            self.shuffle = no_shuffle
        elif orientation == 'skew':
            self.shuffle = shuffle
        
        self.gamma = gamma

    def forward(self, x):
        x_1, x_2 = x.chunk(2,dim=-1)
        
        x_1 = x_1 + self.gamma * self.models[0](x_2)
        x_2 = x_2 + self.gamma * self.models[1](x_1)
        
        x_1, x_2 = self.shuffle(x_1,x_2)
        
        return torch.cat((x_1,x_2),dim=-1)
    
    def inverse(self, x):
        x_1, x_2 = x.chunk(2,dim=-1)
        x_1, x_2 = self.shuffle(x_1,x_2)
        
        x_2 = x_2 - self.gamma * self.models[1](x_1)
        x_1 = x_1 - self.gamma * self.models[0](x_2)
        
        return torch.cat((x_1,x_2),dim=-1)
    

class AdditiveOrthogonalCouplingLayer(nn.Module):
    def __init__(self, model, orthogonal_layer):
        super().__init__()

        self.sub_model = model
        self.orthogonal_map = orthogonal_layer

    def forward(self, x):
        x = self.orthogonal_map(x)
        
        x_1, x_2 = x.chunk(2,dim=-1)
        
        delta = self.sub_model(x_2)
        x_1 = x_1 + delta
        
        x = self.orthogonal_map.inverse(torch.cat((x_1,x_2),dim=-1))
        return x
    
    def inverse(self, x):
        x = self.orthogonal_map(x)
        
        x_1, x_2 = x.chunk(2,dim=-1)
                
        delta = self.sub_model(x_2)
        x_1 = x_1 - delta
        
        x = self.orthogonal_map.inverse(torch.cat((x_1,x_2),dim=-1))
        return x
    
    
class AffineCouplingLayer(nn.Module):
    def __init__(self, model_add, model_mul, * , initial_step_size = 0.1, orientation = 'normal'):
        super().__init__()

        self.sub_model_add = model_add
        self.sub_model_mul = model_mul

        if orientation == 'normal':
            self.shuffle = no_shuffle
        elif orientation == 'skew':
            self.shuffle = shuffle

    def forward(self, x):
        x_1, x_2 = x.chunk(2,dim=-1)
        x_1, x_2 = self.shuffle(x_1,x_2)
        
        delta = self.sub_model_add(x_2)
        mul = torch.exp(self.sub_model_mul(x_2)/10)

        x_1 = x_1*mul + delta
        x_1, x_2 = self.shuffle(x_1,x_2)
        return torch.cat((x_1,x_2),dim=-1)
    
    def inverse(self, x):
        x_1, x_2 = x.chunk(2,dim=-1)
        x_1, x_2 = self.shuffle(x_1,x_2)
        
        delta = self.sub_model_add(x_2)
        mul = torch.exp(-self.sub_model_mul(x_2)/10)
        x_1 = (x_1 - delta)*mul
        
        x_1, x_2 = self.shuffle(x_1,x_2)
        return torch.cat((x_1,x_2),dim=-1)

#----------------------------------------------    
