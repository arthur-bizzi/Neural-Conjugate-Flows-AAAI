import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, sqrt
from .nets_and_flows import OrthogonalLayer


def twin_padding(x: torch.Tensor, pad_times: int = 1):
    return x.repeat(0,0,pad_times+1)

def twin_unpadding(x: torch.Tensor, pad_times: int = 1):
    times = pad_times+1
    x = torch.stack(x.chunk(times,dim=-1))
    return x.mean(dim=-1, keepdim=False)

def zero_padding(x: torch.Tensor, pad_times: int = 1):
    pad_size = x.shape[-1]*(pad_times+1)
    return F.pad(x, (0,pad_size))

def zero_unpadding(x: torch.Tensor, pad_times: int = 1):
    times = pad_times+1
    x = torch.stack(x.chunk(times,dim=-1))
    return x.sum(dim=-1, keepdim=False)

def zero_unpadding2(x: torch.Tensor, pad_times: int = 1):
    times = pad_times+1
    return x[...,:x.shape[-1]//times]

def identity(x: torch.Tensor, pad_times: int = 1):
    return x

class NeuralConjugate(nn.Module):
    # Constructs a conjugation from invertible networks.
    def __init__(self, 
                 layers, # List or tuple of layers
                 psi = None, # Conjugate flow
                 pad = 'no',
                 pad_times = 1
                 ):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        self.psi = psi
        self.pad_times = pad_times
        if pad == 'twin':
            self.pad = twin_padding
            self.unpad = twin_unpadding
        elif pad == 'zero':
            self.pad = zero_padding
            self.unpad = zero_unpadding
        elif pad == 'no':
            self.pad = identity
            self.unpad = identity
        
        self.epsilon = 1e-3

    def H(self, x: torch.Tensor):
        for k in self.layers:
            x = k(x)
        return x
    
    def H_inv(self, x: torch.Tensor):
        for k in reversed(self.layers):
            x = k.inverse(x)
        return x

    def forward(self,x, t):
        psi = self.psi
        pad_times = self.pad_times

        x = self.pad(x, pad_times)
        x = self.H(x)
        x = psi(x,t)
        x = self.H_inv(x)
        x = self.unpad(x, pad_times)
        
        return x
    
    def diff(self, x, t):
        epsilon = self.epsilon
        x_minus = self.forward(x,t-epsilon)
        x_plus = self.forward(x,t+epsilon)
        return (x_plus - x_minus)/(2*epsilon)
