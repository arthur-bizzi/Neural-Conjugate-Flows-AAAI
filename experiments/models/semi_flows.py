import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiFlow(nn.Module):
    def __init__(self, model, time_phi = nn.Tanh(), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.time_phi = time_phi

        self.epsilon = 1e-3
    
    def forward(self, x, t):
        x = x.unsqueeze(0)
        x_width = x.shape[-1]
        xt = F.pad(x, (0,1))
        xt = xt + F.pad(t.view(-1,1,1), (x_width,0))
        phi = self.time_phi(t).view(-1,1,1)
        return x + phi*self.model(xt)
    
    def diff(self, x, t):
        epsilon = self.epsilon
        x_minus = self(x,t-epsilon)
        x_plus = self(x,t+epsilon)

        return (x_plus - x_minus)/(2*epsilon)

# class InvertiblePseudoFlow
