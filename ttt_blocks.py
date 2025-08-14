import torch
import torch.nn as nn
import torch.nn.functional as F

# Define MLP for TTT
class TTTMLP(torch.nn.Module):
    def __init__(self, dim):
        super(TTTMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )
        self.norm = torch.nn.LayerNorm(dim)
        
    def forward(self, x):
        dtype = x.dtype
        y = self.mlp(x.to(torch.float32))
        y = self.norm(y)
        y = y.to(dtype)
        return x + y
    
def ttt_prime(ttt, x, dim=1):
    x_rev = torch.flip(x, dims=[dim])
    ttt_rev = ttt(x_rev)
    return torch.flip(ttt_rev, dims=[dim])

# Define TTT self-supervised module
class ttt_self_supervised_module(nn.Module):
    def __init__(self, input_dim, proj_dim, ttt_module_cls):
        super().__init__()
        self.theta_K = nn.Linear(input_dim, proj_dim, bias=False)
        self.theta_V = nn.Linear(input_dim, proj_dim, bias=False)
        self.ttt_module = ttt_module_cls(proj_dim)
        self.proj_dim = proj_dim

    def forward(self, x):
        x_k = self.theta_K(x)
        x_v = self.theta_V(x)
        out = self.ttt_module(x_k)
        loss = F.mse_loss(out, x_v)
        return loss

class Gating(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = torch.nn.Parameter(0.1 * torch.ones(dim)) 
    
    def forward(self, x, z):
        return x + torch.tanh(self.alpha) * z

class TTTModule(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gating_a = Gating(dim)
        self.gating_b = Gating(dim)
        self.ttt_mlp = TTTMLP(dim)

    def forward(self, x, x_):
        z = self.gating_a(x_, self.ttt_mlp(x_))
        z_ = self.gating_b(z, ttt_prime(self.ttt_mlp, z))
        return x + z_


class TTT(torch.nn.Module):
    def __init__(self, num_layers, dim):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            TTTModule(dim) for _ in range(num_layers)
        ])
    
    def forward(self, index, x, x_):
        return self.layers[index](x, x_)

    def self_supervised_forward(self, xk, xv, use_ttt_prime=True):
        total_loss = 0
        # The loss is applied to each TTTMLP module within the TTT layers
        for layer in self.layers:
            ttt_mlp = layer.ttt_mlp
            if use_ttt_prime:
                # Time-reversal task
                xk_rev = torch.flip(xk, dims=[1])
                out = ttt_mlp(xk_rev)
                out = torch.flip(out, dims=[1])
                loss = F.mse_loss(out, xv)
            else:
                # Simple prediction task
                out = ttt_mlp(xk)
                loss = F.mse_loss(out, xv)
            total_loss += loss
        return total_loss / len(self.layers) if self.layers else 0.0

# you can use this like this
# my_ttt_module = TTT(len(transformer.layers), transformer.dim)
# out = transformer(x, ..., ttt_module = my_ttt_module)