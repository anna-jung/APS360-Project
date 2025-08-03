import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import read_video, merge_video_audio
from stable_audio_tools.data.utils import load_and_process_audio

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
        return x + self.norm(self.mlp(x))
    
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
        self.gating_a = Gating(dim)
        self.gating_b = Gating(dim)
        self.ttt_mlp = TTTMLP(dim)

    def forward(self, x, x_):
        z = self.gating_a(x_, self.ttt_mlp(x_))
        z_ = self.gating_b(z, ttt_prime(self.ttt_mlp, z))
        return x + z_
    