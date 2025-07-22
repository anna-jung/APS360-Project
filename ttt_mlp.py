import torch
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