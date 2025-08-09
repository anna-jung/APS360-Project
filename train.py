import torch
import torch.nn.functional as F
import torchaudio

from ttt_blocks import TTTMLP, ttt_self_supervised_module
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import load_and_process_audio


def run_ttt_inner_loop(ttt_model, x_latent_btd, num_iterations, lr=0.01):

    # Freeze the parameters of the outer loop
    for p in ttt_model.theta_K.parameters(): p.requires_grad_(False)
    for p in ttt_model.theta_V.parameters(): p.requires_grad_(False)

    inner_params = list(ttt_model.ttt_module.parameters())
    optimizer = torch.optim.Adam(inner_params, lr=lr)
    
    ttt_model
    for i in range(num_iterations):
        optimizer.zero_grad()
        xk = ttt_model.theta_K(x_latent_btd)
        xv = ttt_model.theta_V(x_latent_btd)
        xk_rev = torch.flip(xk, dims=[1])
        out = ttt_model.ttt_module(xk_rev)
        out = torch.flip(out, dims=[1])
        loss = F.mse_loss(out, xv)
        loss.backward()
        optimizer.step()
    return ttt_model.ttt_module