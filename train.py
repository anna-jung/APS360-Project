import torch
import torch.nn.functional as F
import torchaudio

from ttt_blocks import TTTMLP, ttt_self_supervised_module
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import load_and_process_audio


def run_ttt_inner_loop(x_latent_btd, num_iterations, ss=ttt_self_supervised_module, lr=0.01, use_ttt_prime=False):

    # Freeze the parameters of the outer loop
    for p in ss.theta_K.parameters(): p.requires_grad_(False)
    for p in ss.theta_V.parameters(): p.requires_grad_(False)

    inner_params = list(ttt_self_supervised_module.ttt_module.parameters())
    optimizer = torch.optim.Adam(inner_params, lr=lr)
    
    ss.train()

    for i in range(num_iterations):
        optimizer.zero_grad()
        if use_ttt_prime:
            xk = ss.theta_K(x_latent_btd)
            xv = ss.theta_V(x_latent_btd)
            xk_rev = torch.flip(xk, dims=[1])
            out = ss.ttt_module(xk_rev)
            out = torch.flip(out, dims=[1])
            loss = F.mse_loss(out, xv)
        else:
            loss = ss(x_latent_btd)

        loss.backward()
        optimizer.step()
    return ss.ttt_module