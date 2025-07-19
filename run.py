# I learned something really cool

# noise = torch.randn(1, 2, 483328)  # batch size of 1, two speakers (left and right), 483328 audio frames (12s)
# latents = model.pretransform.encode(audio) # this will encode the audio into 1 x 64 x 236 latent vector
# model.pretransform.decode(latents) # this will decode the latents

# the autoencoder is already trained for you !!!!! so you can just use it without doing any work

video_path = None
text_prompt = "You can hear the train is coming into the platform and the bell started to ring"
audio_path = None




import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import read_video, merge_video_audio
from stable_audio_tools.data.utils import load_and_process_audio
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Download model
model, model_config = get_pretrained_model("HKUSTAudio/AudioX")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
target_fps = model_config["video_fps"]
seconds_start = 0
seconds_total = 10

model = model.to(device)

video_tensor = read_video(video_path, seek_time=0, duration=seconds_total, target_fps=target_fps)
audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)

conditioning = [{
    "video_prompt": [video_tensor.unsqueeze(0)],        
    "text_prompt": text_prompt,
    "audio_prompt": audio_tensor.unsqueeze(0),
    "seconds_start": seconds_start,
    "seconds_total": seconds_total
}]
    
# Generate stereo audio
output = generate_diffusion_cond(
    model,
    steps=250,
    cfg_scale=7,
    conditioning=conditioning,
    sample_size=sample_size,
    sigma_min=0.3,
    sigma_max=500,
    sampler_type="dpmpp-3m-sde",
    device=device
)

# Rearrange audio batch to a single sequence
output = rearrange(output, "b d n -> d (b n)")

# Peak normalize, clip, convert to int16, and save to file
output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
torchaudio.save("output.wav", output, sample_rate)

if video_path is not None and os.path.exists(video_path):
    merge_video_audio(video_path, "output.wav", "output.mp4", 0, seconds_total)
