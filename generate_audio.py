#!/usr/bin/env python3
"""
Audio generation script using K-diffusion and TTT-enhanced AudioX model.

Generates audio for 3 prompts and combines them into a 30-second clip.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

# Import stable audio tools
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.inference.utils import set_audio_channels
from data_loader import dataset

# Import TTT modules
from ttt_blocks import TTT

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def load_model_and_config(model_dir: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load AudioX model and configuration."""
    print(f"Loading model from {model_dir}")
    
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        model_config = json.load(f)
    
    # Create model
    model = create_model_from_config(model_config)
    
    # Load weights
    ckpt_path = None
    for name in ["model.safetensors", "model.ckpt"]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            ckpt_path = p
            break
    
    if ckpt_path is None:
        raise FileNotFoundError(f"No model weights found in {model_dir}")
    
    print(f"Loading weights from {ckpt_path}")
    state_dict = load_ckpt_state_dict(ckpt_path)
    model.load_state_dict(state_dict)
    
    return model, model_config


def create_ttt_module(model_config: Dict[str, Any], depth: int, dim: int, mini_batch_size: int = 1) -> TTT:
    """Create TTT module with proper dimensions matching saved weights."""
    # Based on the saved weights analysis:
    # W1 shape: [1, 384, 1536] means num_heads=1, head_dim=384, 4*head_dim=1536
    # So proj_dim should be num_heads * head_dim = 1 * 384 = 384
    proj_dim = 384  # Match the saved weights configuration
    num_heads = 24   # Match the saved weights configuration
    
    ttt_module = TTT(
        num_layers=depth,
        model_dim=dim,
        proj_dim=proj_dim,
        mini_batch_size=mini_batch_size,
        num_heads=num_heads
    )
    
    print(f"Created TTT with {depth} layers, dim={dim}, proj_dim={proj_dim}, heads={num_heads}, mini_batch_size={mini_batch_size}")
    return ttt_module


def load_ttt_weights(ttt_module: TTT, ttt_weights_path: str) -> TTT:
    """Load TTT weights from checkpoint."""
    print(f"Loading TTT weights from {ttt_weights_path}")
    
    if not os.path.exists(ttt_weights_path):
        print(f"Warning: TTT weights file {ttt_weights_path} not found. Using random initialization.")
        return ttt_module
    
    try:
        # Try loading just the TTT state dict first
        ttt_state = torch.load(ttt_weights_path, map_location='cpu')
        
        # If it's a checkpoint dict, extract the TTT state
        if isinstance(ttt_state, dict) and 'ttt_state_dict' in ttt_state:
            ttt_state = ttt_state['ttt_state_dict']
        elif isinstance(ttt_state, dict) and 'state_dict' in ttt_state:
            ttt_state = ttt_state['state_dict']
        
        ttt_module.load_state_dict(ttt_state, strict=False)
        print("Successfully loaded TTT weights")
        
    except Exception as e:
        print(f"Failed to load TTT weights: {e}")
        print("Using random TTT initialization")
    
    return ttt_module


def prepare_conditioning(prompts: List[str], model, device: torch.device) -> Dict[str, Any]:
    """Prepare conditioning tensors for the given text prompts."""
    print("Preparing conditioning...")
    
    conditioning = []
    for prompt in prompts:
        cond = {
            "text_prompt": prompt,
            "video_prompt": [torch.zeros(1, 50, 3, 224, 224, device=device)],
            "audio_prompt": torch.zeros(1, 2, 441000, device=device),
            "seconds_start": 0,
            "seconds_total": 10,  # 10 seconds per clip
        }
        conditioning.append(cond)
    
    return conditioning


def generate_all_audio_clips(
    model, 
    ttt_module: TTT,
    conditioning_list: List[Dict[str, Any]],
    duration_seconds: int = 10,
    steps: int = 100,
    cfg_scale: float = 7.0,
    seed: int = -1,
    device: torch.device = torch.device("cuda")
) -> List[torch.Tensor]:
    """Generate all audio clips simultaneously using TTT with mini-batch processing."""
    
    # Calculate sample size for the duration
    sample_rate = model.sample_rate
    sample_size = duration_seconds * sample_rate
    
    print(f"Generating 3x{duration_seconds}s audio simultaneously at {sample_rate}Hz (sample_size={sample_size})")
    print("TTT will adapt during inference using the mini-batch of 3 prompts")
    
    # Use batch_size=3 to generate all clips simultaneously for TTT
    # The conditioning_list should have all 3 conditioning dicts
    audio = generate_diffusion_cond(
        model=model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning_list,  # Pass all 3 conditioning dicts
        batch_size=3,  # Process all 3 clips together
        sample_size=sample_size,
        seed=seed,
        device=str(device),
        ttt_module=ttt_module,  # TTT will adapt using the 3-sample mini-batch
        sampler_type="dpmpp-3m-sde",  # Use the 3M SDE sampler as in training
        sigma_min=0.3,
        sigma_max=500,
        rho=1.0,
    )
    
    # Split the batched output into individual clips
    audio_clips = []
    for i in range(3):
        clip = audio[i:i+1]  # Extract individual clip maintaining batch dimension
        audio_clips.append(clip)
    
    return audio_clips


def combine_audio_clips(audio_clips: List[torch.Tensor]) -> torch.Tensor:
    """Combine multiple audio clips into a single tensor."""
    print(f"Combining {len(audio_clips)} audio clips...")
    
    # Concatenate along the time dimension
    combined = torch.cat(audio_clips, dim=-1)
    
    print(f"Combined audio shape: {combined.shape}")
    return combined


def save_audio(audio: torch.Tensor, output_path: str, sample_rate: int):
    """Save audio tensor to file."""
    print(f"Saving audio to {output_path}")
    
    # Ensure audio is on CPU and has correct shape [channels, time]
    audio = audio.squeeze(0).cpu()  # Remove batch dimension
    
    # Clamp audio to valid range
    audio = torch.clamp(audio, -1.0, 1.0)
    
    torchaudio.save(output_path, audio, sample_rate)
    print(f"Saved audio with shape {audio.shape} at {sample_rate}Hz")


def main():
    parser = argparse.ArgumentParser(description="Generate audio using TTT-enhanced AudioX")
    parser.add_argument("--model_dir", type=str, default="model", 
                        help="Directory containing AudioX model and config")
    parser.add_argument("--ttt_weights", type=str, default="ttt_finetune/ttt_module.pt",
                        help="Path to TTT weights file")
    parser.add_argument("--output", type=str, default="generated_ttt_audio.wav",
                        help="Output audio file")
    parser.add_argument("--steps", type=int, default=250,
                        help="Number of diffusion steps")
    parser.add_argument("--cfg_scale", type=float, default=7.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (-1 for random)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--idx", type=int, default=0, 
                        help="Fallback if there is no prompt, <idx>'th data of the test set")
    parser.add_argument("--prompts", type=str, nargs=3, default=None,
                        help="Custom prompts to use instead of dataset prompts")

    args = parser.parse_args()
    idx = args.idx
    
    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and config
    model, model_config = load_model_and_config(args.model_dir)
    model = model.to(device)
    model.eval()
    
    # Get model dimensions for TTT
    dit = model.model  # DiTWrapper
    diffusion_transformer = dit.model
    ct = diffusion_transformer.transformer  # ContinuousTransformer
    dim = ct.dim
    depth = ct.depth
    
    # Create and load TTT module with mini_batch_size=3 for proper TTT inference
    ttt_module = create_ttt_module(model_config, depth, dim, mini_batch_size=3)  # Use mini_batch_size=3 for TTT
    ttt_module = load_ttt_weights(ttt_module, args.ttt_weights)
    ttt_module = ttt_module.to(device)
    ttt_module.eval()
    
    # Define prompts
    if args.prompts is None:
        prompts = [
            dataset['test'][idx]['text1'],
            dataset['test'][idx]['text2'],
            dataset['test'][idx]['text3']
        ]
    else:
        prompts = args.prompts

    # prompts = ['owl is hooting at windy night', 'cat starts to meow and owl keeps hooting in the in the distance', 'the cat continues meowing while being chased by the hooting owl in the bushes']
    
    print(f"Generating audio for prompts: {prompts}")
    
    # Prepare conditioning for all prompts
    conditioning_list = []
    for prompt in prompts:
        cond = {
            "text_prompt": prompt,
            "video_prompt": [torch.zeros(1, 50, 3, 224, 224, device=device)],
            "audio_prompt": torch.zeros(1, 2, 441000, device=device),
            "seconds_start": 0,
            "seconds_total": 10,
        }
        conditioning_list.append(cond)
    
    # Generate all audio clips simultaneously using TTT
    print(f"\n=== Generating all clips simultaneously with TTT ===")
    audio_clips = generate_all_audio_clips(
        model=model,
        ttt_module=ttt_module,
        conditioning_list=conditioning_list,
        duration_seconds=10,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed if args.seed != -1 else None,
        device=device
    )
    
    # Save individual clips
    for i, (audio_clip, prompt) in enumerate(zip(audio_clips, prompts)):
        individual_path = f"clip_{i+1}_{prompt.replace(' ', '_').replace(',', '').lower()}.wav"
        save_audio(audio_clip, individual_path, model.sample_rate)
    
    # Combine clips into 30-second audio
    print(f"\n=== Combining clips into 30-second audio ===")
    combined_audio = combine_audio_clips(audio_clips)
    
    # Save combined audio
    save_audio(combined_audio, args.output, model.sample_rate)
    
    print(f"\n✅ Generation complete!")
    print(f"Combined 30-second audio saved as: {args.output}")


if __name__ == "__main__":
    main()
