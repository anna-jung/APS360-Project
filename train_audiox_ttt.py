import os
import math
import argparse
from typing import List, Dict, Any, Tuple
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

from accelerate import Accelerator
from accelerate.utils import set_seed

from data_loader import TTT_RecaptionedDataset, SAMPLE_RATE as DS_SR
from datasets import load_dataset

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.utils import set_audio_channels

from ttt_blocks import TTT

# Global training state
train_hist = {"step": [], "loss": []}
kdenoiser = None

# Training config
p_mean = 2.5
p_std = 1.2
sigma_min = 0.3
sigma_max = 500.0
cfg_dropout_prob = 0.1


def collate_batch_ttt(batch):
    """
    Collate function for TTT training.
    
    Input: List of (audios_list, metas_list) where each has 3 audio/meta items
    Output: (batch_audios_list, batch_metas_list) for TTT mini-batch training
    """
    batch_audios = []
    batch_metas = []
    
    for audios_list, metas_list in batch:
        batch_audios.extend(audios_list)  # List of 3 audio tensors
        for label in metas_list:
            meta = {
                "text_prompt": label,
                # default non-None prompts like run.py expects
                # video_prompt should be a list with a batched tensor
                "video_prompt": [torch.zeros(1, 50, 3, 224, 224)],
                # audio_prompt is a batched tensor [B, C, T]
                "audio_prompt": torch.zeros(1, 2, 441000),
                # include timing metadata
                "seconds_start": 0,
                "seconds_total": 10,
            }
            batch_metas.append(meta)

    return torch.stack(batch_audios), batch_metas


def load_local_audiox(model_dir: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model and config from local directory."""
    import json
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r") as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    # Try safetensors then ckpt
    ckpt_path = None
    for name in ["model.safetensors", "model.ckpt"]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No model weights found in {model_dir}")

    sd = load_ckpt_state_dict(ckpt_path)
    model.load_state_dict(sd)
    return model, model_config


def main():
    """Main training script for TTT-enabled diffusion model using Accelerate."""
    global train_hist, kdenoiser
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model", help="Local AudioX model dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default="./ttt_checkpoints", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to load checkpoint from")
    parser.add_argument("--mini_batch_size", type=int, default=3, help="Mini-batch size for TTT")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for TTT training")
    parser.add_argument("--scheduler", type=str, default="constant", choices=["constant", "cosine"])
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set up logging only on main process
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Training TTT with Accelerate")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"Output directory: {args.output_dir}")
        
        # Initialize CSV logging
        csv_path = os.path.join(args.output_dir, "training_log.csv")
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['step', 'loss', 'grad_norm', 'learning_rate'])
        csv_file.flush()
    else:
        csv_file = None
        csv_writer = None

    # Load dataset
    from data_loader import dataset as hf_dataset
    for split in list(hf_dataset.keys()):
        try:
            if 'audio' in hf_dataset[split].features:
                hf_dataset[split] = hf_dataset[split].remove_columns(['audio'])
        except Exception:
            pass

    # Load local AudioX
    diffusion_model, model_cfg = load_local_audiox(args.model_dir)
    sample_rate = model_cfg['sample_rate']

    # Build dataset/loader
    train_ds = TTT_RecaptionedDataset(hf_dataset["train"], sample_rate=sample_rate)
    dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch_ttt,
        pin_memory=True,
    )

    # Introspect transformer to construct TTT
    dit = diffusion_model.model  # DiTWrapper
    diffusion_transformer = dit.model
    ct = diffusion_transformer.transformer  # ContinuousTransformer
    dim = ct.dim
    depth = ct.depth

    # Define projection dimension
    proj_dim = dim // 4
    
    # Create TTT module
    ttt_module = TTT(
        num_layers=depth,
        model_dim=dim,
        proj_dim=proj_dim,
        mini_batch_size=args.mini_batch_size,
        num_heads=model_cfg["model"]["diffusion"]["config"]["num_heads"]
    )

    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint, map_location="cpu")
        if 'ttt_state_dict' in checkpoint:
            # Loading from full checkpoint
            state_dict = checkpoint['ttt_state_dict']
        else:
            # Loading just TTT weights
            state_dict = checkpoint
        
        ttt_module.load_state_dict(state_dict)
        print(f"Loaded TTT checkpoint from {args.load_checkpoint}")

    if accelerator.is_main_process:
        print(f"Created TTT with {depth} layers, dim={dim}, proj_dim={proj_dim}")
        total_params = sum(p.numel() for p in ttt_module.parameters())
        trainable_params = sum(p.numel() for p in ttt_module.parameters() if p.requires_grad)
        print(f"TTT total parameters: {total_params}")
        print(f"TTT trainable parameters: {trainable_params}")

    # Freeze diffusion model, train TTT
    diffusion_model.requires_grad_(False)
    diffusion_model.eval()
    ttt_module.requires_grad_(True)
    ttt_module.train()

    # Create optimizer and scheduler inline
    optimizer = torch.optim.AdamW(
        ttt_module.parameters(), 
        lr=args.learning_rate, 
        betas=(0.9, 0.999), 
        weight_decay=1e-4
    )
    
    match args.scheduler:
        case "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.max_steps)
        case "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-6)

    # Prepare everything with accelerator
    diffusion_model, ttt_module, optimizer, dl, scheduler = accelerator.prepare(
        diffusion_model, ttt_module, optimizer, dl, scheduler
    )
    
    # Optional: faster matmul on Tensor Cores
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass

    # Setup K-diffusion wrapper once
    from k_diffusion.external import VDenoiser
    kdenoiser = VDenoiser(diffusion_model.model)

    # Training loop
    global_step = 0
    
    if accelerator.is_main_process:
        print(f"Starting training for {args.max_steps} steps...")
        
    while global_step < args.max_steps:
        for batch in dl:
            with accelerator.accumulate(diffusion_model):
                # Training step inline
                reals, meta = batch
                
                # Prepare conditioning
                cond = diffusion_model.conditioner(meta, accelerator.device)
                
                # Optional pretransform encode
                diffusion_input = reals
                if getattr(diffusion_model, "pretransform", None) is not None:
                    pt = diffusion_model.pretransform.to(accelerator.device)
                    pt.eval()
                    with torch.no_grad():
                        diffusion_input = set_audio_channels(diffusion_input, pt.io_channels)
                        diffusion_input = pt.encode(diffusion_input)

                # K-diffusion (EDM) objective
                b = diffusion_input.shape[0]
                dtype = diffusion_input.dtype
                
                # Sample log-normal sigmas
                r = torch.randn(b, device=accelerator.device, dtype=dtype) * p_std + p_mean
                sigma = r.exp().clamp(min=sigma_min, max=sigma_max)
                noise = torch.randn_like(diffusion_input)
                
                # Build conditioning inputs
                cond_inputs = diffusion_model.get_conditioning_inputs(cond)
                
                # Diffusion loss from k-diffusion
                loss = kdenoiser.loss(
                    diffusion_input, noise, sigma,
                    cfg_dropout_prob=cfg_dropout_prob,
                    ttt_module=ttt_module,
                    **cond_inputs,
                ).mean()
                
                # Backward pass
                accelerator.backward(loss)
                
                # Calculate gradient norm inline
                total_norm = 0.0
                param_count = 0
                for name, param in ttt_module.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                if param_count > 0 and total_norm > 0:
                    grad_norm = total_norm ** 0.5
                else:
                    grad_norm = 0.0
                    print('Warning: No gradients found')

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update training history
                train_hist["step"].append(global_step)
                train_hist["loss"].append(loss.item())
                
                # Logging
                if accelerator.is_main_process and (global_step % args.log_every == 0 or global_step == 0):
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step}/{args.max_steps} | Loss: {loss.item():.6f} | Grad Norm: {grad_norm:.6f} | LR: {current_lr:.2e}")
                    
                    # Log to CSV
                    if csv_writer is not None:
                        csv_writer.writerow([global_step, loss.item(), grad_norm, current_lr])
                        csv_file.flush()
                
                # Save checkpoint inline
                if accelerator.is_main_process and global_step > 0 and global_step % args.save_every == 0:
                    try:
                        model_path = os.path.join(args.output_dir, f"step_{global_step}.pt")
                        torch.save({
                            'step': global_step,
                            'diffusion_state_dict': accelerator.unwrap_model(diffusion_model).state_dict(),
                            'ttt_state_dict': accelerator.unwrap_model(ttt_module).state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'train_hist': train_hist,
                        }, model_path)
                        
                        ttt_path = os.path.join(args.output_dir, f"ttt_module_step_{global_step}.pt")
                        torch.save(accelerator.unwrap_model(ttt_module).state_dict(), ttt_path)
                        
                        print(f"Saved step {global_step} checkpoint: {model_path}")
                        print(f"Saved step {global_step} TTT weights: {ttt_path}")
                        
                    except Exception as e:
                        print(f"Failed to save step {global_step} checkpoint: {e}")
                
                global_step += 1
                if global_step >= args.max_steps:
                    break
    
    # Final save inline
    if accelerator.is_main_process:
        print("Training completed!")
        
        # Close CSV file
        if csv_file is not None:
            csv_file.close()
            print(f"Saved training log to {csv_path}")
        
        try:
            ttt_path = os.path.join(args.output_dir, "ttt_module.pt")
            torch.save(accelerator.unwrap_model(ttt_module).state_dict(), ttt_path)
            print(f"Saved final TTT weights to {ttt_path}")
            
            # Save training curves
            if len(train_hist["step"]) > 0:
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(train_hist["step"], train_hist["loss"], label="loss")
                    ax.set_xlabel("step")
                    ax.set_ylabel("loss")
                    ax.legend()
                    fig.tight_layout()
                    out_path = os.path.join(args.output_dir, "ttt_training_curves.png")
                    fig.savefig(out_path)
                    print(f"Saved training curves to {out_path}")
                except Exception as e:
                    print(f"Plotting failed: {e}")
                    
        except Exception as e:
            print(f"Saving final weights failed: {e}")

    # Clean up
    accelerator.end_training()


if __name__ == "__main__":
    main()
