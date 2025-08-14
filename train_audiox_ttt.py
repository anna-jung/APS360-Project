import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

import pytorch_lightning as pl

from data_loader import TTT_RecaptionedDataset, SAMPLE_RATE as DS_SR
from datasets import load_dataset

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.utils import set_audio_channels

from ttt_blocks import TTT


def _pad_trunc(audio: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or truncate last dimension to target_len."""
    n = audio.shape[-1]
    if n == target_len:
        return audio
    if n < target_len:
        pad = target_len - n
        return F.pad(audio, (0, pad))
    return audio[..., :target_len]


class TTTRecapTrain(Dataset):
    """Flattened wrapper over TTT_RecaptionedDataset: returns individual 10s segments.

    Output:
      - audio: Tensor [C, T]
      - meta: dict with keys: text_prompt (str), video_prompt (Tensor or None), audio_prompt (Tensor or None)
    """
    def __init__(self, hf_split, target_sr: int, seconds: float = 10.0):
        self.base = TTT_RecaptionedDataset(hf_split, sample_rate=DS_SR)
        self.target_sr = target_sr
        self.target_len = int(seconds * target_sr)

    def __len__(self):
        return len(self.base) * 3

    def __getitem__(self, idx: int):
        item_ix = idx // 3
        seg_ix = idx % 3
        waveforms, labels = self.base[item_ix]
        audio = waveforms[seg_ix]  # [1, T]

        # resample if needed
        if self.target_sr != DS_SR:
            audio = torchaudio.functional.resample(audio, orig_freq=DS_SR, new_freq=self.target_sr)

        audio = _pad_trunc(audio, self.target_len)

        meta = {
            "text_prompt": labels[seg_ix],
            # placeholders for optional modalities; we’ll stub in the module
            "video_prompt": None,
            "audio_prompt": None,
        }
        return audio, meta


def build_stub_conditioning(batch_meta: List[Dict[str, Any]], cond_dim: int = 768,
                            video_tokens: int = 1, text_tokens: int = 1, audio_tokens: int = 1,
                            device: torch.device = torch.device("cpu")) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Create minimal zero conditioning to avoid heavyweight encoders during smoke tests.
    Shapes:
      returns dict[id] -> (tensor [B, T, cond_dim], mask [B, T])
    """
    b = len(batch_meta)
    zeros = lambda t: torch.zeros((b, t, cond_dim), device=device)
    ones_mask = lambda t: torch.ones((b, t), dtype=torch.bool, device=device)

    cond = {
        "video_prompt": (zeros(video_tokens), ones_mask(video_tokens)),
        "text_prompt": (zeros(text_tokens), ones_mask(text_tokens)),
        "audio_prompt": (zeros(audio_tokens), ones_mask(audio_tokens)),
    }
    return cond


class TTTProjections(nn.Module):
    """Learned projections Theta_Q, Theta_K, Theta_V for outer-loop training.
    Returns (q, k, v) with shape [B, T, proj_dim].
    """
    def __init__(self, dim: int, proj_dim: int):
        super().__init__()
        self.theta_Q = nn.Linear(dim, proj_dim, bias=False)
        self.theta_K = nn.Linear(dim, proj_dim, bias=False)
        self.theta_V = nn.Linear(dim, proj_dim, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.theta_Q(h), self.theta_K(h), self.theta_V(h)


class TTTCondTrainingWrapper(pl.LightningModule):
    """Lightweight training loop mirroring stable_audio_tools DiffusionCondTrainingWrapper,
    with TTT gating + inner self-supervised loss.
    """
    def __init__(self, diffusion_wrapper: nn.Module, ttt: TTT, ttt_proj: TTTProjections,
                 sample_rate: int, lr: float = 5e-5, ttt_weight: float = 0.1,
                 use_stub_conditioning: bool = True, cond_dim: int = 768,
                 cfg_dropout_prob: float = 0.1,
                 collect_hidden_states: bool = False,
                 proj_weight: float = 0.05,
                 p_mean: float = -1.2, p_std: float = 1.0,
                 sigma_min: float = 0.002, sigma_max: float = 1.0):
        super().__init__()
        self.diffusion = diffusion_wrapper
        self.sample_rate = sample_rate
        self.lr = lr
        self.cfg_dropout_prob = cfg_dropout_prob

        self.ttt = ttt
        self.ttt_proj = ttt_proj
        self.ttt_weight = ttt_weight
        self.proj_weight = proj_weight

        # Sobol for timesteps
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        # logs
        self.train_hist = {"step": [], "loss": [], "diffusion": [], "ttt": []}

        self.use_stub_conditioning = use_stub_conditioning
        self.cond_dim = cond_dim
        self.collect_hidden_states = collect_hidden_states
        # K-diffusion sigma config (EDM)
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._denoiser = None  # lazy init

        # Freeze everything except TTT layers
        self.diffusion.requires_grad_(False)
        for p in self.diffusion.parameters():
            p.requires_grad = False
        # Projections are trainable in the outer loop (but frozen inside inner TTT loss by detaching their outputs)
        self.ttt_proj.requires_grad_(True)
        for p in self.ttt_proj.parameters():
            p.requires_grad = True
        # Respect requires_grad flags set on TTT layers (optionally adjusted in main)

        # Put frozen modules in eval to save memory
        self.diffusion.eval()

    def configure_optimizers(self):
        # Optimize TTT parameters and projection parameters
        params = []
        params += [p for p in self.ttt.parameters() if p.requires_grad]
        params += [p for p in self.ttt_proj.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-3)

    def _conditioning(self, batch_meta: List[Dict[str, Any]]):
        if self.use_stub_conditioning:
            return build_stub_conditioning(batch_meta, cond_dim=self.cond_dim, device=self.device)
        # Fallback: use model’s conditioner (costly). Expect batch_meta as required by MultiConditioner.
        return self.diffusion.conditioner(batch_meta, self.device)

    def training_step(self, batch, batch_ix):
        reals, meta = batch
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        # prepare conditioning
        cond = self._conditioning(meta)

        # Optional pretransform encode (e.g., autoencoder latents)
        diffusion_input = reals
        if getattr(self.diffusion, "pretransform", None) is not None:
            pt = self.diffusion.pretransform.to(self.device)
            pt.eval()
            with torch.no_grad():
                # match channels
                diffusion_input = set_audio_channels(diffusion_input, pt.io_channels)
                diffusion_input = pt.encode(diffusion_input)

        # K-diffusion (EDM) objective aligned with dpmpp-3m-sde sampling
        b = diffusion_input.shape[0]
        device = diffusion_input.device
        dtype = diffusion_input.dtype
        # sample log-normal sigmas as in EDM, clamp to [sigma_min, sigma_max]
        r = torch.randn(b, device=device, dtype=dtype) * self.p_std + self.p_mean
        sigma = r.exp().clamp(min=self.sigma_min, max=self.sigma_max)
        noise = torch.randn_like(diffusion_input)
        # Build conditioning inputs as used by inference
        cond_inputs = self.diffusion.get_conditioning_inputs(cond)
        # lazy-create VDenoiser on the underlying DiT wrapper
        if getattr(self, "_kdenoiser", None) is None:
            import k_diffusion as K
            self._kdenoiser = K.external.VDenoiser(self.diffusion.model)
        # diffusion loss from k-diffusion (mean over batch)
        diff_loss = self._kdenoiser.loss(
            diffusion_input, noise, sigma,
            cfg_dropout_prob=self.cfg_dropout_prob,
            ttt_module=self.ttt,
            **cond_inputs,
        ).mean()
        # Optional: compute hidden states using the same sigma/noise for inner/outer TTT losses
        info = {}
        if self.collect_hidden_states:
            # rebuild the same noised input consistent with EDM preconditioning
            alpha = 1.0 / torch.sqrt(1.0 + sigma**2)
            sigma_v = torch.sqrt(torch.clamp(1.0 - alpha**2, min=0.0))
            noised = alpha[:, None, None] * diffusion_input + sigma_v[:, None, None] * noise
            # forward on the DiT wrapper with sigma as t, return_info=True
            out2, info = self.diffusion.model(
                noised, sigma,
                return_info=True,
                cfg_dropout_prob=self.cfg_dropout_prob,
                ttt_module=self.ttt,
                **cond_inputs,
            )

        # ttt self-supervised loss from hidden states
        # info["hidden_states"]: list of [B, T, dim]; ensure exists
        h_list = info.get("hidden_states", []) if self.collect_hidden_states else []
        ttt_loss = torch.tensor(0.0, device=self.device)
        proj_outer_loss = torch.tensor(0.0, device=self.device)
        if len(h_list) > 0:
            # choose mid layer to form targets; or average all
            layer_losses = []
            proj_losses = []
            for h in h_list:
                # Inner loss: freeze Theta_* by detaching their outputs
                with torch.no_grad():
                    q_i, k_i, v_i = self.ttt_proj(h)
                # Use K,V only for inner TTT self-supervision
                ttt_loss_i = self.ttt.self_supervised_forward(k_i, v_i, use_ttt_prime=True)
                layer_losses.append(ttt_loss_i)

                # Outer loss: train Theta_* while not updating TTT weights
                # Compute fresh projections with grad
                q_o, k_o, v_o = self.ttt_proj(h)
                # Align reversed K to V and encourage Q to align with K/V
                k_rev = torch.flip(k_o, dims=[1])
                loss_kv = F.mse_loss(k_rev, v_o)
                loss_qk = F.mse_loss(q_o, k_o)
                loss_qv = F.mse_loss(q_o, v_o)
                proj_losses.append(loss_kv + 0.5 * (loss_qk + loss_qv))

            ttt_loss = torch.stack(layer_losses).mean()
            proj_outer_loss = torch.stack(proj_losses).mean()

        loss = diff_loss + self.ttt_weight * ttt_loss + self.proj_weight * proj_outer_loss

        # log
        self.log_dict({
            "train/loss": loss.detach(),
            "train/diffusion": diff_loss.detach(),
            "train/ttt": ttt_loss.detach(),
            "train/proj": proj_outer_loss.detach(),
        }, prog_bar=True, on_step=True)

        self.train_hist["step"].append(self.global_step)
        self.train_hist["loss"].append(loss.item())
        self.train_hist["diffusion"].append(diff_loss.item())
        self.train_hist["ttt"].append(ttt_loss.item())
        if isinstance(proj_outer_loss, torch.Tensor):
            self.train_hist.setdefault("proj", []).append(proj_outer_loss.item())

        return loss

    def on_train_end(self):
        # optional plotting
        try:
            import matplotlib.pyplot as plt
            if len(self.train_hist["step"]) == 0:
                return
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(self.train_hist["step"], self.train_hist["loss"], label="loss")
            ax.plot(self.train_hist["step"], self.train_hist["diffusion"], label="diffusion")
            ax.plot(self.train_hist["step"], self.train_hist["ttt"], label="ttt")
            ax.set_xlabel("step")
            ax.set_ylabel("loss")
            ax.legend()
            fig.tight_layout()
            out_path = os.path.join(self.trainer.default_root_dir or ".", "ttt_training_curves.png")
            fig.savefig(out_path)
            print(f"Saved training curves to {out_path}")
        except Exception as e:
            print(f"Plotting failed: {e}")


def collate_batch(batch: List[Tuple[torch.Tensor, Dict[str, Any]]], target_len: int) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    audios = []
    metas = []
    for audio, meta in batch:
        audios.append(_pad_trunc(audio, target_len))
        metas.append(meta)
    audio_batch = torch.stack(audios, dim=0)  # [B, 1, T]
    return audio_batch, metas


def load_local_audiox(model_dir: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model and config from local directory (model/config.json, model/model.*)."""
    import json
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r") as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    # try safetensors then ckpt
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model", help="Local AudioX model dir with config.json and weights")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20, help="For smoke test")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--use_stub_conditioning", action="store_true", help="Use zero-conditioners to avoid heavy downloads")
    parser.add_argument("--ttt_weight", type=float, default=0.1)
    parser.add_argument("--proj_weight", type=float, default=0.05)
    parser.add_argument("--collect_hidden_states", action="store_true", help="Collect hidden states for TTT loss (uses more memory)")
    parser.add_argument("--train_last_ttt_layers", type=int, default=None, help="If set, only the last N TTT layers are trainable; earlier ones are frozen")
    args = parser.parse_args()

    # load local dataset prepared in data_loader
    from data_loader import dataset as hf_dataset
    # Drop raw 'audio' column to avoid librosa/numba dependency in workers
    for split in list(hf_dataset.keys()):
        try:
            if 'audio' in hf_dataset[split].features:
                hf_dataset[split] = hf_dataset[split].remove_columns(['audio'])
        except Exception:
            pass

    # load local AudioX
    model, model_cfg = load_local_audiox(args.model_dir)

    sample_rate = model_cfg.get("sample_rate", 44100)
    sample_size = model_cfg.get("sample_size", int(10 * sample_rate))

    # Build dataset/loader
    train_ds = TTTRecapTrain(hf_dataset["train"], target_sr=sample_rate, seconds=10.0)
    dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_batch(b, target_len=int(10.0 * sample_rate)),
        pin_memory=True,
    )

    # Introspect transformer to construct TTT
    # ConditionedDiffusionModelWrapper -> DiTWrapper -> DiffusionTransformer -> ContinuousTransformer
    dit = model.model  # DiTWrapper
    # underlying DiffusionTransformer
    diffusion_transformer = dit.model
    ct = diffusion_transformer.transformer  # ContinuousTransformer
    dim = ct.dim
    depth = ct.depth

    ttt_module = TTT(num_layers=depth, dim=dim)
    # Use proj_dim == transformer dim so the TTT MLP shapes match during the self-supervised loss
    ttt_proj = TTTProjections(dim=dim, proj_dim=dim)

    # Optionally freeze early TTT layers to save memory
    if args.train_last_ttt_layers is not None:
        k = max(0, depth - args.train_last_ttt_layers)
        for i, layer in enumerate(ttt_module.layers):
            req = i >= k
            for p in layer.parameters():
                p.requires_grad = req

    lit = TTTCondTrainingWrapper(
        diffusion_wrapper=model,
        ttt=ttt_module,
        ttt_proj=ttt_proj,
        sample_rate=sample_rate,
        lr=5e-5,
        ttt_weight=args.ttt_weight,
        use_stub_conditioning=args.use_stub_conditioning,
        cond_dim=model_cfg["model"]["conditioning"]["cond_dim"],
        collect_hidden_states=args.collect_hidden_states,
        proj_weight=args.proj_weight,
    )

    # Two-GPU ready training; for smoke test limit steps
    # Optional: use faster matmul precision on Tensor Cores
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        strategy=args.strategy,
        max_steps=args.max_steps,
        gradient_clip_val=0.1,
        log_every_n_steps=1,
    )

    trainer.fit(lit, train_dataloaders=dl)


if __name__ == "__main__":
    main()
