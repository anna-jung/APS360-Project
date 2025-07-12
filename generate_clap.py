# clap_score.py

import torch
import torchaudio
from laion_clap import CLAP_Module
import argparse
import os

def load_model(device="cuda"):
    model = CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    model.to(device)
    model.eval()
    return model

def encode_text(model, text, device="cuda"):
    with torch.no_grad():
        return model.get_text_embedding([text], use_tensor=True).to(device)

def encode_audio(model, audio_path, device="cuda"):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
        waveform = resampler(waveform)
    if waveform.shape[0] == 2:
        waveform = waveform.mean(0, keepdim=True)
    with torch.no_grad():
        return model.get_audio_embedding_from_data(x=waveform.to(device), use_tensor=True)

def compute_clap_score(text_embedding, audio_embedding):
    return torch.nn.functional.cosine_similarity(text_embedding, audio_embedding).item()

def main(audio_path, text_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.isfile(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    model = load_model(device)
    print("Model loaded.")

    text_emb = encode_text(model, text_prompt, device)
    audio_emb = encode_audio(model, audio_path, device)
    score = compute_clap_score(text_emb, audio_emb)
    print(f"\nCLAP score between:\n  Text: \"{text_prompt}\"\n  Audio: {audio_path}\n→ Score: {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to the .wav file")
    parser.add_argument("--text", type=str, required=True, help="Text prompt")
    args = parser.parse_args()

    main(args.audio, args.text)
