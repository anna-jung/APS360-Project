#!/usr/bin/env python3
"""
Autoregressive Audio Generation Script
Generates 30 seconds of audio from a text prompt using the trained LSTM model.
"""

import os
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm

# Import model classes and utilities from LSTMBaseline
from LSTMBaseline import (
    AudioTokenizer,
    LatentLSTMTextCond, 
    tokenize_texts,
    encodec_model,
    SAMPLE_RATE,
    device
)

TARGET_DURATION = 30.0  # Generate 30 seconds of audio
print(f"Using device: {device}")


def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create model with same parameters as training
    model = LatentLSTMTextCond(
        num_codebooks=32, 
        vocab_size=1024,
        audio_embed_dim=512,
        hidden_dim=512,
        text_vocab_size=tokenizer.vocab_size
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    print(f"Training loss was: {checkpoint['loss']:.4f}")
    
    return model, tokenizer


def generate_audio_autoregressive(model, text_prompt, tokenizer, duration_seconds=30, 
                                temperature=1.0, top_k=50, device='cpu'):
    """
    Generate audio autoregressively from text prompt
    
    Args:
        model: Trained LSTM model
        text_prompt: Text description of desired audio
        tokenizer: Text tokenizer
        duration_seconds: Length of audio to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (0 = disabled)
        device: torch device
    """
    model.eval()
    
    # Tokenize text prompt using the same function as training
    text_tokens = tokenize_texts([text_prompt], max_len=30).to(device)
    print(f"Text prompt: '{text_prompt}'")
    print(f"Text tokens shape: {text_tokens.shape}")
    
    # Calculate target length in tokens
    # Based on our training: ~3374 tokens for 30 seconds
    tokens_per_second = 3374 / 30.0  # approximately 112.5 tokens per second
    target_tokens = int(duration_seconds * tokens_per_second)
    print(f"Target tokens to generate: {target_tokens}")
    
    # Initialize with silence/random tokens for the first few timesteps
    batch_size = 1
    num_codebooks = 32
    seed_length = 4  # Start with 4 tokens
    
    # Initialize with random tokens (could also use zeros or learned start tokens)
    generated_tokens = torch.randint(0, 1024, (batch_size, num_codebooks, seed_length), device=device)
    
    hidden = None
    
    print("Generating audio tokens...")
    with torch.no_grad():
        for step in tqdm(range(target_tokens - seed_length), desc="Generating"):
            # Get current sequence
            current_seq = generated_tokens
            
            # Forward pass through model
            logits, hidden = model(current_seq, text_tokens, hidden)
            
            # Take logits for the last timestep
            next_logits = logits[:, :, -1, :]  # [B, num_codebooks, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Apply top-k sampling if specified
            if top_k > 0:
                # Get top-k values and indices
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                # Create a mask for top-k values
                mask = torch.full_like(next_logits, float('-inf'))
                mask.scatter_(-1, top_k_indices, top_k_logits)
                next_logits = mask
            
            # Sample next tokens
            probs = F.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(batch_size, num_codebooks, 1)
            
            # Append to generated sequence
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=-1)
            
            # Optional: Reset hidden state periodically to avoid memory issues
            if step % 500 == 0 and step > 0:
                hidden = None
    
    print(f"Generated {generated_tokens.shape[-1]} tokens")
    return generated_tokens


def decode_tokens_to_audio(tokens, sample_rate=24000):
    """Decode generated tokens back to audio waveform using the imported encodec_model"""
    print("Decoding tokens to audio...")
    
    with torch.no_grad():
        # EnCodec expects format: [(codes, scale)]
        encoded_audio = [(tokens, None)]
        
        # Decode to audio using the imported encodec_model
        audio_waveform = encodec_model.decode(encoded_audio)
        
        # audio_waveform is [B, channels, samples]
        audio_waveform = audio_waveform.squeeze(0)  # Remove batch dimension
        
        if audio_waveform.dim() == 1:  # If mono, add channel dimension
            audio_waveform = audio_waveform.unsqueeze(0)
    
    return audio_waveform


def main():
    parser = argparse.ArgumentParser(description='Generate audio from text prompt')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_epoch_50.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, 
                        default='A dog barks violently followed by a cat meowing quietly',
                        help='Text prompt for audio generation')
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Duration of audio to generate (seconds)')
    parser.add_argument('--output', type=str, default='generated_audio.wav',
                        help='Output audio file path')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling (0 = disabled)')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.checkpoint, device)
    
    # Generate audio tokens
    generated_tokens = generate_audio_autoregressive(
        model=model,
        text_prompt=args.prompt,
        tokenizer=tokenizer,
        duration_seconds=args.duration,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    
    # Decode to audio using the imported encodec_model
    audio_waveform = decode_tokens_to_audio(generated_tokens)
    
    # Save audio file
    print(f"Saving audio to {args.output}")
    torchaudio.save(args.output, audio_waveform.cpu(), SAMPLE_RATE)
    
    print(f"Generated audio saved!")
    print(f"Audio shape: {audio_waveform.shape}")
    print(f"Duration: {audio_waveform.shape[-1] / SAMPLE_RATE:.2f} seconds")


if __name__ == "__main__":
    main()
