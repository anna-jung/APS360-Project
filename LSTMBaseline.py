
import os
import torch
import torch.nn as nn
import torch.optim as optim
from encodec import EncodecModel
import torchaudio
from torch.utils.data import DataLoader
from data_loader import AudioDataset, dataset
import torch.nn.functional as F
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
# Use 24kHz sample rate to match EnCodec model
SAMPLE_RATE = 24000
TARGET_DURATION = 30
TARGET_LENGTH   = int(SAMPLE_RATE * TARGET_DURATION)

# Initialize Accelerator for multi-GPU training
accelerator = Accelerator()
device = accelerator.device

# EnCodec Model instantiated
encodec_model = EncodecModel.encodec_model_24khz().to(device).eval()


class AudioTokenizer(nn.Module):
    """Tokenizer that embeds 32 integers (codebooks) into a vector"""
    def __init__(self, num_codebooks=32, vocab_size=1024, embed_dim=256):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Create embedding for each codebook
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim // num_codebooks) 
            for _ in range(num_codebooks)
        ])
        
    def forward(self, tokens):
        # tokens: [B, num_codebooks, T] -> [B, T, embed_dim]
        B, C, T = tokens.shape
        assert C == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {C}"
        
        # Embed each codebook separately
        embeddings = []
        for i in range(self.num_codebooks):
            emb = self.codebook_embeddings[i](tokens[:, i, :])  # [B, T, embed_dim//num_codebooks]
            embeddings.append(emb)
        
        # Concatenate all codebook embeddings
        combined = torch.cat(embeddings, dim=-1)  # [B, T, embed_dim]
        return combined


class LatentLSTMTextCond(nn.Module):
    def __init__(self, num_codebooks=32, vocab_size=1024, audio_embed_dim=512, 
                 text_emb_dim=256, hidden_dim=512, num_layers=2, text_vocab_size=10000):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        
        # Audio tokenizer
        self.audio_tokenizer = AudioTokenizer(num_codebooks, vocab_size, audio_embed_dim)
        
        # Text embedding
        self.text_emb = nn.Embedding(text_vocab_size, text_emb_dim)
        
        # LSTM
        self.lstm = nn.LSTM(audio_embed_dim + text_emb_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output projection - predict logits for each codebook
        self.output_projections = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(num_codebooks)
        ])

    def forward(self, x, text_tokens, hidden=None):
        # x: [B, num_codebooks, T], text_tokens: [B, text_len]
        B, C, T = x.shape
        
        # Embed audio tokens
        x_emb = self.audio_tokenizer(x)  # [B, T, audio_embed_dim]
        
        # Embed text
        text_emb = self.text_emb(text_tokens)  # [B, text_len, text_emb_dim]
        
        # Repeat text embedding to match sequence length
        if text_emb.shape[1] != T:
            # Take mean of text embeddings and repeat
            text_emb = text_emb.mean(dim=1, keepdim=True).repeat(1, T, 1)  # [B, T, text_emb_dim]
        
        # Concatenate audio and text embeddings
        lstm_in = torch.cat([x_emb, text_emb], dim=-1)  # [B, T, audio_embed_dim + text_emb_dim]
        
        # LSTM forward pass
        out, hidden = self.lstm(lstm_in, hidden)  # [B, T, hidden_dim]
        
        # Predict logits for each codebook
        logits = []
        for i in range(self.num_codebooks):
            codebook_logits = self.output_projections[i](out)  # [B, T, vocab_size]
            logits.append(codebook_logits)
        
        # Stack logits: [B, num_codebooks, T, vocab_size]
        logits = torch.stack(logits, dim=1)
        
        return logits, hidden



def sample(model, init_tokens, text_tokens, length, device, temperature=1.0):
    """Generate audio tokens autoregressively"""
    model.eval()
    B, C = init_tokens.shape[:2]  # [B, num_codebooks]
    
    # Initialize with starting tokens
    generated = init_tokens.unsqueeze(-1)  # [B, C, 1]
    hidden = None
    
    with torch.no_grad():
        for _ in range(length):
            # Get predictions for current sequence
            logits, hidden = model(generated, text_tokens, hidden)  # [B, C, T, vocab_size]
            
            # Take logits for the last timestep
            next_logits = logits[:, :, -1, :]  # [B, C, vocab_size]
            
            # Sample from distribution with temperature
            if temperature > 0:
                next_logits = next_logits / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, C, 1)
            else:
                next_tokens = next_logits.argmax(dim=-1).unsqueeze(-1)  # [B, C, 1]
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=-1)  # [B, C, T+1]
            
    return generated[:, :, 1:]  # Remove initial token, return [B, C, length]


def tokenize_texts(texts, max_len=30, tokenizer=None):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    enc = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    return enc['input_ids']

def load_checkpoint(model, optimizer, accelerator, checkpoint_path):
    """Load checkpoint for resuming training"""
    if os.path.exists(checkpoint_path):
        if accelerator.is_local_main_process:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load the accelerator state
        accelerator.load_state(checkpoint_path)
        
        # Extract epoch number from checkpoint path
        epoch_num = int(checkpoint_path.split('_')[-1].split('.')[0])
        
        if accelerator.is_local_main_process:
            print(f"Resumed from epoch {epoch_num}")
            
        return epoch_num + 1  # Return next epoch to start from
    else:
        if accelerator.is_local_main_process:
            print("No checkpoint found, starting from scratch")
        return 1

def train(model, train_loader, optimizer, accelerator, 
          epochs=50, teacher_forcing=0.5, checkpoint_dir="checkpoints"):
    criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for discrete tokens
    model.train()
    
    # Create checkpoint directory
    if accelerator.is_local_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to {checkpoint_dir}/")
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", disable=not accelerator.is_local_main_process)
        
        for batch_idx, (waveform, labels) in enumerate(pbar):
            waveform = waveform[:, :, :TARGET_LENGTH]
            
            # Resample from 16kHz to 24kHz for EnCodec
            if waveform.shape[-1] > 0:
                resampler = torchaudio.transforms.Resample(16000, 24000).to(accelerator.device)
                waveform = resampler(waveform)
            
            texts = [str(lbl) for lbl in labels]
            text_tokens = tokenize_texts(texts, max_len=30).to(accelerator.device)
            
            # Encode to discrete tokens
            with torch.no_grad():
                encoded_frames = encodec_model.encode(waveform)
                # Extract the discrete codes: shape should be [B, num_codebooks, T]
                codes = encoded_frames[0][0]  # First element is codes
                
                # Debug: print shapes for first batch on main process only
                if batch_idx == 0 and accelerator.is_local_main_process and epoch == 1:
                    print(f"Debug - codes shape: {codes.shape}")
                
            # Prepare input and target sequences
            inp = codes[:, :, :-1]    # [B, C, T-1] - input sequence
            targ = codes[:, :, 1:]    # [B, C, T-1] - target sequence (shifted by 1)
            
            B, C, T = inp.shape
            
            # Process in larger but still manageable chunks
            chunk_size = min(256, T)  # Manageable chunks for multi-GPU
            total_loss_batch = 0.0
            num_chunks = 0
            
            for start_idx in range(0, T, chunk_size):
                end_idx = min(start_idx + chunk_size, T)
                chunk_inp = inp[:, :, start_idx:end_idx]
                chunk_targ = targ[:, :, start_idx:end_idx]
                
                # Forward pass on chunk
                logits, _ = model(chunk_inp, text_tokens)  # [B, C, chunk_size, vocab_size]
                
                # Reshape for loss calculation
                # logits: [B, C, chunk_size, vocab_size] -> [B*C*chunk_size, vocab_size]
                # targets: [B, C, chunk_size] -> [B*C*chunk_size]
                logits_flat = logits.reshape(-1, logits.shape[-1])
                targets_flat = chunk_targ.reshape(-1)
                
                # Calculate loss
                loss = criterion(logits_flat, targets_flat)
                
                # Backpropagation with Accelerate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss_batch += loss.item()
                num_chunks += 1
                
            avg_batch_loss = total_loss_batch / num_chunks if num_chunks > 0 else 0
            total_loss += avg_batch_loss
            
            if accelerator.is_local_main_process:
                pbar.set_postfix({'loss': avg_batch_loss, 'seq_len': T})
            
        avg = total_loss / len(train_loader)
        if accelerator.is_local_main_process:
            print(f"Epoch {epoch}/{epochs} — Train Loss: {avg:.4f}")
            
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            if accelerator.is_local_main_process:
                print(f"Saving checkpoint at epoch {epoch}...")
                
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            
            # Save checkpoint using accelerator
            accelerator.save_state(checkpoint_path)
            
            # Also save model weights separately for easier loading
            if accelerator.is_local_main_process:
                model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg,
                }, model_path)
                print(f"Checkpoint saved: {checkpoint_path}")
                print(f"Model weights saved: {model_path}")


if __name__ == "__main__":
    train_dataset = AudioDataset(dataset['train'], audio_len=TARGET_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)  # Reduced num_workers for multi-GPU

    # Use a pretrained HuggingFace tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Initialize model with correct parameters - reduced dimensions for longer sequences
    model = LatentLSTMTextCond(
        num_codebooks=32, 
        vocab_size=1024,
        audio_embed_dim=512,  # Reduced from 512 to save memory
        hidden_dim=512,       # Reduced from 512 to save memory
        text_vocab_size=tokenizer.vocab_size
    )

    # Setup optimizer and prepare for multi-GPU training
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Prepare model, optimizer, and dataloader with accelerate
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Optional: Load from checkpoint to resume training
    # Uncomment the next lines if you want to resume from a specific checkpoint
    # resume_checkpoint = "checkpoints/checkpoint_epoch_5.pt"
    # start_epoch = load_checkpoint(model, optimizer, accelerator, resume_checkpoint)

    # Train the model for 50 epochs
    train(model, train_loader, optimizer, accelerator, epochs=50)
