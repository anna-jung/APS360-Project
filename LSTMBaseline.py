import os
import torch
import torch.nn as nn
import torch.optim as optim
from encodec import EncodecModel
import torchaudio

# We get the dataloaders from dataloaders.py first
from data_loader import train_loader, val_loader, test_loader


SAMPLE_RATE     = 16_000
TARGET_DURATION = 20.0
TARGET_LENGTH   = int(SAMPLE_RATE * TARGET_DURATION)

# Check if you can find a GPU, else use CPU only
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EnCodec Model instantiated
encodec_model = EncodecModel.encodec_model_24khz().to(device).eval()
latent_dim    = encodec_model.quantizer.codebook_size  # e.g. 1024


### The LSTM Class Definition ###
class LatentLSTM(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512, num_layers=2):
        super().__init__()
        self.lstm   = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc_out(out)
        return out, hidden


def train(model, train_loader, device,
          epochs=20, teacher_forcing=0.5, lr=1e-3):
    
    # opimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for waveform, _ in train_loader:

            # Trim length
            waveform = waveform[:, :, :TARGET_LENGTH]
            waveform = waveform.to(device)
            
            # Bring to latent space
            with torch.no_grad():
                lat_seq, _, _ = encodec_model.encode(waveform)

            # sequences ready.
            inp  = lat_seq[:, :-1, :]
            targ = lat_seq[:,  1:, :]

            # autoregressive loop for generation
            outputs = []
            hidden  = None
            prev    = inp[:, 0:1, :]
            for t in range(inp.size(1)):
                out, hidden = model(prev, hidden)
                outputs.append(out)
                if torch.rand(1) < teacher_forcing:
                    prev = inp[:, t+1:t+2, :]
                else:
                    prev = out

            # backprop updates and optimization
            outputs = torch.cat(outputs, dim=1)
            loss    = criterion(outputs, targ)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} — Train Loss: {avg:.4f}")


def sample(model, init_latent, length, device):
    model.eval()
    generated = []
    hidden    = None
    prev      = init_latent.to(device).unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        for _ in range(length):
            out, hidden = model(prev, hidden)
            generated.append(out)
            prev = out
    gen_seq = torch.cat(generated, dim=1) 
    return gen_seq.squeeze(0) 


model = LatentLSTM(latent_dim).to(device)
train(model, train_loader, device)

# pull a batch from the loader and trim it to the desired length
wave, _ = next(iter(train_loader))
wave = wave[..., :TARGET_LENGTH]

wave = wave.to(device)

# encode the first latent of the batch with EnCodec
with torch.no_grad():
    latents, _, _ = encodec_model.encode(wave)

# Batch and Time 0 to be used as seed
seed = latents[0, 0]

# Generate latents and decode back to the waveform
gen = sample(model, seed, gen_length=1000, device=device)
wav, sr = encodec_model.decode(gen.unsqueeze(0))
torchaudio.save("generated.wav", wav.cpu(), sr)
print(f"Saved generated.wav")
