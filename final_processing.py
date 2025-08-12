import os
import shutil
import tempfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import io
import soundfile as sf
from datasets import load_dataset, Audio
import torch.nn.functional as F

SAMPLE_RATE = 16000
DURATION_THRESHOLD = 25.0
NOISE_STD = 0.005
BATCH_SIZE = 16
SEED = 42

def preprocess_split_to_disk(dataset_name, split, temp_dir):
    """
    Processes audio from a dataset split and saves each waveform to a temporary directory.
    Returns a list of paths to the saved files.
    """
    print(f"Processing split '{split}'...")
    ds_stream = load_dataset(dataset_name, split=split, streaming=True)
    ds_stream = ds_stream.cast_column("audio", Audio(decode=False))

    processed_file_paths = []
    count = 0
    for item in ds_stream:
        audio_bytes = item["audio"]["bytes"]
        byte_data = bytes(audio_bytes)
        try:
            with io.BytesIO(byte_data) as f:
                info = sf.info(f)
                duration = info.duration
            if duration < DURATION_THRESHOLD:
                continue

            with io.BytesIO(byte_data) as f:
                waveform_np, sr = sf.read(f, dtype='float32')
            waveform = torch.from_numpy(waveform_np)

            # Mono mix if multi-channel
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)

            # Resample if needed
            if sr != SAMPLE_RATE:
                waveform = F.interpolate(
                    waveform.unsqueeze(0).unsqueeze(0),
                    scale_factor=SAMPLE_RATE / sr,
                    mode='linear',
                    align_corners=False
                ).squeeze()

            # Normalize
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

            # Add Gaussian noise
            waveform = waveform + NOISE_STD * torch.randn_like(waveform)

            # Clip between -1 and 1
            waveform = waveform.clamp(-1.0, 1.0)

            # Save the processed waveform to a file
            file_path = os.path.join(temp_dir, f"{split}_{count}.pt")
            torch.save(waveform, file_path)
            processed_file_paths.append(file_path)

            count += 1
            if count % 100 == 0:
                print(f"Processed {count} audios in '{split}'")

        except Exception:
            continue

    print(f"Processed {count} valid audios in split '{split}'")
    return processed_file_paths

class DiskAudioDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the tensor from disk instead of memory
        return torch.load(self.file_paths[idx])
    
    def create_train_val_test_loaders(dataset_name, batch_size=BATCH_SIZE, seed=SEED):
    # Use a temporary directory to store processed audio files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")

    try:
        dataset = load_dataset(dataset_name, split=None, streaming=True)
        splits = list(dataset.keys())
        print(f"Found splits: {splits}")

        all_file_paths = []
        for split in splits:
            file_paths = preprocess_split_to_disk(dataset_name, split, temp_dir)
            all_file_paths.extend(file_paths)

        print(f"Total combined audios: {len(all_file_paths)}")

        # Create a dataset that loads from disk
        full_dataset = DiskAudioDataset(all_file_paths)

        # Compute sizes for splits (80/10/10)
        n = len(full_dataset)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val

        # Random split with fixed seed
        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader, test_loader

    finally:
        print("Clean up the temporary directory and its contents")
        # shutil.rmtree(temp_dir)

# The `collate_fn` remains the same
def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

# Usage example:
HF_dataset = 'nateraw/fsd50k'
train_loader, val_loader, test_loader = create_train_val_test_loaders(HF_dataset)

for batch in train_loader:
    print("Train batch shape:", batch.shape)
    break

for batch in val_loader:
    print("Val batch shape:", batch.shape)
    break

for batch in test_loader:
    print("Test batch shape:", batch.shape)
    break

# dont forget to clean the temporary dir after all is done including running model