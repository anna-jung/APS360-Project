import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as transforms
from datasets import load_dataset, Audio
from torch.utils.data import Dataset, DataLoader
import os

PROCESSED_DATA_DIR = "./processed_fsd50k"
if os.path.exists(PROCESSED_DATA_DIR):
    print(f"Loading processed dataset from {PROCESSED_DATA_DIR}")
    dataset = load_dataset(PROCESSED_DATA_DIR)

else:
    print("Processed dataset not found, creating from FSD50K")
    TARGET_DURATION = 20.0  # seconds
    SAMPLE_RATE = 16000
    TARGET_LENGTH = int(TARGET_DURATION * SAMPLE_RATE)

    dataset = load_dataset("CLAPv2/FSD50K")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


    def is_valid(sample):
        return sample['audio']['array'].size >= TARGET_LENGTH

    def normalize_audio(sample):
        audio = sample['audio']['array']
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        sample['audio_normalized'] = audio.astype(np.float32)
        return sample

    def add_gaussian_noise(sample, noise_level=0.005):
        audio = np.array(sample['audio_normalized'], dtype=np.float32)
        noise = np.random.normal(0, noise_level, size=audio.shape).astype(np.float32)
        audio_noisy = audio + noise
        sample['audio_noisy'] = np.clip(audio_noisy, -1.0, 1.0)
        return sample

    class AudioDataset(Dataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            waveform = torch.tensor(item['audio_noisy']).unsqueeze(0)  # [1, T]
            label = item.get('label', -1)
            return waveform, label

    dataset["train"] = dataset["train"].filter(is_valid)
    dataset["validation"] = dataset["validation"].filter(is_valid)
    dataset["test"] = dataset["test"].filter(is_valid)

    print("Train size:", len(dataset["train"]))
    print("Validation size:", len(dataset["validation"]))
    print("Test size:", len(dataset["test"]))

    dataset["train"] = dataset["train"].map(normalize_audio)
    dataset["train"] = dataset["train"].map(add_gaussian_noise)
    dataset["validation"] = dataset["validation"].map(normalize_audio)
    dataset["validation"] = dataset["validation"].map(add_gaussian_noise)
    dataset["test"] = dataset["test"].map(normalize_audio)
    dataset["test"] = dataset["test"].map(add_gaussian_noise)

    dataset.save_to_disk("./processed_fsd50k")

train_dataset = AudioDataset(dataset['train'])
val_dataset = AudioDataset(dataset['validation'])
test_dataset = AudioDataset(dataset['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)