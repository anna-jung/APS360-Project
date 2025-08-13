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

class AudioDataset(Dataset):
    def __init__(self, hf_dataset, audio_len=-1):
        self.dataset = hf_dataset
        self.audio_len = audio_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform = torch.tensor(item['audio_noisy']).unsqueeze(0)  # [1, T]
        if self.audio_len > 0:
            waveform = waveform[:, :self.audio_len]
        label = item.get('text', -1)
        return waveform, label

PROCESSED_DATA_DIR = "./processed_fsd50k"
TARGET_DURATION = 25.0  # seconds
SAMPLE_RATE = 16000
TARGET_LENGTH = int(TARGET_DURATION * SAMPLE_RATE)
if os.path.exists(PROCESSED_DATA_DIR):
    print(f"Loading processed dataset from {PROCESSED_DATA_DIR}")
    dataset = load_dataset(PROCESSED_DATA_DIR)

else:
    print("Processed dataset not found, creating from FSD50K")

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


# train_dataset = AudioDataset(dataset['train'], audio_len=TARGET_LENGTH)
# val_dataset = AudioDataset(dataset['validation'], audio_len=TARGET_LENGTH)
# test_dataset = AudioDataset(dataset['test'], audio_len=TARGET_LENGTH)
#
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Test Time Training Naive Captioned Dataset
class TTT_NaiveCaptionedDataset(Dataset):
    """
    Dataset for Test Time Training (TTT) with naive captioning and segmenting.
    Each item returns:
        - waveforms: list of 3 waveforms (10s each, padded if needed)
        - labels: list of 3 captions, prefixed with segment info
    """
    def __init__(self, hf_dataset, sample_rate=16000, min_duration=30.0, segment_duration=10.0):
        self.dataset = hf_dataset
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.segment_duration = segment_duration
        self.min_length = int(min_duration * sample_rate)
        self.segment_length = int(segment_duration * sample_rate)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform = torch.tensor(item['audio_noisy'])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        # Pad to at least 30s
        length = waveform.shape[1]
        if length < self.min_length:
            pad_len = self.min_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        # Take first 30s
        waveform = waveform[:, :self.min_length]
        # Split into 3 segments of 10s
        waveforms = []
        labels = []
        caption = item.get('text', '')
        for i in range(3):
            start = i * self.segment_length
            end = start + self.segment_length
            segment = waveform[:, start:end]
            waveforms.append(segment)
            labels.append(f"Segment {i+1} of 3: {caption}")
        return waveforms, labels


class TTT_RecaptionedDataset(Dataset):
    """
    Dataset for Test Time Training using pre-generated captions per segment.
    Expects columns: 'text1', 'text2', 'text3' in each item.
    Returns per item:
      - waveforms: list of 3 tensors [1, 10s] each (from first 30s, padded if needed)
      - labels: [text1, text2, text3] (fallback to empty string if missing)
    """
    def __init__(self, hf_dataset, sample_rate=16000, min_duration=30.0, segment_duration=10.0):
        self.dataset = hf_dataset
        self.sample_rate = sample_rate
        self.min_length = int(min_duration * sample_rate)
        self.segment_length = int(segment_duration * sample_rate)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform = torch.tensor(item['audio_noisy'])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        # Pad/truncate to 30s
        if waveform.shape[1] < self.min_length:
            pad_len = self.min_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        waveform = waveform[:, :self.min_length]

        # Split into 3x10s segments
        waveforms = []
        for i in range(3):
            start = i * self.segment_length
            end = start + self.segment_length
            waveforms.append(waveform[:, start:end])

        # Captions from text1/2/3
        def _as_str(x):
            if x is None:
                return ""
            return str(x)

        labels = [
            _as_str(item.get('text1', '')),
            _as_str(item.get('text2', '')),
            _as_str(item.get('text3', '')),
        ]
        return waveforms, labels

if __name__ == "__main__":
    # Test for TTT_NaiveCaptionedDataset
    ttt_train_dataset = TTT_NaiveCaptionedDataset(dataset['train'], sample_rate=SAMPLE_RATE)
    print("First 3 items in TTT_NaiveCaptionedDataset (train):")
    for i in range(3):
        waveforms, labels = ttt_train_dataset[i]
        print(f"Item {i}:")
        for j in range(3):
            print(f"  Segment {j+1} shape: {waveforms[j].shape}, Caption: {labels[j]}")
