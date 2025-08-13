
# Installation:
# python -m pip install conette

import os
import torch
from datasets import load_dataset
from data_loader import TTT_NaiveCaptionedDataset, SAMPLE_RATE, PROCESSED_DATA_DIR
from conette import CoNeTTEConfig, CoNeTTEModel
from tqdm import tqdm

device = torch.device('cuda:0')
# Load processed dataset
print(f"Loading processed dataset from {PROCESSED_DATA_DIR}")
dataset = load_dataset(PROCESSED_DATA_DIR)
splits = ["train", "validation", "test"]

# Load CoNeTTE model
config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
model = CoNeTTEModel(config, device)

state_dict = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/Labbeti/conette/resolve/main/pytorch_model.bin",
    map_location="cpu"
)
model.load_state_dict(state_dict)

model = model.to(device)

def process_split(split_name: str, batch_size_samples: int = 16):
    ds_split = dataset[split_name]
    ttt_ds = TTT_NaiveCaptionedDataset(ds_split, sample_rate=SAMPLE_RATE)

    n = len(ttt_ds)
    text1 = [""] * n
    text2 = [""] * n
    text3 = [""] * n

    pbar = tqdm(range(0, n, batch_size_samples), desc=f"Captioning {split_name}")
    for start in pbar:
        end = min(start + batch_size_samples, n)
        # Build a flat list of segments (3 per sample)
        segs = []
        for idx in range(start, end):
            waveforms, _ = ttt_ds[idx]
            segs.extend(waveforms)  # [seg1, seg2, seg3]

        # Run model on all segments at once
        with torch.no_grad():
            outputs = model(segs, sr=[SAMPLE_RATE] * len(segs))
        cands = outputs["cands"]  # length == 3 * (end-start)

        # Scatter back into per-sample fields
        for i, idx in enumerate(range(start, end)):
            base = i * 3
            text1[idx] = cands[base + 0]
            text2[idx] = cands[base + 1]
            text3[idx] = cands[base + 2]

    # Add/replace columns
    existing_cols = set(ds_split.column_names)
    if "text1" in existing_cols: ds_split = ds_split.remove_columns(["text1"]) 
    if "text2" in existing_cols: ds_split = ds_split.remove_columns(["text2"]) 
    if "text3" in existing_cols: ds_split = ds_split.remove_columns(["text3"]) 
    ds_split = ds_split.add_column("text1", text1)
    ds_split = ds_split.add_column("text2", text2)
    ds_split = ds_split.add_column("text3", text3)
    return ds_split


if __name__ == "__main__":
    # Larger batch size per GPU capability; one GPU (cuda:0) is used by CoNeTTEModel
    BATCH_SIZE_SAMPLES = int(os.getenv("CAPTION_BATCH_SIZE", "48"))
    updated = {}
    for split in splits:
        updated[split] = process_split(split, batch_size_samples=BATCH_SIZE_SAMPLES)

    # Save all splits together
    out_root = "./processed_fsd50k_captioned_conette"
    os.makedirs(out_root, exist_ok=True)
    from datasets import DatasetDict
    DatasetDict(updated).save_to_disk(out_root)
