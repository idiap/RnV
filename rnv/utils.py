# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import os
from pathlib import Path
from typing import Any

import requests
import torch


def find_wav_paths(root_dir: str | os.PathLike[Any]):
    wav_paths = []
    for file_path in Path(root_dir).rglob("*.*"):
        if file_path.suffix.lower() in (".flac", ".wav"):
            wav_paths.append(file_path)
    return wav_paths


def load_target_style_feats(feats_base_path, max_num_files=1000):
    feats = []
    for filepath in os.listdir(feats_base_path)[:max_num_files]:
        if ".pt" in filepath:
            filepath = os.path.join(feats_base_path, filepath)
            feats.append(torch.load(filepath, weights_only=False))
    feats = torch.concat(feats, dim=0).cpu()
    return feats


def get_vocoder_checkpoint_path(checkpoints_dir):
    os.makedirs(checkpoints_dir, exist_ok=True)  # Ensure directory exists

    checkpoint_path = os.path.join(checkpoints_dir, "prematch_g_02500000.pt")
    url = "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Downloading checkpoint to {checkpoint_path}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(checkpoint_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        else:
            raise Exception(f"Failed to download checkpoint: {response.status_code}")

    return checkpoint_path
