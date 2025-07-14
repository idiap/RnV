# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import sys
from pathlib import Path

import torch

from rnv.ssl.models import WavLM
from rnv.utils import find_wav_paths


def load_model(model_name):
    if model_name == "wavlm":
        model = WavLM()
    else:
        raise NameError("Invalid model name")
    return model


def rename_embeddings_dir_in_path(output_path: Path, model_name: str) -> Path:
    parts = list(output_path.parts)
    parts[-2] = model_name if parts[-2] == "wavs" else f"{parts[-2]}-{model_name}"
    return Path(*parts)


def save_embeddings(embeddings, input_path: Path, output_root: Path, dataset_root: Path, model_name: str):
    relative_path = input_path.relative_to(dataset_root)
    output_path = output_root / relative_path.with_suffix(".pt")
    output_path = rename_embeddings_dir_in_path(output_path, model_name)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(embeddings, output_path)
    print(f"Saved {output_path}")


def main(model_name: str, dataset_path: Path, output_path: Path):
    model = load_model(model_name)
    wav_paths = find_wav_paths(dataset_path)
    for wav_path in wav_paths:
        with torch.no_grad():
            embeddings = model.extract_framewise_features(wav_path).cpu()
        save_embeddings(embeddings, wav_path, output_path, dataset_path, model_name)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_dataset_embeddings.py [model_name] [dataset_path] [output_path]")
        sys.exit(1)
    MODEL_NAME = sys.argv[1]
    DATASET_PATH = Path(sys.argv[2])
    OUTPUT_PATH = Path(sys.argv[3])
    main(MODEL_NAME, DATASET_PATH, OUTPUT_PATH)
