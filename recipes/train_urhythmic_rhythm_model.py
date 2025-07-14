# SPDX-FileCopyrightText: 2023 Benjamin van Niekerk
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from rnv.rhythm.urhythmic.rhythm_model import RhythmModelFineGrained, RhythmModelGlobal
from rnv.rhythm.urhythmic.segmenter import Segmenter
from rnv.rhythm.urhythmic.utils import HOP_LENGTH, SAMPLE_RATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_rhythm_model(
    speaker_id: str,
    model_type_str: str,
    feats_dir: Path,
    segmenter_checkpoint_path: Path,
    output_dir: Path,
):
    logger.info("Loading segmenter from %s", segmenter_checkpoint_path)
    segmenter = Segmenter(num_clusters=3, gamma=2)
    segmenter.load_state_dict(torch.load(segmenter_checkpoint_path, weights_only=False))

    logger.info("Segmenting features from %s and preparing utterances", feats_dir)
    utterances = []
    for feat_path in tqdm(list(feats_dir.rglob("*.pt"))):
        feats = torch.load(feat_path, map_location="cpu")
        segments, boundaries = segmenter(feats)
        utterances.append((list(segments), list(boundaries)))

    SelectedModel = RhythmModelFineGrained if model_type_str == "fine" else RhythmModelGlobal
    rhythm_model = SelectedModel(hop_length=HOP_LENGTH, sample_rate=SAMPLE_RATE)
    logger.info("Fitting %s rhythm model", model_type_str)
    dists = rhythm_model.fit(utterances)

    rhythm_checkpoint_path = f"{output_dir}/{speaker_id}_{model_type_str}_urhythmic_model.pth"
    logger.info("Saving rhythm model checkpoint to %s", rhythm_checkpoint_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    torch.save(dists, rhythm_checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment features and train Urhythmic rhythm model.")
    parser.add_argument(
        "speaker_id",
        metavar="speaker-id",
        help="Speaker ID.",
        type=Path,
    )
    parser.add_argument(
        "model_type",
        metavar="model-type",
        help="rhythm model type (global, fine).",
        type=str,
        choices=["global", "fine"],
    )
    parser.add_argument(
        "feats_dir",
        metavar="feats-dir",
        help="path to the directory of feature files (*.pt).",
        type=Path,
    )
    parser.add_argument(
        "segmenter_checkpoint_path",
        metavar="segmenter-checkpoint-path",
        help="path to the segmenter model checkpoint.",
        type=Path,
    )
    parser.add_argument("out_dir", metavar="out-dir", type=Path, help="path to the output directory.")
    args = parser.parse_args()
    train_rhythm_model(
        args.speaker_id,
        args.model_type,
        args.feats_dir,
        args.segmenter_checkpoint_path,
        args.out_dir,
    )
