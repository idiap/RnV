# SPDX-FileCopyrightText: 2023 Benjamin van Niekerk
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse
import logging
import struct
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import webrtcvad
from tqdm import tqdm

from rnv.rhythm.urhythmic.segmenter import Segmenter

INT16_MAX = (2**15) - 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mark_silences(
    vad: webrtcvad.Vad,
    wav: torch.Tensor,
    hop_length: int = 320,
    sample_rate: int = 16000,
    pad: int = 40,
):
    """Marks silent frames using webrtcvad.

    Args:
        vad (webrtcvad.Vad): instance of the webrtcvad.Vad class.
        wav (Tensor): audio waveform of shape (1, T) where T is the number of samples.
        hop_length (int): the hop length measured in number of frames (defaults to 320).
        sample_rate (int): the sample rate (defaults to 16kHz).
        pad (int): padding (defaults to 40)

    Returns:
        NDArray: array of booleans indicating whether each frame is silent.
    """
    win_length = hop_length

    wav = F.pad(wav, (pad, pad))  # add padding to match HuBERT
    wav = wav[:, : wav.size(-1) - (wav.size(-1) % win_length)]

    pcm = struct.pack(
        f"{wav.size(-1)}h",
        *(np.round(wav.squeeze().numpy() * INT16_MAX)).astype(np.int16),
    )

    flags = []
    for window_start in range(0, wav.size(-1), hop_length):
        window_end = window_start + win_length
        flag = vad.is_speech(pcm[window_start * 2 : window_end * 2], sample_rate)
        flags.append(flag)
    return ~np.array(flags)


def mark_voiced(
    wav: torch.Tensor,
    hop_length: int = 320,
    win_length: int = 1024,
    sample_rate: int = 16000,
):
    _, voiced_flags, _ = librosa.pyin(
        wav.squeeze().numpy(),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C5"),
        sr=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
    )
    return voiced_flags


def train_segmenter(feats_dir, checkpoint_path, gamma):
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

    segmenter = Segmenter(num_kmeans_classes=100, num_clusters=3, gamma=gamma)

    segmenter.cluster(feats_dir)

    logger.info("Saving pre-identify checkpoint to %s", checkpoint_path)
    torch.save(segmenter.state_dict(), checkpoint_path)

    vad = webrtcvad.Vad(2)

    utterances = []
    feats_paths = list(feats_dir.rglob("*.pt"))
    for feats_path in tqdm(feats_paths):
        wav_path = feats_path.parent.parent / "wavs" / feats_path.with_suffix(".wav").name
        wav, _ = torchaudio.load(wav_path)
        feats = torch.load(feats_path, weights_only=True)

        segments, boundaries = segmenter.segment(feats)
        silences = mark_silences(vad, wav)
        voiced_flags = mark_voiced(wav)
        utterances.append((segments, boundaries, silences, voiced_flags))

    logger.info("Identifying the cluster id corresponding to each sound type")
    sound_types = segmenter.identify(utterances)

    logger.info("cluster 0 - %s", sound_types[0])
    logger.info("cluster 1 - %s", sound_types[1])
    logger.info("cluster 2 - %s", sound_types[2])

    logger.info("Saving checkpoint to %s", checkpoint_path)
    torch.save(segmenter.state_dict(), checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Cluster the codebook of discrete speech units
        and identify the cluster id corresponding to sonorants, obstruents, and silences.
        """
    )
    parser.add_argument(
        "feats_dir",
        metavar="feats-dir",
        help="path to the feats directory.",
        type=Path,
    )
    parser.add_argument(
        "checkpoint_path",
        metavar="checkpoint-path",
        help="path to save checkpoint.",
        type=Path,
    )
    parser.add_argument(
        "gamma",
        metavar="gamma",
        help="gamma val",
        type=int,
    )
    args = parser.parse_args()
    train_segmenter(args.feats_dir, args.checkpoint_path, args.gamma)
