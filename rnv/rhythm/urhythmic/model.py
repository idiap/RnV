# SPDX-FileCopyrightText: 2023 Benjamin van Niekerk
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import itertools
from typing import List

import torch
import torch.nn.functional as F

from .rhythm_model import RhythmModelFineGrained, RhythmModelGlobal
from .segmenter import Segmenter
from .utils import SILENCE, SoundType


class RhythmConverterFine:
    def __init__(
        self,
        source_rhythm_model_checkpoint_path,
        target_rhythm_model_checkpoint_path,
        segmenter_checkpoint_path,
        device=None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.segmenter = Segmenter(num_clusters=3, gamma=2, device=self.device)
        self.segmenter.load_state_dict(torch.load(segmenter_checkpoint_path, weights_only=False))

        self.rhythm_model = RhythmModelFineGrained()
        source_rhythm_checkpoint = torch.load(source_rhythm_model_checkpoint_path, weights_only=False)
        target_rhythm_checkpoint = torch.load(target_rhythm_model_checkpoint_path, weights_only=False)
        self.rhythm_model.load_checkpoints(source_rhythm_checkpoint, target_rhythm_checkpoint)

    @torch.inference_mode()
    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        clusters, boundaries = self.segmenter(feats)
        tgt_durations = self.rhythm_model(clusters, boundaries)
        converted_feats = time_stretch(feats.t().unsqueeze(0), clusters, boundaries, tgt_durations)
        converted_feats = converted_feats.squeeze().t()
        return converted_feats


def time_stretch(
    units: torch.Tensor,
    clusters: List[SoundType],
    boundaries: List[int],
    tgt_duartations: List[int],
) -> torch.Tensor:
    """
    Args:
        units (Tensor): units of shape (1, D, T)
            where D is the dimension of the units and T is the number of frames.
        clusters (List[SoundType]): list of sound types for each segment of shape (N,)
            where N is the number of segments.
        boundaries (List[int]): list of segment bounaries of shape (N+1,).
        tgt_durations (List[int]): list of target durations of shape (N,).
    Returns:
        Tensor: up/down sampled soft speech units.
    """
    units = [units[..., t0:tn] for cluster, (t0, tn) in zip(clusters, itertools.pairwise(boundaries)) if cluster not in SILENCE or tn - t0 > 3]
    units = [F.interpolate(segment, mode="linear", size=duration) for segment, duration in zip(units, tgt_duartations) if duration > 0]
    units = torch.cat(units, dim=-1)
    return units


class RhythmConverterGlobal:
    def __init__(
        self,
        source_rhythm_model_checkpoint_path,
        target_rhythm_model_checkpoint_path,
        segmenter_checkpoint_path,
        device=None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.segmenter = Segmenter(num_clusters=3, gamma=2, device=self.device)
        self.segmenter.load_state_dict(torch.load(segmenter_checkpoint_path, weights_only=False))

        self.rhythm_model = RhythmModelGlobal()
        source_rhythm_checkpoint = torch.load(source_rhythm_model_checkpoint_path, weights_only=False)
        target_rhythm_checkpoint = torch.load(target_rhythm_model_checkpoint_path, weights_only=False)
        self.rhythm_model.load_checkpoints(source_rhythm_checkpoint, target_rhythm_checkpoint)

    @torch.inference_mode()
    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        ratio = self.rhythm_model()
        converted_feats = F.interpolate(feats.t().unsqueeze(0), scale_factor=ratio, mode="linear")
        converted_feats = converted_feats.squeeze().t()
        return converted_feats
