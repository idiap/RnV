# SPDX-FileCopyrightText: 2023 Benjamin van Niekerk
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import itertools
from typing import Dict, List, Mapping, Tuple

import numpy as np
from scipy import stats

from .utils import HOP_LENGTH, SAMPLE_RATE, SILENCE, SONORANT, SoundType


def transform(source: stats.rv_continuous, target: stats.rv_continuous, sample: float) -> float:
    transformed = target.ppf(source.cdf(sample))
    return transformed if transformed < float("inf") else sample


class RhythmModelFineGrained:
    """Rhythm modeling block, estimates the duration distribution of each sound type."""

    def __init__(self, hop_length: int = HOP_LENGTH, sample_rate: int = SAMPLE_RATE):
        """
        Args:
            hop_length (int): hop length between the frames of speech units.
            sample_rate (int): the sample rate of the audio waveforms.
        """
        self.hop_rate = hop_length / sample_rate
        self.source = None
        self.target = None

    def _tally_durations(self, utterances: List[Tuple[List[SoundType], List[int]]]) -> Dict[SoundType, np.ndarray]:
        durations_dict = {}
        for clusters, boundaries in utterances:
            durations = np.diff(boundaries)
            for cluster, duration in zip(clusters, durations):
                if cluster in SILENCE and duration <= 3:  # ignore silences that are too short
                    continue
                durations_dict.setdefault(cluster, []).append(self.hop_rate * duration)
        return {cluster: np.array(durations) for cluster, durations in durations_dict.items()}

    def state_dict(self) -> Mapping[str, Mapping[SoundType, Tuple[float, ...]]]:
        state_dict = {}
        if self.source:
            state_dict["source"] = {cluster: (dist.args[0], dist.kwds["scale"]) for cluster, dist in self.source.items()}
        if self.target:
            state_dict["target"] = {cluster: (dist.args[0], dist.kwds["scale"]) for cluster, dist in self.target.items()}
        return state_dict

    def load_checkpoints(
        self,
        source_rhythm_checkpoint: Mapping[SoundType, Tuple[float, ...]],
        target_rhythm_checkpoint: Mapping[SoundType, Tuple[float, ...]] = None,
    ):
        self.source = {cluster: stats.gamma(a, scale=scale) for cluster, (a, _, scale) in source_rhythm_checkpoint.items()}

        if target_rhythm_checkpoint:
            self.target = {cluster: stats.gamma(a, scale=scale) for cluster, (a, _, scale) in target_rhythm_checkpoint.items()}

    def set_custom_target(self, custom_target: Mapping[SoundType, Tuple[float, ...]]):
        self.target = {cluster: stats.gamma(a, scale=scale) for cluster, (a, scale) in custom_target.items()}

    def fit(self, utterances: List[Tuple[List[SoundType], List[int]]]) -> Mapping[SoundType, Tuple[float, ...]]:
        duration_tally = self._tally_durations(utterances)
        dists = {cluster: stats.gamma.fit(durations, floc=0) for cluster, durations in duration_tally.items()}
        return dists

    def __call__(self, clusters: List[SoundType], boundaries: List[int]) -> List[int]:
        """Transforms the source durations to match the target rhythm.

        Args:
            clusters (List[SoundType]): list of segmented sound types of shape (N,).
            boundaries (List[int]): list of segment boundaries of shape (N+1,).

        Returns:
            List[int]: list of target durations of shape (N,)
        """
        durations = self.hop_rate * np.diff(boundaries)
        durations = [
            transform(self.source[cluster], self.target[cluster], duration)
            for cluster, duration in zip(clusters, durations)
            if cluster not in SILENCE or duration > 3 * self.hop_rate  # ignore silences that are too short
        ]
        durations = [round(duration / self.hop_rate) for duration in durations]
        return durations


def segment_rate(
    codes: List[SoundType],
    boundaries: List[int],
    sonorant: SoundType = SONORANT,
    silence: SoundType = SILENCE,
    unit_rate=0.02,
) -> float:
    times = np.round(np.array(boundaries) * unit_rate, 2)
    segments = [(code, t0, tn) for code, (t0, tn) in zip(codes, itertools.pairwise(times)) if code not in silence]

    total_duration = sum([tn - t0 for _, t0, tn in segments])
    if total_duration == 0:
        return 0
    return len([code for code, _, _ in segments if code in sonorant]) / total_duration


class RhythmModelGlobal:
    """Rhythm modeling block (Global). Estimates speaking rate."""

    def __init__(self, hop_length: int = 320, sample_rate: int = 16000):
        """
        Args:
            hop_length (int): hop length between the frames of speech units.
            sample_rate (int): the sample rate of the audio waveforms.
        """
        self.hop_rate = hop_length / sample_rate
        self.source_rate = None
        self.target_rate = None

    def state_dict(self):
        state_dict = {}
        if self.source_rate:
            state_dict["source_rate"] = self.source_rate
        if self.target_rate:
            state_dict["target_rate"] = self.target_rate
        return state_dict

    def load_state_dict(self, state_dict):
        if "source_rate" in state_dict:
            self.source_rate = state_dict["source_rate"]
        if "target_rate" in state_dict:
            self.target_rate = state_dict["target_rate"]

    def load_checkpoints(
        self,
        source_rhythm_rate,
        target_rhythm_rate,
    ):
        self.source_rate = source_rhythm_rate
        self.target_rate = target_rhythm_rate

    def fit(self, utterances: List[Tuple[List[SoundType], List[int]]]) -> float:
        rates = [segment_rate(clusters, boundaries, SONORANT, SILENCE, self.hop_rate) for clusters, boundaries in utterances]
        rates = [rate for rate in rates if rate > 0]
        return np.mean(rates)

    def fit_source(self, utterances: List[Tuple[List[SoundType], List[int]]]):
        """Estimate the speaking rate of the source speaker.

        Args:
            utterances (List[Tuple[List[SoundType], List[int]]]): list of segemented utterances.
        """
        self.source_rate = self._fit(utterances)

    def fit_target(self, utterances: List[Tuple[List[SoundType], List[int]]]):
        """Estimate the speaking rate of the target speaker.

        Args:
            utterances (List[Tuple[List[SoundType], List[int]]]): list of segemented utterances.
        """
        self.target_rate = self._fit(utterances)

    def __call__(self) -> float:
        """
        Returns:
            float: ratio between the source and target speaking rates.
        """
        return self.source_rate / self.target_rate
