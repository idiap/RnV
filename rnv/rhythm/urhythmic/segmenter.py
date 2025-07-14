# SPDX-FileCopyrightText: 2023 Benjamin van Niekerk
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import itertools
import os
from collections import Counter
from pathlib import Path
from typing import Any, List, Mapping, Tuple

import numba
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from tqdm import tqdm

from .utils import OBSTRUENT, SILENCE, SONORANT, SoundType


class Segmenter:
    def __init__(
        self,
        num_kmeans_classes: int = 100,
        num_clusters: int = 3,
        gamma: float = 2,
        device=None,
    ):
        """
        Args:
            num_kmeans_classes (int): number of clusters used in the initial KMeans step
            num_clusters (int): number of clusters used for agglomerative clustering.
            gamma (float): regularizer weight encouraging longer segments
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.kmeans = MiniBatchKMeans(n_clusters=num_kmeans_classes, random_state=42, verbose=1)
        self.clustering = AgglomerativeClustering(n_clusters=num_clusters)
        self.sound_types = {}
        self.codebook = None

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "n_clusters_": self.clustering.n_clusters_,
            "labels_": torch.from_numpy(self.clustering.labels_),
            "n_leaves_": self.clustering.n_leaves_,
            "n_features_in_": self.clustering.n_features_in_,
            "children_": torch.from_numpy(self.clustering.children_),
            "sound_types": self.sound_types,
            "gamma": self.gamma,
            "codebook": self.codebook.cpu().numpy(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        if self.clustering.n_clusters != state_dict["n_clusters_"]:
            raise RuntimeError(f"Error in loading state_dict for {self.__class__.__name__}")
        self.clustering.labels_ = state_dict["labels_"].numpy()
        self.clustering.n_leaves_ = state_dict["n_leaves_"]
        self.clustering.n_features_in_ = state_dict["n_features_in_"]
        self.clustering.children_ = state_dict["children_"].numpy()
        self.sound_types = state_dict["sound_types"]
        self.gamma = state_dict["gamma"]
        self.codebook = state_dict["codebook"]
        self.codebook = torch.from_numpy(self.codebook).to(self.device)

    def cluster(self, feats_dir: Path):
        filepaths = list(feats_dir.rglob("*.pt"))
        batch_size = 1024
        for i in tqdm(range(0, len(filepaths), batch_size), desc="Clustering batches"):
            feats = []
            batch_filepaths = filepaths[i : i + batch_size]
            for filepath in batch_filepaths:
                if ".pt" in str(filepath):
                    filepath = os.path.join(feats_dir, filepath)
                    feats.append(torch.load(filepath, weights_only=False))
            feats = torch.concat(feats, dim=0).cpu().numpy()
            self.kmeans.partial_fit(feats)

        self.codebook = self.kmeans.cluster_centers_
        self.clustering.fit(self.codebook)
        self.codebook = torch.from_numpy(self.codebook).to(self.device)

    def segment(self, feats: torch.Tensor) -> Tuple[List[int], List[int]]:
        log_probs = calculate_log_probs(feats.to(self.device), self.codebook)
        log_probs = log_probs.cpu().numpy()
        codes, boundaries = get_segments(log_probs, self.gamma)
        segments = codes[boundaries[:-1]]
        segments, boundaries = cluster_merge(self.clustering, segments, boundaries)
        return list(segments), list(boundaries)

    def identify(
        self,
        utterances: List[Tuple[np.ndarray, ...]],
    ) -> Mapping[int, SoundType]:
        """Identify which clusters correspond to sonorants, obstruents, and silences.
        Only implemented for num_clusters = 3.

        Args:
            utterances: list of segmented utterances along with marked silences and voiced frames.

        Returns:
            Mapping[int, SoundType]: mapping of cluster id to sonorant, obstruent, or silence.
        """
        if self.clustering.n_clusters_ != 3:
            raise ValueError("Cluster identification is only implemented for num_clusters = 3.")

        silence_overlap = Counter()
        voiced_overlap = Counter()
        total = Counter()

        for segments, boundaries, silences, voiced_flags in utterances:
            for code, (a, b) in zip(segments, itertools.pairwise(boundaries)):
                silence_overlap[code] += np.count_nonzero(silences[a : b + 1])
                voiced_overlap[code] += np.count_nonzero(voiced_flags[a : b + 1])
                total[code] += b - a + 1

        clusters = {0, 1, 2}

        silence, _ = max(((k, v / total[k]) for (k, v) in silence_overlap.items()), key=lambda x: x[1])
        clusters.remove(silence)

        sonorant, _ = max(((k, v / total[k]) for (k, v) in voiced_overlap.items() if k in clusters), key=lambda x: x[1])
        clusters.remove(sonorant)

        obstruent = clusters.pop()

        self.sound_types = {
            silence: SILENCE,
            sonorant: SONORANT,
            obstruent: OBSTRUENT,
        }
        return self.sound_types

    def __call__(self, feats: torch.Tensor) -> Tuple[List[SoundType], List[int]]:
        segments, boundaries = self.segment(feats)
        segments = [self.sound_types[cluster] for cluster in segments]
        return segments, boundaries


def get_segments(log_probs: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    alpha, P = _segment_dp(log_probs, gamma)
    return _backtrack(alpha, P)


@numba.njit()
def _backtrack(alpha, P):
    rhs = len(alpha) - 1
    segments = np.zeros(len(alpha) - 1, dtype=np.int32)
    boundaries = [rhs]
    while rhs != 0:
        lhs, code = P[rhs, :]
        boundaries.append(lhs)
        segments[lhs:rhs] = code
        rhs = lhs
    boundaries.reverse()
    return segments, np.array(boundaries)


@numba.njit()
def _segment_dp(log_probs, gamma):
    T, K = log_probs.shape

    alpha = np.zeros(T + 1, dtype=np.float32)
    P = np.zeros((T + 1, 2), dtype=np.int32)
    D = np.zeros((T, T, K), dtype=np.float32)

    for t in range(T):
        for k in range(K):
            D[t, t, k] = log_probs[t, k]
    for t in range(T):
        for s in range(t + 1, T):
            D[t, s, :] = D[t, s - 1, :] + log_probs[s, :]

    for t in range(T):
        alpha[t + 1] = -np.inf
        for s in range(t + 1):
            k = np.argmax(D[t - s, t, :])
            alpha_max = alpha[t - s] + D[t - s, t, k] + gamma * s
            if alpha_max > alpha[t + 1]:
                P[t + 1, :] = t - s, k
                alpha[t + 1] = alpha_max
    return alpha, P


def cluster_merge(clustering: AgglomerativeClustering, segments: np.ndarray, boundaries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    clusters = clustering.labels_[segments]
    cluster_switches = np.diff(clusters, prepend=-1, append=-1)
    (cluster_boundaries,) = np.nonzero(cluster_switches)
    clusters = clusters[cluster_boundaries[:-1]]
    cluster_boundaries = boundaries[cluster_boundaries]
    return clusters, cluster_boundaries


def calculate_log_probs(feats: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    # Normalize the feature vectors and centroids
    feats_normalized = F.normalize(feats, p=2, dim=-1)
    centroids_normalized = F.normalize(centroids, p=2, dim=-1)
    # Compute cosine similarity
    logits = torch.matmul(feats_normalized, centroids_normalized.t())
    # Apply temperature scaling
    logits = logits / 0.1
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs
