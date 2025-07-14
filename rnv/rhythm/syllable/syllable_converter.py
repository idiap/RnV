# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import torch
import torch.nn.functional as F
from scipy.stats import gamma

from rnv.rhythm.syllable.syllable_segmenter import SyllableSegmenter


class SyllableRhythmConverter:
    def __init__(
        self,
        source_rhythm_model_checkpoint_path,
        target_rhythm_model_checkpoint_path,
        urhythmic_segmenter_checkpoint_path,
        device=None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.sr = 16000

        self.syllable_segmenter = SyllableSegmenter(urhythmic_segmenter_checkpoint_path)

        self.set_source_rhythm_params(source_rhythm_model_checkpoint_path)
        self.set_target_rhythm_params(target_rhythm_model_checkpoint_path)

    def set_source_rhythm_params(self, source_rhythm_model_checkpoint_path):
        source_rhythm_model = torch.load(source_rhythm_model_checkpoint_path, weights_only=False)
        self.source_speaking_rate = source_rhythm_model["speaking_rate"]
        self.source_syllable_gamma = gamma(source_rhythm_model["syllable_shape"], scale=source_rhythm_model["syllable_scale"])
        self.source_silence_gamma = gamma(source_rhythm_model["silence_shape"], scale=source_rhythm_model["silence_scale"])

    def set_target_rhythm_params(self, target_rhythm_model_checkpoint_path):
        target_rhythm_model = torch.load(target_rhythm_model_checkpoint_path, weights_only=False)
        self.target_speaking_rate = target_rhythm_model["speaking_rate"]
        self.target_syllable_gamma = gamma(target_rhythm_model["syllable_shape"], scale=target_rhythm_model["syllable_scale"])
        self.target_silence_gamma = gamma(target_rhythm_model["silence_shape"], scale=target_rhythm_model["silence_scale"])

    def convert_global(self, feats):
        ratio = self.source_speaking_rate / self.target_speaking_rate
        converted_feats = F.interpolate(feats.t().unsqueeze(0), scale_factor=ratio, mode="linear")
        converted_feats = converted_feats.squeeze().t()
        return converted_feats

    def convert_syllable_duration(self, duration):
        target_duration = self.target_syllable_gamma.ppf(self.source_syllable_gamma.cdf(duration))
        return target_duration if target_duration < float("inf") else duration

    def get_syllable_boundaries(self, peak_indices, speech_segments):
        syllable_boundaries = []

        if len(peak_indices) < 2 or len(speech_segments) == 1:
            return speech_segments

        for i in range(len(peak_indices) - 1):
            start_peak = peak_indices[i]
            end_peak = peak_indices[i + 1]
            syllable_boundaries.append((start_peak, end_peak))

        # Handle the first segment
        first_peak = peak_indices[0]
        for segment in speech_segments:
            if segment[0] <= first_peak <= segment[1]:
                syllable_boundaries.insert(0, (segment[0], first_peak))
                break

        # Handle the last segment
        last_peak = peak_indices[-1]
        for segment in speech_segments:
            if segment[0] <= last_peak <= segment[1]:
                syllable_boundaries.append((last_peak, segment[1]))
                break

        return syllable_boundaries

    def convert_fine_grained(self, wav, feats, add_silences=False):
        filtered_peak_indices, speech_segments, silence_segments = self.syllable_segmenter.get_segments_and_filtered_peaks(wav, feats)

        syllable_boundaries = self.get_syllable_boundaries(filtered_peak_indices, speech_segments)

        syllable_durations = [syllable_boundaries[i][1] - syllable_boundaries[i][0] for i in range(len(syllable_boundaries))]

        syllable_durations_in_s = [duration / self.sr for duration in syllable_durations]

        target_syllable_durations_in_s = [self.convert_syllable_duration(duration) for duration in syllable_durations_in_s]

        target_syllable_durations = [int(duration * self.sr) for duration in target_syllable_durations_in_s]

        # Convert syllable_boundaries indices to indices in the feats
        feats_len = feats.size(0)
        wav_len = len(wav)
        ratio = feats_len / wav_len
        syllable_boundaries_feats = [(int(start * ratio), int(end * ratio)) for start, end in syllable_boundaries]

        # Collect in a list each segment from the feats
        feats_segments = [feats[start:end, :] for start, end in syllable_boundaries_feats]

        # Interpolate each segment to the target duration
        target_syllable_durations_feats = [int(duration * ratio) for duration in target_syllable_durations]

        interpolated_segments = [
            F.interpolate(segment.t().unsqueeze(0), size=target_syllable_durations_feats[i], mode="linear").squeeze(0).t()
            for i, segment in enumerate(feats_segments)
            if target_syllable_durations_feats[i] > 0 and segment.size(0) > 0  # Only include segments with non-zero len and non-zero target size
        ]

        if add_silences:
            # Add the start and end silence segments to the feats
            start_silence = feats[int(silence_segments[0][0] * ratio) : int(silence_segments[0][1] * ratio)]
            end_silence = feats[int(silence_segments[-1][0] * ratio) : int(silence_segments[-1][1] * ratio)]

            converted_feats = torch.cat([start_silence] + interpolated_segments + [end_silence], dim=0)
        else:
            converted_feats = torch.cat(interpolated_segments, dim=0)
        return converted_feats
