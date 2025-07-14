# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import torch

from rnv.rhythm.syllable.segment_syllables import getOnsets
from rnv.rhythm.urhythmic.segmenter import Segmenter
from rnv.rhythm.urhythmic.utils import SILENCE


class SyllableSegmenter:
    def __init__(self, urhythmic_segmenter_checkpoint_path):
        self.sr = 16000
        self.urhythmic_segmenter = Segmenter(num_clusters=3, gamma=3)
        self.urhythmic_segmenter.load_state_dict(torch.load(urhythmic_segmenter_checkpoint_path, weights_only=False))

    def get_segments_and_filtered_peaks(self, wav, feats):
        valley_indices, peak_indices, _ = getOnsets(wav, self.sr)

        segments, boundaries = self.urhythmic_segmenter(feats)
        # Fix sampling frequencies
        peak_indices = [index * 16 for index in peak_indices]
        valley_indices = [index * 16 for index in valley_indices]
        boundaries = [boundary * 320 for boundary in boundaries]  # 320 = 16kHz * 20ms
        boundaries = [max(boundary, 0) for boundary in boundaries]

        silence_segments = []
        speech_segments = []
        for i, segment in enumerate(segments):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(wav)

            if segment == SILENCE:
                silence_segments.append((start, end))
            else:
                speech_segments.append((start, end))

        filtered_peak_indices = []
        for peak_index in peak_indices:
            for segment in speech_segments:
                if segment[0] <= peak_index <= segment[1]:
                    filtered_peak_indices.append(peak_index)
                    break

        return filtered_peak_indices, speech_segments, silence_segments

    def get_audio_peak_to_peak_and_silence_durations(self, wav, feats):
        if len(wav) > self.sr * 60:
            return [], []

        filtered_peak_indices, speech_segments, silence_segments = self.get_segments_and_filtered_peaks(wav, feats)

        silence_durations = [segment[1] - segment[0] for segment in silence_segments]
        if len(silence_durations) > 0:
            silence_durations.pop(0)  # Remove the first silence segment
        if len(silence_durations) > 0:
            silence_durations.pop(-1)  # Remove the last silence segment
        silence_durations_in_s = [duration / self.sr for duration in silence_durations]

        peak_to_peak_durations = [filtered_peak_indices[i + 1] - filtered_peak_indices[i] for i in range(len(filtered_peak_indices) - 1)]

        # When there is only one peak, the peak to peak duration is the duration of the speech segment
        if len(filtered_peak_indices) < 2 and len(speech_segments) == 1:
            peak_to_peak_durations = [speech_segments[0][1] - speech_segments[0][0]]

        peak_to_peak_durations_in_s = [duration / self.sr for duration in peak_to_peak_durations]

        return peak_to_peak_durations_in_s, silence_durations_in_s
