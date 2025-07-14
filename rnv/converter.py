# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import torch
import torchaudio

from .rhythm.syllable.syllable_converter import SyllableRhythmConverter
from .rhythm.urhythmic.model import RhythmConverterFine, RhythmConverterGlobal
from .utils import load_target_style_feats
from .vc.knn import knn_vc
from .vocoder.models import HiFiGANWavLM


class Converter:
    def __init__(
        self,
        vocoder_checkpoint_path=None,
        source_rhythm_model_checkpoint_path=None,
        target_rhythm_model_checkpoint_path=None,
        rhythm_segmenter_checkpoint_path=None,
        rhythm_converter="syllable",
        rhythm_model_type="fine",
        device=None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.waveform_decoder = None
        if vocoder_checkpoint_path is not None:
            self.waveform_decoder = HiFiGANWavLM(checkpoint_path=vocoder_checkpoint_path, device=self.device)
        self.target_style_feats_path = None
        self.target_style_feats = None

        if rhythm_converter == "syllable":
            self.rhythm_model_class = SyllableRhythmConverter
        elif rhythm_model_type == "fine":
            self.rhythm_model_class = RhythmConverterFine
        else:
            self.rhythm_model_class = RhythmConverterGlobal

        if source_rhythm_model_checkpoint_path and target_rhythm_model_checkpoint_path and rhythm_segmenter_checkpoint_path:
            self.rhythm_converter = self.rhythm_model_class(source_rhythm_model_checkpoint_path, target_rhythm_model_checkpoint_path, rhythm_segmenter_checkpoint_path)

        self.rhythm_converter = rhythm_converter
        self.rhythm_model_type = rhythm_model_type
        self.source_rhythm_model_checkpoint_path = source_rhythm_model_checkpoint_path
        self.target_rhythm_model_checkpoint_path = target_rhythm_model_checkpoint_path
        self.rhythm_segmenter_checkpoint_path = rhythm_segmenter_checkpoint_path

    def convert_rhythm(self, source_feats, source_rhythm_model_checkpoint_path, target_rhythm_model_checkpoint_path, rhythm_segmenter_checkpoint_path, custom_target_rhythm=None, source_wav=None):
        source_feats = source_feats.to(self.device)
        if source_rhythm_model_checkpoint_path == target_rhythm_model_checkpoint_path:
            return source_feats
        with torch.inference_mode():
            source_rhythm_changed = source_rhythm_model_checkpoint_path != self.source_rhythm_model_checkpoint_path
            target_rhythm_changed = target_rhythm_model_checkpoint_path != self.target_rhythm_model_checkpoint_path
            segmenter_changed = rhythm_segmenter_checkpoint_path != self.rhythm_segmenter_checkpoint_path

            if source_rhythm_changed or target_rhythm_changed or segmenter_changed or custom_target_rhythm is not None:
                self.rhythm_converter = self.rhythm_model_class(source_rhythm_model_checkpoint_path, target_rhythm_model_checkpoint_path, rhythm_segmenter_checkpoint_path)
                self.target_rhythm_model_checkpoint_path = target_rhythm_model_checkpoint_path

            if custom_target_rhythm is not None:
                self.rhythm_converter.rhythm_model.set_custom_target(custom_target_rhythm)
                self.target_rhythm_model_checkpoint_path = None

            if self.rhythm_model_class == SyllableRhythmConverter:
                if self.rhythm_model_type == "global":
                    rhythm_feats = self.rhythm_converter.convert_global(source_feats)
                else:
                    if source_wav is None:
                        raise ValueError("source_wav must be provided for fine-grained syllable-based rhythm conversion.")
                    rhythm_feats = self.rhythm_converter.convert_fine_grained(source_wav, source_feats, add_silences=True)
            else:
                rhythm_feats = self.rhythm_converter(source_feats)
            return rhythm_feats

    def convert_voice(self, source_feats, target_style_feats_path, knnvc_topk, interpolation_rate, max_target_num_files=1000):
        if target_style_feats_path is None:
            return source_feats
        source_feats = source_feats.to(self.device)
        if abs(interpolation_rate) < 1e-9:
            return source_feats
        with torch.no_grad():
            if target_style_feats_path != self.target_style_feats_path:
                self.target_style_feats_path = target_style_feats_path
                self.target_style_feats = load_target_style_feats(target_style_feats_path, max_target_num_files)
            selected_feats = knn_vc(
                source_feats,
                self.target_style_feats,
                topk=knnvc_topk,
                device=self.device,
            )
            converted_feats = interpolation_rate * selected_feats + (1.0 - interpolation_rate) * source_feats
            return converted_feats

    def convert(
        self,
        source_feats,
        target_style_feats_path,
        source_rhythm_model_checkpoint_path,
        target_rhythm_model_checkpoint_path,
        rhythm_segmenter_checkpoint_path,
        knnvc_topk=4,
        interpolation_rate=1.0,
        save_path=None,
        max_target_num_files=1000,
        return_intermediate_wavs=False,
        return_feats=False,
        custom_target_rhythm=None,
        source_wav=None,
    ):
        with torch.no_grad():
            source_feats = source_feats.to(self.device)

            rhythm_feats = self.convert_rhythm(
                source_feats, source_rhythm_model_checkpoint_path, target_rhythm_model_checkpoint_path, rhythm_segmenter_checkpoint_path, custom_target_rhythm, source_wav
            )
            converted_feats = self.convert_voice(rhythm_feats, target_style_feats_path, knnvc_topk, interpolation_rate, max_target_num_files)

            if return_feats or self.waveform_decoder is None:
                return converted_feats, rhythm_feats

            converted_wav = self.waveform_decoder(converted_feats.unsqueeze(0)).unsqueeze(0)

            if save_path is not None:
                torchaudio.save(save_path, converted_wav, 16000)

            if return_intermediate_wavs:
                source_wav = self.waveform_decoder(source_feats.unsqueeze(0)).unsqueeze(0)
                rhythm_wav = self.waveform_decoder(rhythm_feats.unsqueeze(0)).unsqueeze(0)
                return converted_wav, source_wav, rhythm_wav

        return converted_wav
