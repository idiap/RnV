# SPDX-FileCopyrightText: 2024 Idiap Research Institute
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

import argparse
from pathlib import Path

import pandas as pd
from data_preprocessing.preprocess_audio_data import preprocess_wav

bad_utts = []
with open("scripts/bad_utts_torgo") as f:
    for line in f:
        bad_utts.append(line.strip().split()[0])


def is_bad_transcript(transcript):
    bad_substr = ["[", "]", ".jpg", "/"]
    for substr in bad_substr:
        if substr in transcript:
            return True
    return False


def preprocess_torgo(sample_rate, source_dir, target_dir):
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Iterate over speaker directories
    for speaker_dir in source_path.iterdir():
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        print(f"Processing speaker: {speaker_id}")
        speaker_wavs_target_dir = target_path / speaker_id / "wavs"
        speaker_wavs_target_dir.mkdir(parents=True, exist_ok=True)

        metadata_entries = []

        for session_dir in speaker_dir.glob("Session*"):
            if not session_dir.is_dir():
                continue

            session_num = session_dir.name.replace("Session", "")
            print(f"  Processing session: {session_num}")

            prompt_dir = session_dir / "prompts"
            if not prompt_dir.exists():
                print(f"    Warning: prompts directory not found in {session_dir}")
                continue

            for wav_file in session_dir.rglob("*.wav"):
                mic = wav_file.parent.stem.split("_")[1]
                utt_id = f"{speaker_id}-{session_dir.stem}-{mic}-{wav_file.stem}"
                if utt_id in bad_utts:
                    continue

                # Get corresponding prompt file
                prompt_file = prompt_dir / f"{wav_file.stem}.txt"
                if not prompt_file.exists():
                    print(f"    Warning: No prompt file found for {wav_file.name}")
                    continue

                # Read transcript
                with open(prompt_file, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()

                # Check for prompts that don't correspond to transcriptions
                if is_bad_transcript(transcript):
                    print(f"    Warning: Transcript for {wav_file.name} contains unwanted characters")
                    continue

                processed_audio = preprocess_wav(wav_file, sample_rate, vad_sr=None, vad_mode=None, vad_frame_duration=None, do_trim_silences=False)

                # Filter out audio files that are too long; they will have an inaccurate transcript
                if len(processed_audio) > 60 * 1000:  # Returned length is in milliseconds
                    print(f"    Warning: {wav_file.name} is too long")
                    continue

                # Create new filename
                new_filename = f"{utt_id}.wav"
                new_filepath = speaker_wavs_target_dir / new_filename

                # Save processed audio
                processed_audio.export(new_filepath, format="wav")

                # Add to metadata
                metadata_entries.append({"wav_name": f"{new_filename}", "transcript": transcript})

        if metadata_entries:
            metadata_df = pd.DataFrame(metadata_entries)
            metadata_path = target_path / speaker_id / "metadata.csv"
            metadata_df.to_csv(metadata_path, sep="|", index=False)
            print(f"  Saved metadata for {speaker_id}: {len(metadata_entries)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Torgo dataset.")
    parser.add_argument("--base_dir", required=True, help="Path to the base directory of the Torgo dataset.")
    parser.add_argument("--target_dir", required=True, help="Path to the target directory for processed data.")
    args = parser.parse_args()

    SAMPLE_RATE = 16000

    preprocess_torgo(SAMPLE_RATE, args.base_dir, args.target_dir)
