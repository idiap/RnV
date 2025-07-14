# SPDX-FileCopyrightText: 2023 Benjamin van Niekerk
# SPDX-FileContributor: Karl El Hajal
#
# SPDX-License-Identifier: GPL-3.0-only

from enum import Flag, auto


class SoundType(Flag):
    VOWEL = auto()
    APPROXIMANT = auto()
    NASAL = auto()
    FRICATIVE = auto()
    STOP = auto()
    SILENCE = auto()


SONORANT = SoundType.VOWEL | SoundType.APPROXIMANT | SoundType.NASAL
OBSTRUENT = SoundType.FRICATIVE | SoundType.STOP
SILENCE = SoundType.SILENCE

HOP_LENGTH = 320
SAMPLE_RATE = 16000
