#!/usr/bin/env python3
"""
Typing Music Generator - Relative Pitch Navigation
Maps keyboard to melodic movements through pentatonic scale.
"""

import numpy as np
import pyaudio
import queue
import threading
import time
import configparser
import os
from typing import Optional
import string
from music21 import chord as m21chord
from music21 import pitch as m21pitch
from pynput import keyboard
from PySide6.QtCore import Qt, QCoreApplication, QObject, Signal, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Letter frequencies (descending order)
LETTER_FREQ = "etaoinshrdlcumfpgwybvkxjqz"

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_OPTIONS = [
    ("C", 0),
    ("C#/Db", 1),
    ("D", 2),
    ("D#/Eb", 3),
    ("E", 4),
    ("F", 5),
    ("F#/Gb", 6),
    ("G", 7),
    ("G#/Ab", 8),
    ("A", 9),
    ("A#/Bb", 10),
    ("B", 11),
]

MIN_SCALE_INDEX = 0

# Audio queue - thread-safe communication
audio_queue = queue.Queue()
running = True
STATE_LOCK = threading.RLock()
CONFIG_PATH = "config.ini"

MODE_STEP_PATTERNS = [
    ("Major", [0, 2, 2, 1, 2, 2, 2, 1]),
    ("Natural minor", [0, 2, 1, 2, 2, 1, 2, 2]),
    ("Harmonic minor", [0, 2, 1, 2, 2, 1, 3, 1]),
    ("Melodic minor", [0, 2, 1, 2, 2, 2, 2, 2]),
    ("Dorian mode", [0, 2, 1, 2, 2, 2, 1, 2]),
    ("Phrygian mode", [0, 1, 2, 2, 2, 1, 2, 2]),
    ("Lydian mode", [0, 2, 2, 2, 1, 2, 2, 1]),
    ("Mixolydian mode", [0, 2, 2, 1, 2, 2, 1, 2]),
    ("Locrian mode", [0, 1, 2, 2, 1, 2, 2, 2]),
    ("Ahava raba mode", [0, 1, 3, 1, 2, 1, 2, 2]),
    ("Minor pentatonic", [0, 3, 2, 2, 3, 2]),
    ("Pentatonic", [0, 2, 2, 3, 2, 3]),
    ("Blues", [0, 3, 2, 1, 1, 3]),
]


def step_pattern_to_intervals(step_pattern: list[int]) -> list[int]:
    offsets = [0]
    total = 0
    for step in step_pattern[1:]:
        total += step
        offsets.append(total)
    if offsets and offsets[-1] == 12:
        return offsets[:-1]
    return offsets


MODE_DEFS: dict[str, dict[str, object]] = {
    name: {
        "intervals": step_pattern_to_intervals(steps),
    }
    for name, steps in MODE_STEP_PATTERNS
}

for _name, _defn in MODE_DEFS.items():
    _defn["degrees"] = len(_defn["intervals"])


def midi_to_freq(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def pitch_class_to_name(pc: int) -> str:
    return NOTE_NAMES_SHARP[pc % 12]


class MusicState:
    """Maintains current musical state"""

    def __init__(self):
        # 4 voices, each starting on the root note in a different octave
        # 3 octaves total range: indices 0..21 (roots at 0, 7, 14, 21)
        self.voice1_index = 0
        self.voice2_index = 7
        self.voice3_index = 14
        self.voice4_index = 21

        self.octaves = 3
        self.min_scale_index = 0
        self.mode_name = "Dorian mode"
        self.scale_degrees = int(MODE_DEFS[self.mode_name]["degrees"])
        self.mode_intervals = list(MODE_DEFS[self.mode_name]["intervals"])
        self.max_scale_index = self.octaves * self.scale_degrees

        self.key_pitch_class = 9
        self.base_octave = 2
        self.base_midi = (self.base_octave + 1) * 12 + self.key_pitch_class
        self.base_freq = midi_to_freq(self.base_midi)

        self.voice2_index = min(self.scale_degrees, self.max_scale_index)
        self.voice3_index = min(self.scale_degrees * 2, self.max_scale_index)
        self.voice4_index = min(self.scale_degrees * 3, self.max_scale_index)

    def get_frequency(self, scale_index):
        """Convert scale index to frequency using current mode"""
        octave = scale_index // self.scale_degrees
        position = scale_index % self.scale_degrees
        semitones = octave * 12 + self.mode_intervals[position]
        return self.base_freq * (2 ** (semitones / 12))

    def get_note_name(self, scale_index):
        """Convert scale index to note name (with octave)"""
        octave = scale_index // self.scale_degrees
        position = scale_index % self.scale_degrees
        semitones = octave * 12 + self.mode_intervals[position]

        midi = self.base_midi + semitones
        note = NOTE_NAMES_SHARP[midi % 12]
        note_octave = (midi // 12) - 1
        return f"{note}{note_octave}"

    def get_midi(self, scale_index: int) -> int:
        octave = scale_index // self.scale_degrees
        position = scale_index % self.scale_degrees
        semitones = octave * 12 + self.mode_intervals[position]
        return self.base_midi + semitones

    def set_key_pitch_class(self, pitch_class: int):
        self.key_pitch_class = pitch_class
        self.base_midi = (self.base_octave + 1) * 12 + self.key_pitch_class
        self.base_freq = midi_to_freq(self.base_midi)

    def set_mode(self, mode_name: str):
        self.mode_name = mode_name
        self.scale_degrees = int(MODE_DEFS[mode_name]["degrees"])
        self.mode_intervals = list(MODE_DEFS[mode_name]["intervals"])
        self.max_scale_index = self.octaves * self.scale_degrees
        self.voice1_index = min(self.voice1_index, self.max_scale_index)
        self.voice2_index = min(self.voice2_index, self.max_scale_index)
        self.voice3_index = min(self.voice3_index, self.max_scale_index)
        self.voice4_index = min(self.voice4_index, self.max_scale_index)

    def set_octaves(self, octaves: int):
        self.octaves = octaves
        self.max_scale_index = self.octaves * self.scale_degrees
        self.voice1_index = min(self.voice1_index, self.max_scale_index)
        self.voice2_index = min(self.voice2_index, self.max_scale_index)
        self.voice3_index = min(self.voice3_index, self.max_scale_index)
        self.voice4_index = min(self.voice4_index, self.max_scale_index)

    def set_voice_indices(self, indices: list[int]):
        self.voice1_index, self.voice2_index, self.voice3_index, self.voice4_index = indices

    def _move_voice(self, attr_name: str, steps: int, char: str, movement_map: dict[str, int]) -> float:
        current = getattr(self, attr_name)
        new_index = current + steps

        if new_index > self.max_scale_index:
            movement_map[char] = -movement_map[char]
            overflow = new_index - self.max_scale_index
            reflected = self.max_scale_index - overflow
            if reflected == current:
                reflected -= 1
            setattr(self, attr_name, reflected)
        elif new_index < self.min_scale_index:
            movement_map[char] = -movement_map[char]
            reflected = self.min_scale_index + (self.min_scale_index - new_index)
            if reflected == current:
                reflected += 1
            setattr(self, attr_name, reflected)
        else:
            setattr(self, attr_name, new_index)

        return self.get_frequency(getattr(self, attr_name))

    def move_voice1(self, steps, char):
        """Move voice 1"""
        return self._move_voice("voice1_index", steps, char, voice1_movements)

    def move_voice2(self, steps, char):
        """Move voice 2"""
        return self._move_voice("voice2_index", steps, char, voice2_movements)

    def move_voice3(self, steps, char):
        """Move voice 3"""
        return self._move_voice("voice3_index", steps, char, voice3_movements)

    def move_voice4(self, steps, char):
        """Move voice 4"""
        return self._move_voice("voice4_index", steps, char, voice4_movements)


# Global music state
music_state = MusicState()

# Letter movement mappings - SEPARATE for each voice
voice1_movements = {}
voice2_movements = {}
voice3_movements = {}
voice4_movements = {}

TIMBRE_OPTIONS = [
    "Sine",
    "Triangle",
    "Square",
    "Saw",
    "Pulse 25%",
    "Pulse 12.5%",
    "Organ",
    "Soft Organ",
    "Strings",
    "Brass",
    "Bell",
    "FM EP",
]

TIMBRES = ["Organ", "Organ", "Organ", "Organ"]  # per voice (must be values from TIMBRE_OPTIONS)

TIMBRE_PRESETS: list[tuple[str, list[str]]] = [
    ("Classic Organ Choir", ["Organ", "Organ", "Organ", "Organ"]),
    ("Soft Pads", ["Soft Organ", "Strings", "Strings", "Soft Organ"]),
    ("Bright Synth Stack", ["Saw", "Square", "Pulse 25%", "Triangle"]),
    ("FM Keys + Pad", ["FM EP", "FM EP", "Strings", "Strings"]),
    ("Bell Choir", ["Bell", "Bell", "Sine", "Triangle"]),
    ("Brass Section", ["Brass", "Brass", "Brass", "Saw"]),
    ("Pulse Ensemble", ["Pulse 12.5%", "Pulse 25%", "Square", "Triangle"]),
    ("Warm Trio + Lead", ["Triangle", "Soft Organ", "Strings", "Saw"]),
    ("Minimal", ["Sine", "Sine", "Sine", "Sine"]),
    ("Hybrid Organ + FM", ["Organ", "FM EP", "Soft Organ", "Bell"]),
    ("Chiptune 1", ["Triangle", "Pulse 12.5%", "Square", "Pulse 25%"]),
    ("Chiptune 2", ["Sine", "Square", "Pulse 25%", "Triangle"]),
    ("Chiptune 3", ["Triangle", "Square", "Pulse 12.5%", "Sine"]),
    ("Chiptune 4", ["Square", "Pulse 25%", "Square", "Pulse 12.5%"]),
    ("Chiptune 5", ["Triangle", "Saw", "Pulse 25%", "Square"]),
]

DEFAULT_GROUPS = {
    "Top 8": {
        "letters": list(LETTER_FREQ[:8]),
        "v1": [1, 1, -1, 2, -2, -1, 1, -1],
        "v2": [-1, -1, 1, -2, 2, 1, -1, 1],
        "v3": [1, -1, 2, -1, 1, -2, 1, -1],
        "v4": [-1, 1, -2, 1, -1, 2, -1, 1],
    },
    "Next 8": {
        "letters": list(LETTER_FREQ[8:16]),
        "v1": [2, -2, 1, -1, 2, 1, -2, -3],
        "v2": [-2, 2, -1, 1, -2, -1, 2, 3],
        "v3": [2, -1, 2, -2, 1, -2, 2, -3],
        "v4": [-2, 1, -2, 2, -1, 2, -2, 3],
    },
    "Next 6": {
        "letters": list(LETTER_FREQ[16:22]),
        "v1": [3, -3, 2, -2, 3, -3],
        "v2": [-3, 3, -2, 2, -3, 3],
        "v3": [3, -2, 3, -3, 2, -3],
        "v4": [-3, 2, -3, 3, -2, 3],
    },
    "Rarest 4": {
        "letters": list(LETTER_FREQ[22:]),
        "v1": [5, -5, 4, -4],
        "v2": [-5, 5, -4, 4],
        "v3": [4, -5, 5, -4],
        "v4": [-4, 5, -5, 4],
    },
}

GROUPS = {
    name: {
        "letters": group["letters"][:],
        "v1": group["v1"][:],
        "v2": group["v2"][:],
        "v3": group["v3"][:],
        "v4": group["v4"][:],
    }
    for name, group in DEFAULT_GROUPS.items()
}

TENSION_OPTIONS = list(MODE_DEFS.keys())

PUNCT_SYMBOLS: list[tuple[str, str]] = [
    ("SPC", " "),
    (",", ","),
    (".", "."),
    ("'", "'"),
    ('"', '"'),
    ("-", "-"),
    ("!", "!"),
    ("?", "?"),
    ("(", "("),
    (")", ")"),
    (":", ":"),
    (";", ";"),
    ("/", "/"),
    ("[", "["),
    ("]", "]"),
    ("_", "_"),
    ("+", "+"),
    ("=", "="),
    ("\\", "\\"),
    ("|", "|"),
    ("{", "{"),
    ("}", "}"),
    ("<", "<"),
    (">", ">"),
    ("~", "~"),
    ("`", "`"),
    ("@", "@"),
    ("#", "#"),
    ("$", "$"),
    ("%", "%"),
    ("^", "^"),
    ("&", "&"),
    ("*", "*"),
    ("TAB", "\t"),
    ("BKSP", "\b"),
    ("DEL", "\x7f"),
    ("ENT", "\n"),
]

DIGIT_MOVES = {
    "v3": {str(d): d for d in range(10)},
    "v4": {str(d): -d for d in range(10)},
}

PUNCT_MOVES = {"v1": {}, "v2": {}, "v3": {}, "v4": {}}
PUNCT_SILENT = {"v1": {}, "v2": {}, "v3": {}, "v4": {}}
DIGIT_SILENT = {"v3": {str(d): 0 for d in range(10)}, "v4": {str(d): 0 for d in range(10)}}

# Per-letter bucket silence flags (aligned to each group's letters list)
LETTER_SILENT = {
    name: {"v1": [0] * len(group["letters"]), "v2": [0] * len(group["letters"]), "v3": [0] * len(group["letters"]), "v4": [0] * len(group["letters"])}
    for name, group in GROUPS.items()
}


def init_punct_moves_by_frequency():
    common = [sym for _label, sym in PUNCT_SYMBOLS if sym not in (" ", "\n", "\t", "\b", "\x7f")]
    tiers = [common[:8], common[8:16], common[16:24], common[24:30], common[30:]]
    tier_patterns = [
        {
            "v1": [1, 1, -1, 2, -2, -1, 1, -1],
            "v2": [-1, -1, 1, -2, 2, 1, -1, 1],
            "v3": [1, -1, 2, -1, 1, -2, 1, -1],
            "v4": [-1, 1, -2, 1, -1, 2, -1, 1],
        },
        {
            "v1": [2, -2, 1, -1, 2, 1, -2, -3],
            "v2": [-2, 2, -1, 1, -2, -1, 2, 3],
            "v3": [2, -1, 2, -2, 1, -2, 2, -3],
            "v4": [-2, 1, -2, 2, -1, 2, -2, 3],
        },
        {
            "v1": [3, -3, 2, -2, 3, -3, 2, -2],
            "v2": [-3, 3, -2, 2, -3, 3, -2, 2],
            "v3": [3, -2, 3, -3, 2, -3, 2, -2],
            "v4": [-3, 2, -3, 3, -2, 3, -2, 2],
        },
        {
            "v1": [5, -5, 4, -4, 5, -5],
            "v2": [-5, 5, -4, 4, -5, 5],
            "v3": [4, -5, 5, -4, 4, -5],
            "v4": [-4, 5, -5, 4, -4, 5],
        },
        {
            "v1": [7, -7, 6, -6],
            "v2": [-7, 7, -6, 6],
            "v3": [6, -7, 7, -6],
            "v4": [-6, 7, -7, 6],
        },
    ]

    for voice_key in ["v1", "v2", "v3", "v4"]:
        PUNCT_MOVES[voice_key].clear()
        PUNCT_SILENT[voice_key].clear()
        PUNCT_MOVES[voice_key][" "] = 0
        PUNCT_MOVES[voice_key]["\n"] = 0
        PUNCT_MOVES[voice_key]["\t"] = 0
        PUNCT_MOVES[voice_key]["\b"] = 0
        PUNCT_MOVES[voice_key]["\x7f"] = 0
        PUNCT_SILENT[voice_key][" "] = 0
        PUNCT_SILENT[voice_key]["\n"] = 0
        PUNCT_SILENT[voice_key]["\t"] = 0
        PUNCT_SILENT[voice_key]["\b"] = 0
        PUNCT_SILENT[voice_key]["\x7f"] = 0

    for tier_syms, patterns in zip(tiers, tier_patterns):
        for i, sym in enumerate(tier_syms):
            for voice_key in ["v1", "v2", "v3", "v4"]:
                pat = patterns[voice_key]
                PUNCT_MOVES[voice_key][sym] = pat[i % len(pat)]
                PUNCT_SILENT[voice_key][sym] = 0

init_punct_moves_by_frequency()

def init_letter_movements():
    """Initialize letter movement mappings for 4 voices"""
    voice1_movements.clear()
    voice2_movements.clear()
    voice3_movements.clear()
    voice4_movements.clear()

    for group in (GROUPS["Top 8"], GROUPS["Next 8"], GROUPS["Next 6"], GROUPS["Rarest 4"]):
        for i, ch in enumerate(group["letters"]):
            voice1_movements[ch] = group["v1"][i]
            voice2_movements[ch] = group["v2"][i]
            voice3_movements[ch] = group["v3"][i]
            voice4_movements[ch] = group["v4"][i]

    for _label, sym in PUNCT_SYMBOLS:
        voice1_movements[sym] = PUNCT_MOVES["v1"][sym]
        voice2_movements[sym] = PUNCT_MOVES["v2"][sym]
        voice3_movements[sym] = PUNCT_MOVES["v3"][sym]
        voice4_movements[sym] = PUNCT_MOVES["v4"][sym]


def get_character_action(char: str, is_upper: bool, is_digit: bool):
    """Map characters to musical actions"""
    if is_digit:
        with STATE_LOCK:
            if char not in DIGIT_MOVES["v3"]:
                return None
            v3_steps = DIGIT_MOVES["v3"][char]
            v4_steps = DIGIT_MOVES["v4"][char]
            v3_on = DIGIT_SILENT["v3"][char] == 0
            v4_on = DIGIT_SILENT["v4"][char] == 0
        return {
            "type": "note",
            "v1": 0,
            "v2": 0,
            "v3": v3_steps,
            "v4": v4_steps,
            "enabled": (False, False, v3_on, v4_on),
            "source": "digits",
            "duration": 0.15,
            "char": char,
        }

    with STATE_LOCK:
        if char not in voice1_movements:
            return None
        v1_steps = voice1_movements[char]
        v2_steps = voice2_movements[char]
        v3_steps = voice3_movements[char]
        v4_steps = voice4_movements[char]

        v1_on = True
        v2_on = True
        v3_on = True
        v4_on = True
        if char in voice1_movements and char not in LETTER_FREQ:
            v1_on = PUNCT_SILENT["v1"].get(char, 0) == 0
            v2_on = PUNCT_SILENT["v2"].get(char, 0) == 0
            v3_on = PUNCT_SILENT["v3"].get(char, 0) == 0
            v4_on = PUNCT_SILENT["v4"].get(char, 0) == 0
        else:
            if char in LETTER_FREQ:
                if char in GROUPS["Top 8"]["letters"]:
                    g = "Top 8"
                elif char in GROUPS["Next 8"]["letters"]:
                    g = "Next 8"
                elif char in GROUPS["Next 6"]["letters"]:
                    g = "Next 6"
                else:
                    g = "Rarest 4"
                pos = GROUPS[g]["letters"].index(char)
                v1_on = LETTER_SILENT[g]["v1"][pos] == 0
                v2_on = LETTER_SILENT[g]["v2"][pos] == 0
                v3_on = LETTER_SILENT[g]["v3"][pos] == 0
                v4_on = LETTER_SILENT[g]["v4"][pos] == 0

    if char in (" ", "\n", "\t", "\b", "\x7f") and (
        (v1_steps == 0 and v2_steps == 0 and v3_steps == 0 and v4_steps == 0) or (not (v1_on or v2_on or v3_on or v4_on))
    ):
        return {"type": "rest", "duration": 0.15, "char": "·", "v1": 0, "v2": 0, "v3": 0, "v4": 0}

    if is_upper:
        return {
            "type": "note",
            "v1": v1_steps,
            "v2": v2_steps,
            "v3": 0,
            "v4": 0,
            "enabled": (v1_on, v2_on, False, False),
            "source": "letters",
            "duration": 0.15,
            "char": char,
        }

    return {
        "type": "note",
        "v1": v1_steps,
        "v2": v2_steps,
        "v3": v3_steps,
        "v4": v4_steps,
        "enabled": (v1_on, v2_on, v3_on, v4_on),
        "source": "letters",
        "duration": 0.15,
        "char": char,
    }


def generate_tone(frequency, duration=0.15, volume=0.2):
    """Generate a musical tone with envelope"""
    sample_rate = 44100
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)

    # Sine wave with harmonics
    wave = np.sin(2 * np.pi * frequency * t)
    wave += 0.3 * np.sin(4 * np.pi * frequency * t)
    wave += 0.15 * np.sin(6 * np.pi * frequency * t)

    # ADSR envelope
    attack = min(0.02, duration * 0.1)
    decay = min(0.05, duration * 0.2)
    sustain_level = 0.6

    envelope = np.ones_like(t)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)

    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0 and attack_samples + decay_samples < len(envelope):
        envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
        envelope[attack_samples + decay_samples:] = sustain_level

    # Release
    release_samples = int(0.05 * sample_rate)
    if release_samples < len(envelope):
        envelope[-release_samples:] *= np.linspace(1, 0, release_samples)

    wave = wave * envelope * volume
    return wave.astype(np.float32)


def generate_quad_tone(freq1, freq2, freq3, freq4, duration=0.15, volume=0.12, enabled=(True, True, True, True)):
    """Generate four-note harmony"""
    sample_rate = 44100
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)

    def make_wave(freq: float, timbre: str, on: bool) -> np.ndarray:
        if not on:
            return np.zeros_like(t)
        phase = 2 * np.pi * freq * t

        if timbre == "Sine":
            return np.sin(phase)
        if timbre == "Triangle":
            return (2 / np.pi) * np.arcsin(np.sin(phase))
        if timbre == "Square":
            return np.sign(np.sin(phase))
        if timbre == "Saw":
            x = freq * t
            return 2 * (x - np.floor(x + 0.5))
        if timbre == "Pulse 25%":
            x = freq * t
            frac = x - np.floor(x)
            return np.where(frac < 0.25, 1.0, -1.0)
        if timbre == "Pulse 12.5%":
            x = freq * t
            frac = x - np.floor(x)
            return np.where(frac < 0.125, 1.0, -1.0)
        if timbre == "Organ":
            return np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
        if timbre == "Soft Organ":
            return np.sin(phase) + 0.18 * np.sin(2 * phase) + 0.07 * np.sin(3 * phase) + 0.04 * np.sin(5 * phase)
        if timbre == "Strings":
            detune = 0.003
            phase2 = 2 * np.pi * (freq * (1 + detune)) * t
            phase3 = 2 * np.pi * (freq * (1 - detune)) * t
            return (
                0.6 * (np.sin(phase) + np.sin(phase2) + np.sin(phase3)) / 3
                + 0.25 * np.sin(2 * phase)
                + 0.1 * np.sin(3 * phase)
            )
        if timbre == "Brass":
            return (
                np.sin(phase)
                + 0.55 * np.sin(2 * phase)
                + 0.32 * np.sin(3 * phase)
                + 0.18 * np.sin(4 * phase)
                + 0.1 * np.sin(5 * phase)
            )
        if timbre == "Bell":
            return (
                0.9 * np.sin(phase) * np.exp(-6.0 * t)
                + 0.5 * np.sin(2.7 * phase) * np.exp(-7.5 * t)
                + 0.35 * np.sin(5.8 * phase) * np.exp(-9.0 * t)
            )
        if timbre == "FM EP":
            mod = 2.0
            index = 3.0
            return np.sin(phase + index * np.sin(mod * phase))

        return np.sin(phase)

    wave1 = make_wave(freq1, TIMBRES[0], enabled[0])
    wave2 = make_wave(freq2, TIMBRES[1], enabled[1])
    wave3 = make_wave(freq3, TIMBRES[2], enabled[2])
    wave4 = make_wave(freq4, TIMBRES[3], enabled[3])
    active = int(enabled[0]) + int(enabled[1]) + int(enabled[2]) + int(enabled[3])
    denom = active if active else 1
    wave = (wave1 + wave2 + wave3 + wave4) / denom

    # ADSR envelope
    attack = min(0.02, duration * 0.1)
    decay = min(0.05, duration * 0.2)
    sustain_level = 0.6

    envelope = np.ones_like(t)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)

    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0 and attack_samples + decay_samples < len(envelope):
        envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
        envelope[attack_samples + decay_samples:] = sustain_level

    # Release
    release_samples = int(0.05 * sample_rate)
    if release_samples < len(envelope):
        envelope[-release_samples:] *= np.linspace(1, 0, release_samples)

    wave = wave * envelope * volume
    return wave.astype(np.float32)


def audio_worker():
    """Dedicated audio thread - processes queue"""
    global running

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=44100,
        output=True,
        frames_per_buffer=1024
    )

    print("Audio thread started")

    try:
        while running:
            try:
                action = audio_queue.get(timeout=0.1)

                if action is None:  # Shutdown signal
                    break

                if action['type'] == 'rest':
                    time.sleep(action['duration'])
                    if PLAYBACK_BUS is not None:
                        PLAYBACK_BUS.updated.emit(
                            {
                                "type": "rest",
                                "symbol": action.get("char", "·"),
                            }
                        )
                elif action['type'] == 'note':
                    with STATE_LOCK:
                        enabled = action.get("enabled", (True, True, True, True))
                        source = action.get("source", "letters")
                        v1_map = voice1_movements
                        v2_map = voice2_movements
                        v3_map = DIGIT_MOVES["v3"] if source == "digits" else voice3_movements
                        v4_map = DIGIT_MOVES["v4"] if source == "digits" else voice4_movements
                        prev_notes = [
                            music_state.get_note_name(music_state.voice1_index),
                            music_state.get_note_name(music_state.voice2_index),
                            music_state.get_note_name(music_state.voice3_index),
                            music_state.get_note_name(music_state.voice4_index),
                        ]
                        prev_midis_all = [
                            music_state.get_midi(music_state.voice1_index),
                            music_state.get_midi(music_state.voice2_index),
                            music_state.get_midi(music_state.voice3_index),
                            music_state.get_midi(music_state.voice4_index),
                        ]
                        if enabled[0]:
                            f1 = music_state._move_voice("voice1_index", action["v1"], action["char"], v1_map)
                        else:
                            f1 = music_state.get_frequency(music_state.voice1_index)
                        if enabled[1]:
                            f2 = music_state._move_voice("voice2_index", action["v2"], action["char"], v2_map)
                        else:
                            f2 = music_state.get_frequency(music_state.voice2_index)
                        if enabled[2]:
                            f3 = music_state._move_voice("voice3_index", action["v3"], action["char"], v3_map)
                        else:
                            f3 = music_state.get_frequency(music_state.voice3_index)
                        if enabled[3]:
                            f4 = music_state._move_voice("voice4_index", action["v4"], action["char"], v4_map)
                        else:
                            f4 = music_state.get_frequency(music_state.voice4_index)
                        tone = generate_quad_tone(f1, f2, f3, f4, action["duration"], enabled=enabled)
                    stream.write(tone.tobytes())

                symbol = action['char']
                with STATE_LOCK:
                    n1 = music_state.get_note_name(music_state.voice1_index)
                    n2 = music_state.get_note_name(music_state.voice2_index)
                    n3 = music_state.get_note_name(music_state.voice3_index)
                    n4 = music_state.get_note_name(music_state.voice4_index)
                    enabled_list = list(action.get("enabled", (True, True, True, True)))
                    cur_midis_all = [
                        music_state.get_midi(music_state.voice1_index),
                        music_state.get_midi(music_state.voice2_index),
                        music_state.get_midi(music_state.voice3_index),
                        music_state.get_midi(music_state.voice4_index),
                    ]
                    prev_midis = [m for on, m in zip(enabled_list, prev_midis_all) if on]
                    cur_midis = [m for on, m in zip(enabled_list, cur_midis_all) if on]
                    prev_chord_name = chord_name_from_midis(prev_midis) if prev_midis else "—"
                    chord_name = chord_name_from_midis(cur_midis) if cur_midis else "—"
                print(f"{symbol}\t{action['v1']}\t{action['v2']}\t{action['v3']}\t{action['v4']}")
                print(f"\t{n1}\t{n2}\t{n3}\t{n4}")
                print(f"\tChord: {chord_name}")

                if PLAYBACK_BUS is not None and action["type"] == "note":
                    PLAYBACK_BUS.updated.emit(
                        {
                            "type": "note",
                            "symbol": symbol,
                            "steps": [action["v1"], action["v2"], action["v3"], action["v4"]],
                            "prev_notes": prev_notes,
                            "notes": [n1, n2, n3, n4],
                            "enabled": enabled_list,
                            "prev_chord": prev_chord_name,
                            "chord": chord_name,
                        }
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio error: {e}")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio thread stopped")


last_key_time = 0
shift_down = False


def on_press(key):
    """Handle key press events"""
    global last_key_time
    global shift_down

    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        shift_down = True
        return

    current_time = time.time()
    # Debounce
    if current_time - last_key_time < 0.05:
        return
    last_key_time = current_time

    try:
        if key == keyboard.Key.enter:
            char_raw = "\n"
            action = get_character_action(char_raw, is_upper=False, is_digit=False)
        elif key == keyboard.Key.tab:
            char_raw = "\t"
            action = get_character_action(char_raw, is_upper=False, is_digit=False)
        elif key == keyboard.Key.backspace:
            char_raw = "\b"
            action = get_character_action(char_raw, is_upper=False, is_digit=False)
        elif key == keyboard.Key.delete:
            char_raw = "\x7f"
            action = get_character_action(char_raw, is_upper=False, is_digit=False)
        elif hasattr(key, 'char') and key.char is not None:
            char_raw = key.char
            is_upper = char_raw.isalpha() and (char_raw.isupper() or shift_down)
            is_digit = char_raw.isdigit()
            char = char_raw.lower() if char_raw.isalpha() else char_raw
            action = get_character_action(char, is_upper=is_upper, is_digit=is_digit)
        else:
            action = None

        if action:
            # Queue the action instead of playing directly
            audio_queue.put(action)

    except AttributeError:
        pass


def on_release(key):
    """Handle key release - exit on ESC"""
    global running
    global shift_down

    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        shift_down = False
        return

    if key == keyboard.Key.esc:
        print("\nShutting down...")
        running = False
        audio_queue.put(None)  # Signal audio thread to stop
        QCoreApplication.quit()
        return False


def _csv_ints(values: list[int]) -> str:
    return ",".join(str(v) for v in values)


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


SILENT_TOKEN = "S"


def _csv_tokens(values: list[str]) -> str:
    return ",".join(values)


def _parse_csv_tokens(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip() != ""]


def chord_name_from_midis(midis: list[int]) -> str:
    pitches = [m21pitch.Pitch(midi=m) for m in midis]
    c = m21chord.Chord(pitches)
    name = c.pitchedCommonName
    root = c.root().name
    bass = c.bass().name
    if bass != root:
        return f"{name}/{bass}"
    return name


class PlaybackBus(QObject):
    updated = Signal(dict)


PLAYBACK_BUS: Optional[PlaybackBus] = None


def letter_group_pos(letter: str) -> tuple[str, int]:
    if letter in GROUPS["Top 8"]["letters"]:
        return "Top 8", GROUPS["Top 8"]["letters"].index(letter)
    if letter in GROUPS["Next 8"]["letters"]:
        return "Next 8", GROUPS["Next 8"]["letters"].index(letter)
    if letter in GROUPS["Next 6"]["letters"]:
        return "Next 6", GROUPS["Next 6"]["letters"].index(letter)
    return "Rarest 4", GROUPS["Rarest 4"]["letters"].index(letter)
def save_config():
    cfg = configparser.ConfigParser()

    cfg["scale"] = {
        "key_pitch_class": str(music_state.key_pitch_class),
        "mode": str(music_state.mode_name),
        "octaves": str(music_state.octaves),
    }
    cfg["voices"] = {
        "voice1_index": str(music_state.voice1_index),
        "voice2_index": str(music_state.voice2_index),
        "voice3_index": str(music_state.voice3_index),
        "voice4_index": str(music_state.voice4_index),
    }
    cfg["timbre"] = {f"voice{i+1}": str(TIMBRES[i]) for i in range(4)}

    for group_name, group in GROUPS.items():
        section = f"intervals:{group_name}"
        cfg[section] = {
            "v1": _csv_tokens([SILENT_TOKEN if LETTER_SILENT[group_name]["v1"][i] else str(group["v1"][i]) for i in range(len(group["letters"]))]),
            "v2": _csv_tokens([SILENT_TOKEN if LETTER_SILENT[group_name]["v2"][i] else str(group["v2"][i]) for i in range(len(group["letters"]))]),
            "v3": _csv_tokens([SILENT_TOKEN if LETTER_SILENT[group_name]["v3"][i] else str(group["v3"][i]) for i in range(len(group["letters"]))]),
            "v4": _csv_tokens([SILENT_TOKEN if LETTER_SILENT[group_name]["v4"][i] else str(group["v4"][i]) for i in range(len(group["letters"]))]),
        }

    cfg["punctuation"] = {}
    for voice_key in ["v1", "v2", "v3", "v4"]:
        cfg["punctuation"][voice_key] = _csv_tokens(
            [
                SILENT_TOKEN if PUNCT_SILENT[voice_key].get(sym, 0) else str(PUNCT_MOVES[voice_key][sym])
                for _label, sym in PUNCT_SYMBOLS
            ]
        )

    cfg["numbers"] = {}
    for voice_key in ["v3", "v4"]:
        cfg["numbers"][voice_key] = _csv_tokens(
            [SILENT_TOKEN if DIGIT_SILENT[voice_key][str(x)] else str(DIGIT_MOVES[voice_key][str(x)]) for x in range(10)]
        )

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        cfg.write(f)


def load_config():
    cfg = configparser.ConfigParser(strict=False, delimiters=("=",))
    cfg.read(CONFIG_PATH, encoding="utf-8")

    with STATE_LOCK:
        if "scale" in cfg:
            music_state.set_key_pitch_class(int(cfg["scale"].get("key_pitch_class", str(music_state.key_pitch_class))))
            music_state.set_mode(str(cfg["scale"].get("mode", music_state.mode_name)))
            music_state.set_octaves(int(cfg["scale"].get("octaves", str(music_state.octaves))))

        if "voices" in cfg:
            indices = [
                int(cfg["voices"].get("voice1_index", str(music_state.voice1_index))),
                int(cfg["voices"].get("voice2_index", str(music_state.voice2_index))),
                int(cfg["voices"].get("voice3_index", str(music_state.voice3_index))),
                int(cfg["voices"].get("voice4_index", str(music_state.voice4_index))),
            ]
            music_state.set_voice_indices(indices)

        if "timbre" in cfg:
            for i in range(4):
                TIMBRES[i] = str(cfg["timbre"].get(f"voice{i+1}", TIMBRES[i]))

        for group_name, group in GROUPS.items():
            section = f"intervals:{group_name}"
            if section in cfg:
                for voice_key in ["v1", "v2", "v3", "v4"]:
                    tokens = _parse_csv_tokens(cfg[section].get(voice_key, _csv_ints(group[voice_key])))
                    for i, ch in enumerate(group["letters"]):
                        tok = tokens[i]
                        if tok == SILENT_TOKEN:
                            group[voice_key][i] = 0
                            LETTER_SILENT[group_name][voice_key][i] = 1
                        else:
                            group[voice_key][i] = int(tok)
                            LETTER_SILENT[group_name][voice_key][i] = 0

        if "punctuation" in cfg:
            if all(vk in cfg["punctuation"] for vk in ["v1", "v2", "v3", "v4"]):
                for voice_key in ["v1", "v2", "v3", "v4"]:
                    tokens = _parse_csv_tokens(cfg["punctuation"][voice_key])
                    for i, (_label, sym) in enumerate(PUNCT_SYMBOLS):
                        tok = tokens[i]
                        if tok == SILENT_TOKEN:
                            PUNCT_MOVES[voice_key][sym] = 0
                            PUNCT_SILENT[voice_key][sym] = 1
                        else:
                            PUNCT_MOVES[voice_key][sym] = int(tok)
                            PUNCT_SILENT[voice_key][sym] = 0
            else:
                for voice_key in ["v1", "v2", "v3", "v4"]:
                    for _label, sym in PUNCT_SYMBOLS:
                        k = f"{voice_key}:{repr(sym)}"
                        if k in cfg["punctuation"]:
                            PUNCT_MOVES[voice_key][sym] = int(cfg["punctuation"][k])
                            PUNCT_SILENT[voice_key][sym] = 0

        if "numbers" in cfg:
            if all(vk in cfg["numbers"] for vk in ["v3", "v4"]):
                for voice_key in ["v3", "v4"]:
                    tokens = _parse_csv_tokens(cfg["numbers"][voice_key])
                    for i, d in enumerate([str(x) for x in range(10)]):
                        tok = tokens[i]
                        if tok == SILENT_TOKEN:
                            DIGIT_MOVES[voice_key][d] = 0
                            DIGIT_SILENT[voice_key][d] = 1
                        else:
                            DIGIT_MOVES[voice_key][d] = int(tok)
                            DIGIT_SILENT[voice_key][d] = 0
            else:
                # Old format / older routing support (v1/v2) -> map into v3/v4
                for src_voice, dst_voice in [("v1", "v3"), ("v2", "v4")]:
                    for d in [str(x) for x in range(10)]:
                        k = f"{src_voice}:{d}"
                        if k in cfg["numbers"]:
                            DIGIT_MOVES[dst_voice][d] = int(cfg["numbers"][k])
                            DIGIT_SILENT[dst_voice][d] = 0

        init_letter_movements()
        save_config()


class MelotypeConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HarmoniKeys by Crawsome")
        self.setMinimumWidth(900)

        self._build_menu()

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        tabs = QTabWidget()
        root_layout.addWidget(tabs)

        tabs.addTab(self._build_basic_tab(), "Basic")
        tabs.addTab(self._build_config_tab(), "Config")
        tabs.addTab(self._build_intervals_tab(), "Intervals")

        footer = QHBoxLayout()
        open_btn = QPushButton("Open project folder")
        open_btn.clicked.connect(self._on_open_project_folder)
        footer.addWidget(open_btn)
        footer.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)
        root_layout.addLayout(footer)

        self._refresh_voice_note_dropdowns()
        self.refresh_from_state()
        tabs.setCurrentIndex(0)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")

        save_action = QAction("Save config.ini", self)
        save_action.triggered.connect(self._on_save_config)
        file_menu.addAction(save_action)

        load_action = QAction("Load config.ini", self)
        load_action.triggered.connect(self._on_load_config)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        open_folder_action = QAction("Open project folder", self)
        open_folder_action.triggered.connect(self._on_open_project_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _on_save_config(self):
        with STATE_LOCK:
            save_config()

    def _on_load_config(self):
        load_config()
        self.refresh_from_state()

    def _on_open_project_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.getcwd()))

    def _on_about(self):
        QMessageBox.about(
            self,
            "About melotype",
            (
                "Made by <a href='https://github.com/crawsome'>github.com/crawsome</a><br/>"
                "See my main project, <b>Wyrm Warrior</b>: "
                "<a href='https://gx.games/games/z0d76p/wyrm-warrior/'>gx.games</a>"
            ),
        )

    def refresh_from_state(self):
        with STATE_LOCK:
            key_pc = music_state.key_pitch_class
            mode_name = music_state.mode_name
            octaves = music_state.octaves
            timbres = TIMBRES[:]
            group_snapshot = {
                g: {
                    "v1": GROUPS[g]["v1"][:],
                    "v2": GROUPS[g]["v2"][:],
                    "v3": GROUPS[g]["v3"][:],
                    "v4": GROUPS[g]["v4"][:],
                }
                for g in GROUPS
            }
            punct_snapshot = {vk: dict(PUNCT_MOVES[vk]) for vk in ["v1", "v2", "v3", "v4"]}
            digit_snapshot = {vk: dict(DIGIT_MOVES[vk]) for vk in ["v3", "v4"]}

        for combo in [self.key_combo, self.basic_key_combo]:
            combo.blockSignals(True)
            for i in range(combo.count()):
                if combo.itemData(i) == key_pc:
                    combo.setCurrentIndex(i)
                    break
            combo.blockSignals(False)

        for combo in [self.tension_combo, self.basic_tension_combo]:
            combo.blockSignals(True)
            combo.setCurrentText(mode_name)
            combo.blockSignals(False)

        self.octave_combo.blockSignals(True)
        self.octave_combo.setCurrentText(str(octaves))
        self.octave_combo.blockSignals(False)

        self._refresh_voice_note_dropdowns()

        for i, cb in enumerate(self.timbre_combos):
            cb.blockSignals(True)
            cb.setCurrentText(timbres[i])
            cb.blockSignals(False)
        self._sync_tone_preset_combos_from_timbres(timbres)

        for (group_name, voice_key, pos), cb in self.interval_combos.items():
            cb.blockSignals(True)
            if LETTER_SILENT[group_name][voice_key][pos]:
                cb.setCurrentText("S")
            else:
                cb.setCurrentText(str(group_snapshot[group_name][voice_key][pos]))
            cb.blockSignals(False)

        for (voice_key, sym), cb in self.punct_combos.items():
            cb.blockSignals(True)
            if PUNCT_SILENT[voice_key].get(sym, 0):
                cb.setCurrentText("S")
            else:
                cb.setCurrentText(str(punct_snapshot[voice_key][sym]))
            cb.blockSignals(False)

        for (voice_key, digit), cb in self.digit_combos.items():
            cb.blockSignals(True)
            if DIGIT_SILENT[voice_key][digit]:
                cb.setCurrentText("S")
            else:
                cb.setCurrentText(str(digit_snapshot[voice_key][digit]))
            cb.blockSignals(False)

        for (voice_key, letter), cb in self.capital_combos.items():
            group_name, pos = letter_group_pos(letter)
            cb.blockSignals(True)
            if LETTER_SILENT[group_name][voice_key][pos]:
                cb.setCurrentText("S")
            else:
                cb.setCurrentText(str(group_snapshot[group_name][voice_key][pos]))
            cb.blockSignals(False)

    def _build_config_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scale_box = QGroupBox("Scale")
        scale_v = QVBoxLayout(scale_box)
        scale_help = QLabel("Use this section to choose key, mode, octaves, and per-voice starting notes. Changes apply live.")
        scale_help.setWordWrap(True)
        scale_v.addWidget(scale_help)
        scale_layout = QGridLayout()
        scale_v.addLayout(scale_layout)
        layout.addWidget(scale_box)

        scale_layout.addWidget(QLabel("Key"), 0, 0)
        self.key_combo = QComboBox()
        for label, pc in KEY_OPTIONS:
            self.key_combo.addItem(label, pc)
        for i in range(self.key_combo.count()):
            if self.key_combo.itemData(i) == music_state.key_pitch_class:
                self.key_combo.setCurrentIndex(i)
                break
        self.key_combo.currentIndexChanged.connect(lambda _idx: self._on_key_changed_from(self.key_combo))
        scale_layout.addWidget(self.key_combo, 0, 1)

        scale_layout.addWidget(QLabel("Tension (mode)"), 1, 0)
        self.tension_combo = QComboBox()
        for label in TENSION_OPTIONS:
            self.tension_combo.addItem(label, label)
        self.tension_combo.setCurrentText(music_state.mode_name)
        self.tension_combo.currentIndexChanged.connect(lambda _idx: self._on_tension_changed_from(self.tension_combo))
        scale_layout.addWidget(self.tension_combo, 1, 1)

        scale_layout.addWidget(QLabel("# Octaves"), 2, 0)
        self.octave_combo = QComboBox()
        for o in range(1, 6):
            self.octave_combo.addItem(str(o), o)
        self.octave_combo.setCurrentText(str(music_state.octaves))
        self.octave_combo.currentIndexChanged.connect(self._on_octaves_changed)
        scale_layout.addWidget(self.octave_combo, 2, 1)

        self.voice_note_combos: list[QComboBox] = []
        for i in range(4):
            scale_layout.addWidget(QLabel(f"Voice {i + 1} start note"), 3 + i, 0)
            combo = QComboBox()
            combo.currentIndexChanged.connect(lambda _idx, v=i: self._on_voice_note_changed(v))
            self.voice_note_combos.append(combo)
            scale_layout.addWidget(combo, 3 + i, 1)

        tone_box = QGroupBox("Tone")
        tone_v = QVBoxLayout(tone_box)
        tone_help = QLabel("Use this section to choose a timbre (waveform/preset) for each voice. Changes apply live.")
        tone_help.setWordWrap(True)
        tone_v.addWidget(tone_help)
        tone_layout = QGridLayout()
        tone_v.addLayout(tone_layout)
        layout.addWidget(tone_box)

        tone_layout.addWidget(QLabel("Preset"), 0, 0)
        self.tone_preset_combo = QComboBox()
        self.tone_preset_combo.addItem("Custom", None)
        for name, timbres in TIMBRE_PRESETS:
            self.tone_preset_combo.addItem(name, timbres)
        self.tone_preset_combo.currentIndexChanged.connect(lambda _idx: self._on_tone_preset_changed_from(self.tone_preset_combo))
        tone_layout.addWidget(self.tone_preset_combo, 0, 1)

        self.timbre_combos: list[QComboBox] = []
        for i in range(4):
            tone_layout.addWidget(QLabel(f"Voice {i + 1} timbre"), i + 1, 0)
            combo = QComboBox()
            for timbre in TIMBRE_OPTIONS:
                combo.addItem(timbre, timbre)
            combo.setCurrentText(TIMBRES[i])
            combo.currentIndexChanged.connect(lambda _idx, v=i: self._on_timbre_changed(v))
            self.timbre_combos.append(combo)
            tone_layout.addWidget(combo, i + 1, 1)

        layout.addStretch(1)
        return tab

    def _build_intervals_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)

        top_help = QLabel(
            "Use this page to customize how keys map to interval movements. "
            "Common symbols should usually have smaller movements; rare symbols can have bigger jumps."
        )
        top_help.setWordWrap(True)
        outer.addWidget(top_help)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        contents = QWidget()
        scroll.setWidget(contents)
        layout = QVBoxLayout(contents)

        interval_value_options = list(range(-8, 9))
        misc_value_options = list(range(-8, 9))

        def _style_compact_combo(cb: QComboBox):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
            cb.setMinimumContentsLength(2)
            return cb

        self.interval_combos: dict[tuple[str, str, int], QComboBox] = {}

        for group_name, group in GROUPS.items():
            box = QGroupBox(group_name)
            layout.addWidget(box)
            box_v = QVBoxLayout(box)
            box_help = QLabel("Use this section to set per-voice interval moves for this letter-frequency group.")
            box_help.setWordWrap(True)
            box_v.addWidget(box_help)
            grid = QGridLayout()
            box_v.addLayout(grid)

            grid.addWidget(QLabel(""), 0, 0)
            for col, letter in enumerate(group["letters"]):
                grid.addWidget(QLabel(letter), 0, col + 1, alignment=Qt.AlignCenter)

            for row, voice_key in enumerate(["v1", "v2", "v3", "v4"]):
                grid.addWidget(QLabel(voice_key.upper()), row + 1, 0)
                for col, _letter in enumerate(group["letters"]):
                    cb = QComboBox()
                    _style_compact_combo(cb)
                    cb.addItem("S", "silent")
                    for v in interval_value_options:
                        cb.addItem(str(v), v)
                    current = group[voice_key][col]
                    if LETTER_SILENT[group_name][voice_key][col]:
                        cb.setCurrentText("S")
                    else:
                        cb.setCurrentText(str(current))
                    cb.currentIndexChanged.connect(
                        lambda _idx, g=group_name, vk=voice_key, pos=col: self._on_interval_changed(g, vk, pos)
                    )
                    self.interval_combos[(group_name, voice_key, col)] = cb
                    grid.addWidget(cb, row + 1, col + 1)

        punct_box = QGroupBox("Punctuation")
        layout.addWidget(punct_box)
        punct_v = QVBoxLayout(punct_box)
        punct_help = QLabel("Use this section to set per-voice interval moves for punctuation and special keys (SPACE/ENTER/TAB/etc).")
        punct_help.setWordWrap(True)
        punct_v.addWidget(punct_help)
        punct_grid = QGridLayout()
        punct_v.addLayout(punct_grid)
        punct_grid.addWidget(QLabel(""), 0, 0)
        for col, (label, _sym) in enumerate(PUNCT_SYMBOLS):
            h = QLabel(label)
            h.setAlignment(Qt.AlignCenter)
            h.setWordWrap(True)
            punct_grid.addWidget(h, 0, col + 1, alignment=Qt.AlignCenter)

        self.punct_combos: dict[tuple[str, str], QComboBox] = {}
        for row, voice_key in enumerate(["v1", "v2", "v3", "v4"]):
            punct_grid.addWidget(QLabel(voice_key.upper()), row + 1, 0)
            for col, (_label, ch) in enumerate(PUNCT_SYMBOLS):
                cb = QComboBox()
                _style_compact_combo(cb)
                cb.addItem("S", "silent")
                for v in misc_value_options:
                    cb.addItem(str(v), v)
                if PUNCT_SILENT[voice_key].get(ch, 0):
                    cb.setCurrentText("S")
                else:
                    cb.setCurrentText(str(PUNCT_MOVES[voice_key][ch]))
                cb.currentIndexChanged.connect(lambda _idx, vk=voice_key, c=ch: self._on_punct_changed(vk, c))
                self.punct_combos[(voice_key, ch)] = cb
                punct_grid.addWidget(cb, row + 1, col + 1)

        caps_box = QGroupBox("Capitals (voices 1–2 only)")
        layout.addWidget(caps_box)
        caps_v = QVBoxLayout(caps_box)
        caps_help = QLabel("Use this section to set interval moves for A–Z. Capitals only drive voices 1–2.")
        caps_help.setWordWrap(True)
        caps_v.addWidget(caps_help)
        caps_grid = QGridLayout()
        caps_v.addLayout(caps_grid)

        caps_grid.addWidget(QLabel(""), 0, 0)
        for col, ch in enumerate(string.ascii_uppercase):
            caps_grid.addWidget(QLabel(ch), 0, col + 1, alignment=Qt.AlignCenter)

        self.capital_combos: dict[tuple[str, str], QComboBox] = {}
        for row, voice_key in enumerate(["v1", "v2"]):
            caps_grid.addWidget(QLabel(voice_key.upper()), row + 1, 0)
            for col, upper in enumerate(string.ascii_uppercase):
                letter = upper.lower()
                group_name, pos = letter_group_pos(letter)
                cb = QComboBox()
                _style_compact_combo(cb)
                cb.addItem("S", "silent")
                for v in interval_value_options:
                    cb.addItem(str(v), v)
                if LETTER_SILENT[group_name][voice_key][pos]:
                    cb.setCurrentText("S")
                else:
                    cb.setCurrentText(str(GROUPS[group_name][voice_key][pos]))
                cb.currentIndexChanged.connect(
                    lambda _idx, vk=voice_key, ltr=letter: self._on_capital_changed(vk, ltr)
                )
                self.capital_combos[(voice_key, letter)] = cb
                caps_grid.addWidget(cb, row + 1, col + 1)

        digits_box = QGroupBox("Numbers (voices 3–4 only)")
        layout.addWidget(digits_box)
        digits_v = QVBoxLayout(digits_box)
        digits_help = QLabel("Use this section to set interval moves for digits 0–9. Numbers only drive voices 3–4.")
        digits_help.setWordWrap(True)
        digits_v.addWidget(digits_help)
        digits_grid = QGridLayout()
        digits_v.addLayout(digits_grid)
        digits_grid.addWidget(QLabel(""), 0, 0)
        for col, d in enumerate([str(x) for x in range(10)]):
            digits_grid.addWidget(QLabel(d), 0, col + 1, alignment=Qt.AlignCenter)

        self.digit_combos: dict[tuple[str, str], QComboBox] = {}
        for row, voice_key in enumerate(["v3", "v4"]):
            digits_grid.addWidget(QLabel(voice_key.upper()), row + 1, 0)
            for col, d in enumerate([str(x) for x in range(10)]):
                cb = QComboBox()
                _style_compact_combo(cb)
                cb.addItem("S", "silent")
                for v in misc_value_options:
                    cb.addItem(str(v), v)
                if DIGIT_SILENT[voice_key][d]:
                    cb.setCurrentText("S")
                else:
                    cb.setCurrentText(str(DIGIT_MOVES[voice_key][d]))
                cb.currentIndexChanged.connect(lambda _idx, vk=voice_key, digit=d: self._on_digit_changed(vk, digit))
                self.digit_combos[(voice_key, d)] = cb
                digits_grid.addWidget(cb, row + 1, col + 1)

        layout.addStretch(1)
        return tab

    def _on_capital_changed(self, voice_key: str, letter: str):
        cb = self.capital_combos[(voice_key, letter)]
        group_name, pos = letter_group_pos(letter)
        with STATE_LOCK:
            if cb.currentData() == "silent":
                GROUPS[group_name][voice_key][pos] = 0
                LETTER_SILENT[group_name][voice_key][pos] = 1
            else:
                value = int(cb.currentData())
                GROUPS[group_name][voice_key][pos] = value
                LETTER_SILENT[group_name][voice_key][pos] = 0
            init_letter_movements()

        other = self.interval_combos[(group_name, voice_key, pos)]
        other.blockSignals(True)
        other.setCurrentText("S" if LETTER_SILENT[group_name][voice_key][pos] else str(GROUPS[group_name][voice_key][pos]))
        other.blockSignals(False)

    def _scale_note_options(self) -> list[tuple[str, int]]:
        with STATE_LOCK:
            return [(music_state.get_note_name(i), i) for i in range(0, music_state.max_scale_index + 1)]

    def _refresh_voice_note_dropdowns(self):
        options = self._scale_note_options()
        with STATE_LOCK:
            current_indices = [
                music_state.voice1_index,
                music_state.voice2_index,
                music_state.voice3_index,
                music_state.voice4_index,
            ]

        for voice_i, combo in enumerate(self.voice_note_combos):
            combo.setUpdatesEnabled(False)
            combo.blockSignals(True)
            combo.clear()
            for label, idx in options:
                combo.addItem(label, idx)
            target = current_indices[voice_i]
            for j in range(combo.count()):
                if combo.itemData(j) == target:
                    combo.setCurrentIndex(j)
                    break
            combo.blockSignals(False)
            combo.setUpdatesEnabled(True)

    def _on_key_changed(self, _idx: int):
        pitch_class = int(self.key_combo.currentData())
        with STATE_LOCK:
            music_state.set_key_pitch_class(pitch_class)
        self._refresh_voice_note_dropdowns()

    def _on_octaves_changed(self, _idx: int):
        octaves = int(self.octave_combo.currentData())
        with STATE_LOCK:
            music_state.set_octaves(octaves)
        self._refresh_voice_note_dropdowns()

    def _on_voice_note_changed(self, voice_i: int):
        combo = self.voice_note_combos[voice_i]
        idx = int(combo.currentData())
        with STATE_LOCK:
            indices = [music_state.voice1_index, music_state.voice2_index, music_state.voice3_index, music_state.voice4_index]
            indices[voice_i] = idx
            music_state.set_voice_indices(indices)

    def _on_timbre_changed(self, voice_i: int):
        with STATE_LOCK:
            TIMBRES[voice_i] = self.timbre_combos[voice_i].currentData()
            timbres = TIMBRES[:]
        self._sync_tone_preset_combos_from_timbres(timbres)

    def _sync_tone_preset_combos_from_timbres(self, timbres: list[str]):
        match_name = None
        for name, preset_timbres in TIMBRE_PRESETS:
            if preset_timbres == timbres:
                match_name = name
                break
        for combo in [self.tone_preset_combo, self.basic_preset_combo]:
            combo.blockSignals(True)
            if match_name is None:
                combo.setCurrentIndex(0)  # Custom
            else:
                combo.setCurrentText(match_name)
            combo.blockSignals(False)

    def _on_tone_preset_changed_from(self, combo: QComboBox):
        preset = combo.currentData()
        if preset is None:
            return
        with STATE_LOCK:
            for i in range(4):
                TIMBRES[i] = preset[i]
        for i, cb in enumerate(self.timbre_combos):
            cb.blockSignals(True)
            cb.setCurrentText(preset[i])
            cb.blockSignals(False)
        self._sync_tone_preset_combos_from_timbres(list(preset))

    def _on_tension_changed_from(self, combo: QComboBox):
        with STATE_LOCK:
            music_state.set_mode(str(combo.currentData()))
        other = self.tension_combo if combo is self.basic_tension_combo else self.basic_tension_combo
        other.blockSignals(True)
        other.setCurrentText(str(combo.currentData()))
        other.blockSignals(False)
        self._refresh_voice_note_dropdowns()

    def _on_key_changed_from(self, combo: QComboBox):
        pitch_class = int(combo.currentData())
        with STATE_LOCK:
            music_state.set_key_pitch_class(pitch_class)
        other = self.key_combo if combo is self.basic_key_combo else self.basic_key_combo
        other.blockSignals(True)
        for i in range(other.count()):
            if other.itemData(i) == pitch_class:
                other.setCurrentIndex(i)
                break
        other.blockSignals(False)
        self._refresh_voice_note_dropdowns()

    def _build_basic_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        help_lbl = QLabel("Use this page for quick setup: pick a preset, key, and tension/mode. Changes apply live.")
        help_lbl.setWordWrap(True)
        layout.addWidget(help_lbl)

        box = QGroupBox("Quick setup")
        layout.addWidget(box)
        grid = QGridLayout(box)

        grid.addWidget(QLabel("Preset"), 0, 0)
        self.basic_preset_combo = QComboBox()
        self.basic_preset_combo.addItem("Custom", None)
        for name, timbres in TIMBRE_PRESETS:
            self.basic_preset_combo.addItem(name, timbres)
        self.basic_preset_combo.currentIndexChanged.connect(lambda _idx: self._on_tone_preset_changed_from(self.basic_preset_combo))
        grid.addWidget(self.basic_preset_combo, 0, 1)

        grid.addWidget(QLabel("Key"), 1, 0)
        self.basic_key_combo = QComboBox()
        for label, pc in KEY_OPTIONS:
            self.basic_key_combo.addItem(label, pc)
        self.basic_key_combo.currentIndexChanged.connect(lambda _idx: self._on_key_changed_from(self.basic_key_combo))
        grid.addWidget(self.basic_key_combo, 1, 1)

        grid.addWidget(QLabel("Tension (mode)"), 2, 0)
        self.basic_tension_combo = QComboBox()
        for label in TENSION_OPTIONS:
            self.basic_tension_combo.addItem(label, label)
        self.basic_tension_combo.currentIndexChanged.connect(
            lambda _idx: self._on_tension_changed_from(self.basic_tension_combo)
        )
        grid.addWidget(self.basic_tension_combo, 2, 1)

        now_box = QGroupBox("Now Playing")
        layout.addWidget(now_box)
        now_v = QVBoxLayout(now_box)
        now_help = QLabel("This mirrors the console output, but uses fixed labels. It updates as you type.")
        now_help.setWordWrap(True)
        now_v.addWidget(now_help)

        now_grid = QGridLayout()
        now_v.addLayout(now_grid)

        now_grid.addWidget(QLabel(""), 0, 0)
        for i in range(4):
            now_grid.addWidget(QLabel(f"V{i+1}"), 0, i + 1, alignment=Qt.AlignCenter)
        now_grid.addWidget(QLabel("Chord"), 0, 6)

        now_grid.addWidget(QLabel("Previous"), 1, 0)
        self.prev_note_labels: list[QLabel] = []
        for i in range(4):
            lbl = QLabel("—")
            lbl.setMinimumWidth(70)
            lbl.setMaximumWidth(70)
            self.prev_note_labels.append(lbl)
            now_grid.addWidget(lbl, 1, i + 1, alignment=Qt.AlignCenter)

        self.prev_chord_label = QLabel("—")
        self.prev_chord_label.setWordWrap(True)
        self.prev_chord_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        now_grid.addWidget(self.prev_chord_label, 1, 6)

        now_grid.addWidget(QLabel("Key"), 2, 0)
        self.key_used_label = QLabel("—")
        now_grid.addWidget(self.key_used_label, 2, 1, 1, 4)

        now_grid.addWidget(QLabel("Intervals"), 3, 0)
        self.step_labels: list[QLabel] = []
        for i in range(4):
            lbl = QLabel("0")
            lbl.setMinimumWidth(70)
            lbl.setMaximumWidth(70)
            self.step_labels.append(lbl)
            now_grid.addWidget(lbl, 3, i + 1, alignment=Qt.AlignCenter)

        now_grid.addWidget(QLabel("Current"), 4, 0)
        self.note_labels: list[QLabel] = []
        for i in range(4):
            lbl = QLabel("—")
            lbl.setMinimumWidth(70)
            lbl.setMaximumWidth(70)
            self.note_labels.append(lbl)
            now_grid.addWidget(lbl, 4, i + 1, alignment=Qt.AlignCenter)

        self.chord_label = QLabel("—")
        self.chord_label.setWordWrap(True)
        self.chord_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        now_grid.addWidget(self.chord_label, 4, 6)

        now_grid.setColumnMinimumWidth(0, 80)
        for c in [1, 2, 3, 4]:
            now_grid.setColumnMinimumWidth(c, 70)
        now_grid.setColumnMinimumWidth(6, 260)
        now_grid.setColumnStretch(6, 1)

        layout.addStretch(1)
        return tab

    def _on_playback_update(self, payload: dict):
        if payload.get("type") == "rest":
            self.key_used_label.setText("Rest")
            return
        self.key_used_label.setText(str(payload.get("symbol", "—")))
        prev = payload.get("prev_notes", ["—"] * 4)
        cur = payload.get("notes", ["—"] * 4)
        steps = payload.get("steps", [0, 0, 0, 0])
        enabled = payload.get("enabled", [True, True, True, True])
        for i in range(4):
            self.prev_note_labels[i].setText(prev[i] if enabled[i] else "Silent")
            self.step_labels[i].setText(str(steps[i]))
            self.note_labels[i].setText(cur[i] if enabled[i] else "Silent")
        self.prev_chord_label.setText(str(payload.get("prev_chord", "—")))
        self.chord_label.setText(str(payload.get("chord", "—")))

    def _on_interval_changed(self, group_name: str, voice_key: str, pos: int):
        cb = self.interval_combos[(group_name, voice_key, pos)]
        with STATE_LOCK:
            if cb.currentData() == "silent":
                GROUPS[group_name][voice_key][pos] = 0
                LETTER_SILENT[group_name][voice_key][pos] = 1
            else:
                value = int(cb.currentData())
                GROUPS[group_name][voice_key][pos] = value
                LETTER_SILENT[group_name][voice_key][pos] = 0
            init_letter_movements()

    def _on_punct_changed(self, voice_key: str, ch: str):
        cb = self.punct_combos[(voice_key, ch)]
        with STATE_LOCK:
            if cb.currentData() == "silent":
                PUNCT_MOVES[voice_key][ch] = 0
                PUNCT_SILENT[voice_key][ch] = 1
            else:
                value = int(cb.currentData())
                PUNCT_MOVES[voice_key][ch] = value
                PUNCT_SILENT[voice_key][ch] = 0
            init_letter_movements()

    def _on_digit_changed(self, voice_key: str, digit: str):
        cb = self.digit_combos[(voice_key, digit)]
        with STATE_LOCK:
            if cb.currentData() == "silent":
                DIGIT_MOVES[voice_key][digit] = 0
                DIGIT_SILENT[voice_key][digit] = 1
            else:
                value = int(cb.currentData())
                DIGIT_MOVES[voice_key][digit] = value
                DIGIT_SILENT[voice_key][digit] = 0


def main():
    global running

    print("=" * 60)
    print("Typing Music Generator - 4-Part Mode Harmony")
    print("=" * 60)
    print("\n4 voices, each starting in different octaves")
    print("Mode + key are configurable in the GUI")
    print("Range: configurable octaves")
    print("\nPress ESC to exit")
    print("=" * 60)
    print()

    with STATE_LOCK:
        n1 = music_state.get_note_name(music_state.voice1_index)
        n2 = music_state.get_note_name(music_state.voice2_index)
        n3 = music_state.get_note_name(music_state.voice3_index)
        n4 = music_state.get_note_name(music_state.voice4_index)
    print(f"\t{n1}\t{n2}\t{n3}\t{n4}")

    load_config()

    # Start audio thread
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    # Give audio thread time to initialize
    time.sleep(0.2)

    app = QApplication([])
    global PLAYBACK_BUS
    PLAYBACK_BUS = PlaybackBus()
    window = MelotypeConfigWindow()
    PLAYBACK_BUS.updated.connect(window._on_playback_update)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)

    try:
        window.show()
        listener.start()
        app.exec()
    except KeyboardInterrupt:
        print("\nInterrupted...")
    finally:
        running = False
        audio_queue.put(None)
        listener.stop()
        audio_thread.join(timeout=2)
        print("Goodbye!")


if __name__ == "__main__":
    main()