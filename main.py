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
from pynput import keyboard
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
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

    def move_voice1(self, steps, char):
        """Move voice 1"""
        new_index = self.voice1_index + steps

        if new_index > self.max_scale_index:
            voice1_movements[char] = -voice1_movements[char]
            overflow = new_index - self.max_scale_index
            reflected = self.max_scale_index - overflow
            if reflected == self.voice1_index:
                reflected -= 1
            self.voice1_index = reflected
        elif new_index < self.min_scale_index:
            voice1_movements[char] = -voice1_movements[char]
            reflected = self.min_scale_index + (self.min_scale_index - new_index)
            if reflected == self.voice1_index:
                reflected += 1
            self.voice1_index = reflected
        else:
            self.voice1_index = new_index

        return self.get_frequency(self.voice1_index)

    def move_voice2(self, steps, char):
        """Move voice 2"""
        new_index = self.voice2_index + steps

        if new_index > self.max_scale_index:
            voice2_movements[char] = -voice2_movements[char]
            overflow = new_index - self.max_scale_index
            reflected = self.max_scale_index - overflow
            if reflected == self.voice2_index:
                reflected -= 1
            self.voice2_index = reflected
        elif new_index < self.min_scale_index:
            voice2_movements[char] = -voice2_movements[char]
            reflected = self.min_scale_index + (self.min_scale_index - new_index)
            if reflected == self.voice2_index:
                reflected += 1
            self.voice2_index = reflected
        else:
            self.voice2_index = new_index

        return self.get_frequency(self.voice2_index)

    def move_voice3(self, steps, char):
        """Move voice 3"""
        new_index = self.voice3_index + steps

        if new_index > self.max_scale_index:
            voice3_movements[char] = -voice3_movements[char]
            overflow = new_index - self.max_scale_index
            reflected = self.max_scale_index - overflow
            if reflected == self.voice3_index:
                reflected -= 1
            self.voice3_index = reflected
        elif new_index < self.min_scale_index:
            voice3_movements[char] = -voice3_movements[char]
            reflected = self.min_scale_index + (self.min_scale_index - new_index)
            if reflected == self.voice3_index:
                reflected += 1
            self.voice3_index = reflected
        else:
            self.voice3_index = new_index

        return self.get_frequency(self.voice3_index)

    def move_voice4(self, steps, char):
        """Move voice 4"""
        new_index = self.voice4_index + steps

        if new_index > self.max_scale_index:
            voice4_movements[char] = -voice4_movements[char]
            overflow = new_index - self.max_scale_index
            reflected = self.max_scale_index - overflow
            if reflected == self.voice4_index:
                reflected -= 1
            self.voice4_index = reflected
        elif new_index < self.min_scale_index:
            voice4_movements[char] = -voice4_movements[char]
            reflected = self.min_scale_index + (self.min_scale_index - new_index)
            if reflected == self.voice4_index:
                reflected += 1
            self.voice4_index = reflected
        else:
            self.voice4_index = new_index

        return self.get_frequency(self.voice4_index)


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

PUNCT_MOVES = {
    "v1": {",": -1, ".": -2},
    "v2": {",": 1, ".": 2},
    "v3": {",": -1, ".": 1},
    "v4": {",": 2, ".": -1},
}

TENSION_OPTIONS = list(MODE_DEFS.keys())


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

    voice1_movements[","] = PUNCT_MOVES["v1"][","]
    voice1_movements["."] = PUNCT_MOVES["v1"]["."]
    voice2_movements[","] = PUNCT_MOVES["v2"][","]
    voice2_movements["."] = PUNCT_MOVES["v2"]["."]
    voice3_movements[","] = PUNCT_MOVES["v3"][","]
    voice3_movements["."] = PUNCT_MOVES["v3"]["."]
    voice4_movements[","] = PUNCT_MOVES["v4"][","]
    voice4_movements["."] = PUNCT_MOVES["v4"]["."]


# Initialize on load
init_letter_movements()


def get_character_action(char):
    """Map characters to musical actions"""

    if char == ' ':
        return {'type': 'rest', 'duration': 0.15, 'char': '·', 'v1': 0, 'v2': 0, 'v3': 0, 'v4': 0}

    with STATE_LOCK:
        if char not in voice1_movements:
            return None
        v1_steps = voice1_movements[char]
        v2_steps = voice2_movements[char]
        v3_steps = voice3_movements[char]
        v4_steps = voice4_movements[char]

    return {'type': 'note', 'v1': v1_steps, 'v2': v2_steps, 'v3': v3_steps, 'v4': v4_steps, 'duration': 0.15,
            'char': char}


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


def generate_quad_tone(freq1, freq2, freq3, freq4, duration=0.15, volume=0.12):
    """Generate four-note harmony"""
    sample_rate = 44100
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)

    def make_wave(freq: float, timbre: str) -> np.ndarray:
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

    wave1 = make_wave(freq1, TIMBRES[0])
    wave2 = make_wave(freq2, TIMBRES[1])
    wave3 = make_wave(freq3, TIMBRES[2])
    wave4 = make_wave(freq4, TIMBRES[3])
    wave = (wave1 + wave2 + wave3 + wave4) / 4

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
                elif action['type'] == 'note':
                    with STATE_LOCK:
                        f1 = music_state.move_voice1(action['v1'], action['char'])
                        f2 = music_state.move_voice2(action['v2'], action['char'])
                        f3 = music_state.move_voice3(action['v3'], action['char'])
                        f4 = music_state.move_voice4(action['v4'], action['char'])
                        tone = generate_quad_tone(f1, f2, f3, f4, action['duration'])
                    stream.write(tone.tobytes())

                symbol = action['char']
                with STATE_LOCK:
                    n1 = music_state.get_note_name(music_state.voice1_index)
                    n2 = music_state.get_note_name(music_state.voice2_index)
                    n3 = music_state.get_note_name(music_state.voice3_index)
                    n4 = music_state.get_note_name(music_state.voice4_index)
                print(f"{symbol}\t{action['v1']}\t{action['v2']}\t{action['v3']}\t{action['v4']}")
                print(f"\t{n1}\t{n2}\t{n3}\t{n4}")

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


def on_press(key):
    """Handle key press events"""
    global last_key_time

    current_time = time.time()
    # Debounce
    if current_time - last_key_time < 0.05:
        return
    last_key_time = current_time

    try:
        if hasattr(key, 'char') and key.char:
            char = key.char.lower()
            action = get_character_action(char)

            if action:
                # Queue the action instead of playing directly
                audio_queue.put(action)

    except AttributeError:
        pass


def on_release(key):
    """Handle key release - exit on ESC"""
    global running

    if key == keyboard.Key.esc:
        print("\nShutting down...")
        running = False
        audio_queue.put(None)  # Signal audio thread to stop
        QCoreApplication.quit()
        return False


class MelotypeConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("melotype config")
        self.setMinimumWidth(900)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        tabs = QTabWidget()
        root_layout.addWidget(tabs)

        tabs.addTab(self._build_config_tab(), "Config")
        tabs.addTab(self._build_intervals_tab(), "Intervals")

        footer = QHBoxLayout()
        footer.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)
        root_layout.addLayout(footer)

        self._refresh_voice_note_dropdowns()

    def _build_config_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scale_box = QGroupBox("Scale")
        scale_layout = QGridLayout(scale_box)
        layout.addWidget(scale_box)

        scale_layout.addWidget(QLabel("Key"), 0, 0)
        self.key_combo = QComboBox()
        for label, pc in KEY_OPTIONS:
            self.key_combo.addItem(label, pc)
        for i in range(self.key_combo.count()):
            if self.key_combo.itemData(i) == music_state.key_pitch_class:
                self.key_combo.setCurrentIndex(i)
                break
        self.key_combo.currentIndexChanged.connect(self._on_key_changed)
        scale_layout.addWidget(self.key_combo, 0, 1)

        scale_layout.addWidget(QLabel("Tension (mode)"), 1, 0)
        self.tension_combo = QComboBox()
        for label in TENSION_OPTIONS:
            self.tension_combo.addItem(label, label)
        self.tension_combo.setCurrentText(music_state.mode_name)
        self.tension_combo.currentIndexChanged.connect(self._on_tension_changed)
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
        tone_layout = QGridLayout(tone_box)
        layout.addWidget(tone_box)

        self.timbre_combos: list[QComboBox] = []
        for i in range(4):
            tone_layout.addWidget(QLabel(f"Voice {i + 1} timbre"), i, 0)
            combo = QComboBox()
            for timbre in TIMBRE_OPTIONS:
                combo.addItem(timbre, timbre)
            combo.setCurrentText(TIMBRES[i])
            combo.currentIndexChanged.connect(lambda _idx, v=i: self._on_timbre_changed(v))
            self.timbre_combos.append(combo)
            tone_layout.addWidget(combo, i, 1)

        layout.addStretch(1)
        return tab

    def _build_intervals_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        contents = QWidget()
        scroll.setWidget(contents)
        layout = QVBoxLayout(contents)

        value_options = list(range(-8, 9))
        value_options.remove(0)

        self.interval_combos: dict[tuple[str, str, int], QComboBox] = {}

        for group_name, group in GROUPS.items():
            box = QGroupBox(group_name)
            layout.addWidget(box)
            grid = QGridLayout(box)

            grid.addWidget(QLabel(""), 0, 0)
            for col, letter in enumerate(group["letters"]):
                grid.addWidget(QLabel(letter), 0, col + 1, alignment=Qt.AlignCenter)

            for row, voice_key in enumerate(["v1", "v2", "v3", "v4"]):
                grid.addWidget(QLabel(voice_key.upper()), row + 1, 0)
                for col, _letter in enumerate(group["letters"]):
                    cb = QComboBox()
                    for v in value_options:
                        cb.addItem(str(v), v)
                    current = group[voice_key][col]
                    cb.setCurrentText(str(current))
                    cb.currentIndexChanged.connect(
                        lambda _idx, g=group_name, vk=voice_key, pos=col: self._on_interval_changed(g, vk, pos)
                    )
                    self.interval_combos[(group_name, voice_key, col)] = cb
                    grid.addWidget(cb, row + 1, col + 1)

        punct_box = QGroupBox("Punctuation")
        layout.addWidget(punct_box)
        punct_grid = QGridLayout(punct_box)
        punct_grid.addWidget(QLabel(""), 0, 0)
        punct_grid.addWidget(QLabel(","), 0, 1, alignment=Qt.AlignCenter)
        punct_grid.addWidget(QLabel("."), 0, 2, alignment=Qt.AlignCenter)

        self.punct_combos: dict[tuple[str, str], QComboBox] = {}
        for row, voice_key in enumerate(["v1", "v2", "v3", "v4"]):
            punct_grid.addWidget(QLabel(voice_key.upper()), row + 1, 0)
            for col, ch in enumerate([",", "."]):
                cb = QComboBox()
                for v in value_options:
                    cb.addItem(str(v), v)
                cb.setCurrentText(str(PUNCT_MOVES[voice_key][ch]))
                cb.currentIndexChanged.connect(lambda _idx, vk=voice_key, c=ch: self._on_punct_changed(vk, c))
                self.punct_combos[(voice_key, ch)] = cb
                punct_grid.addWidget(cb, row + 1, col + 1)

        layout.addStretch(1)
        return tab

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

    def _on_tension_changed(self, _idx: int):
        with STATE_LOCK:
            music_state.set_mode(str(self.tension_combo.currentData()))
        self._refresh_voice_note_dropdowns()

    def _on_interval_changed(self, group_name: str, voice_key: str, pos: int):
        cb = self.interval_combos[(group_name, voice_key, pos)]
        value = int(cb.currentData())
        with STATE_LOCK:
            GROUPS[group_name][voice_key][pos] = value
            init_letter_movements()

    def _on_punct_changed(self, voice_key: str, ch: str):
        cb = self.punct_combos[(voice_key, ch)]
        value = int(cb.currentData())
        with STATE_LOCK:
            PUNCT_MOVES[voice_key][ch] = value
            init_letter_movements()


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

    # Start audio thread
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    # Give audio thread time to initialize
    time.sleep(0.2)

    app = QApplication([])
    window = MelotypeConfigWindow()
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