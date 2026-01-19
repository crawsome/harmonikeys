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

# Letter frequencies (descending order)
LETTER_FREQ = "etaoinshrdlcumfpgwybvkxjqz"

# 7-note Dorian scale over 3 octaves: indices 0..21 (roots at 0, 7, 14, 21)
SCALE_DEGREES = 7
OCTAVES = 3
MIN_SCALE_INDEX = 0
MAX_SCALE_INDEX = OCTAVES * SCALE_DEGREES

# Audio queue - thread-safe communication
audio_queue = queue.Queue()
running = True


class MusicState:
    """Maintains current musical state"""

    def __init__(self):
        # 4 voices, each starting on the root note in a different octave
        # 3 octaves total range: indices 0..21 (roots at 0, 7, 14, 21)
        self.voice1_index = 0
        self.voice2_index = 7
        self.voice3_index = 14
        self.voice4_index = 21
        self.base_freq = 110.0  # A2
        self.base_midi = 45  # A2 (matches base_freq)

    def get_frequency(self, scale_index):
        """Convert scale index to frequency using Dorian mode"""
        # Dorian mode intervals: 0, 2, 3, 5, 7, 9, 10 (W-H-W-W-W-H-W)
        octave = scale_index // 7
        position = scale_index % 7
        semitones = octave * 12 + [0, 2, 3, 5, 7, 9, 10][position]
        return self.base_freq * (2 ** (semitones / 12))

    def get_note_name(self, scale_index):
        """Convert scale index to note name (with octave)"""
        octave = scale_index // 7
        position = scale_index % 7
        semitones = octave * 12 + [0, 2, 3, 5, 7, 9, 10][position]

        midi = self.base_midi + semitones
        note = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][midi % 12]
        note_octave = (midi // 12) - 1
        return f"{note}{note_octave}"

    def move_voice1(self, steps, char):
        """Move voice 1"""
        new_index = self.voice1_index + steps

        if new_index > MAX_SCALE_INDEX:
            voice1_movements[char] = -voice1_movements[char]
            overflow = new_index - MAX_SCALE_INDEX
            reflected = MAX_SCALE_INDEX - overflow
            if reflected == self.voice1_index:
                reflected -= 1
            self.voice1_index = reflected
        elif new_index < MIN_SCALE_INDEX:
            voice1_movements[char] = -voice1_movements[char]
            reflected = MIN_SCALE_INDEX + (MIN_SCALE_INDEX - new_index)
            if reflected == self.voice1_index:
                reflected += 1
            self.voice1_index = reflected
        else:
            self.voice1_index = new_index

        return self.get_frequency(self.voice1_index)

    def move_voice2(self, steps, char):
        """Move voice 2"""
        new_index = self.voice2_index + steps

        if new_index > MAX_SCALE_INDEX:
            voice2_movements[char] = -voice2_movements[char]
            overflow = new_index - MAX_SCALE_INDEX
            reflected = MAX_SCALE_INDEX - overflow
            if reflected == self.voice2_index:
                reflected -= 1
            self.voice2_index = reflected
        elif new_index < MIN_SCALE_INDEX:
            voice2_movements[char] = -voice2_movements[char]
            reflected = MIN_SCALE_INDEX + (MIN_SCALE_INDEX - new_index)
            if reflected == self.voice2_index:
                reflected += 1
            self.voice2_index = reflected
        else:
            self.voice2_index = new_index

        return self.get_frequency(self.voice2_index)

    def move_voice3(self, steps, char):
        """Move voice 3"""
        new_index = self.voice3_index + steps

        if new_index > MAX_SCALE_INDEX:
            voice3_movements[char] = -voice3_movements[char]
            overflow = new_index - MAX_SCALE_INDEX
            reflected = MAX_SCALE_INDEX - overflow
            if reflected == self.voice3_index:
                reflected -= 1
            self.voice3_index = reflected
        elif new_index < MIN_SCALE_INDEX:
            voice3_movements[char] = -voice3_movements[char]
            reflected = MIN_SCALE_INDEX + (MIN_SCALE_INDEX - new_index)
            if reflected == self.voice3_index:
                reflected += 1
            self.voice3_index = reflected
        else:
            self.voice3_index = new_index

        return self.get_frequency(self.voice3_index)

    def move_voice4(self, steps, char):
        """Move voice 4"""
        new_index = self.voice4_index + steps

        if new_index > MAX_SCALE_INDEX:
            voice4_movements[char] = -voice4_movements[char]
            overflow = new_index - MAX_SCALE_INDEX
            reflected = MAX_SCALE_INDEX - overflow
            if reflected == self.voice4_index:
                reflected -= 1
            self.voice4_index = reflected
        elif new_index < MIN_SCALE_INDEX:
            voice4_movements[char] = -voice4_movements[char]
            reflected = MIN_SCALE_INDEX + (MIN_SCALE_INDEX - new_index)
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


def init_letter_movements():
    """Initialize letter movement mappings for 4 voices"""
    for idx, char in enumerate(LETTER_FREQ):
        # Top 8 letters: small movements
        if idx < 8:
            v1_moves = [1, 1, -1, 2, -2, -1, 1, -1]
            v2_moves = [-1, -1, 1, -2, 2, 1, -1, 1]
            v3_moves = [1, -1, 2, -1, 1, -2, 1, -1]
            v4_moves = [-1, 1, -2, 1, -1, 2, -1, 1]
            voice1_movements[char] = v1_moves[idx]
            voice2_movements[char] = v2_moves[idx]
            voice3_movements[char] = v3_moves[idx]
            voice4_movements[char] = v4_moves[idx]
        # Next 8: medium movements
        elif idx < 16:
            v1_moves = [2, -2, 1, -1, 2, 1, -2, -3]
            v2_moves = [-2, 2, -1, 1, -2, -1, 2, 3]
            v3_moves = [2, -1, 2, -2, 1, -2, 2, -3]
            v4_moves = [-2, 1, -2, 2, -1, 2, -2, 3]
            voice1_movements[char] = v1_moves[idx - 8]
            voice2_movements[char] = v2_moves[idx - 8]
            voice3_movements[char] = v3_moves[idx - 8]
            voice4_movements[char] = v4_moves[idx - 8]
        # Next 6: larger jumps
        elif idx < 22:
            v1_moves = [3, -3, 2, -2, 3, -3]
            v2_moves = [-3, 3, -2, 2, -3, 3]
            v3_moves = [3, -2, 3, -3, 2, -3]
            v4_moves = [-3, 2, -3, 3, -2, 3]
            voice1_movements[char] = v1_moves[idx - 16]
            voice2_movements[char] = v2_moves[idx - 16]
            voice3_movements[char] = v3_moves[idx - 16]
            voice4_movements[char] = v4_moves[idx - 16]
        # Rarest 4: dramatic leaps
        else:
            v1_moves = [5, -5, 4, -4]
            v2_moves = [-5, 5, -4, 4]
            v3_moves = [4, -5, 5, -4]
            v4_moves = [-4, 5, -5, 4]
            voice1_movements[char] = v1_moves[idx - 22]
            voice2_movements[char] = v2_moves[idx - 22]
            voice3_movements[char] = v3_moves[idx - 22]
            voice4_movements[char] = v4_moves[idx - 22]

    voice1_movements[','] = -1
    voice1_movements['.'] = -2
    voice2_movements[','] = 1
    voice2_movements['.'] = 2
    voice3_movements[','] = -1
    voice3_movements['.'] = 1
    voice4_movements[','] = 2
    voice4_movements['.'] = -1


# Initialize on load
init_letter_movements()


def get_character_action(char):
    """Map characters to musical actions"""

    if char == ' ':
        return {'type': 'rest', 'duration': 0.15}

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

    # Four sine waves with harmonics
    wave1 = np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(4 * np.pi * freq1 * t)
    wave2 = np.sin(2 * np.pi * freq2 * t) + 0.3 * np.sin(4 * np.pi * freq2 * t)
    wave3 = np.sin(2 * np.pi * freq3 * t) + 0.3 * np.sin(4 * np.pi * freq3 * t)
    wave4 = np.sin(2 * np.pi * freq4 * t) + 0.3 * np.sin(4 * np.pi * freq4 * t)
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
                    f1 = music_state.move_voice1(action['v1'], action['char'])
                    f2 = music_state.move_voice2(action['v2'], action['char'])
                    f3 = music_state.move_voice3(action['v3'], action['char'])
                    f4 = music_state.move_voice4(action['v4'], action['char'])
                    tone = generate_quad_tone(f1, f2, f3, f4, action['duration'])
                    stream.write(tone.tobytes())

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

                symbol = '·' if action['type'] == 'rest' else char
                n1 = music_state.get_note_name(music_state.voice1_index)
                n2 = music_state.get_note_name(music_state.voice2_index)
                n3 = music_state.get_note_name(music_state.voice3_index)
                n4 = music_state.get_note_name(music_state.voice4_index)
                print(f"{symbol} -> V1:{n1} V2:{n2} V3:{n3} V4:{n4}")

    except AttributeError:
        pass


def on_release(key):
    """Handle key release - exit on ESC"""
    global running

    if key == keyboard.Key.esc:
        print("\nShutting down...")
        running = False
        audio_queue.put(None)  # Signal audio thread to stop
        return False


def main():
    global running

    print("=" * 60)
    print("Typing Music Generator - 4-Part Dorian Harmony")
    print("=" * 60)
    print("\n4 voices, each starting in different octaves")
    print("Dorian mode (D Dorian: D E F G A B C)")
    print("Range: 3 octaves")
    print("\nPress ESC to exit")
    print("=" * 60)
    print()

    # Start audio thread
    audio_thread = threading.Thread(target=audio_worker, daemon=True)
    audio_thread.start()

    # Give audio thread time to initialize
    time.sleep(0.2)

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nInterrupted...")
    finally:
        running = False
        audio_queue.put(None)
        audio_thread.join(timeout=2)
        print("Goodbye!")


if __name__ == "__main__":
    main()