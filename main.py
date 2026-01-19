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

# Audio queue - thread-safe communication
audio_queue = queue.Queue()
running = True


class MusicState:
    """Maintains current musical state"""

    def __init__(self):
        self.melody_note_index = 7  # Start in middle of range
        self.harmony_note_index = 7  # Start at same position
        self.base_freq = 110.0  # A2

    def get_frequency(self, scale_index):
        """Convert scale index to frequency"""
        # Map index to pentatonic note (A C D E G pattern)
        octave = scale_index // 5
        position = scale_index % 5
        semitones = octave * 12 + [0, 3, 5, 7, 10][position]
        return self.base_freq * (2 ** (semitones / 12))

    def move_melody(self, steps, char):
        """Move melody by steps, invert letter at boundaries"""
        new_index = self.melody_note_index + steps

        # Hit boundary? Invert and bounce back
        if new_index > 14:
            melody_movements[char] = -melody_movements[char]
            self.melody_note_index = 14 - (new_index - 14)
            self.melody_note_index = max(0, self.melody_note_index)
        elif new_index < 0:
            melody_movements[char] = -melody_movements[char]
            self.melody_note_index = abs(new_index)
            self.melody_note_index = min(14, self.melody_note_index)
        else:
            self.melody_note_index = new_index

        return self.get_frequency(self.melody_note_index)

    def move_harmony(self, steps, char):
        """Move harmony with its own rules"""
        new_index = self.harmony_note_index + steps

        # Hit boundary? Invert and bounce back
        if new_index > 14:
            harmony_movements[char] = -harmony_movements[char]
            self.harmony_note_index = 14 - (new_index - 14)
            self.harmony_note_index = max(0, self.harmony_note_index)
        elif new_index < 0:
            harmony_movements[char] = -harmony_movements[char]
            self.harmony_note_index = abs(new_index)
            self.harmony_note_index = min(14, self.harmony_note_index)
        else:
            self.harmony_note_index = new_index

        return self.get_frequency(self.harmony_note_index)


# Global music state
music_state = MusicState()

# Letter movement mappings - SEPARATE for melody and harmony
melody_movements = {}
harmony_movements = {}


def init_letter_movements():
    """Initialize letter movement mappings"""
    for idx, char in enumerate(LETTER_FREQ):
        # Top 8 letters: small movements (NO ZEROS)
        if idx < 8:
            melody_moves = [1, 1, -1, 2, -2, -1, 1, -1]
            harmony_moves = [-1, -1, 1, -2, 2, 1, -1, 1]  # Opposite tendencies
            melody_movements[char] = melody_moves[idx]
            harmony_movements[char] = harmony_moves[idx]
        # Next 8: medium movements
        elif idx < 16:
            melody_moves = [2, -2, 1, -1, 2, 1, -2, -3]
            harmony_moves = [-2, 2, -1, 1, -2, -1, 2, 3]  # Opposite tendencies
            melody_movements[char] = melody_moves[idx - 8]
            harmony_movements[char] = harmony_moves[idx - 8]
        # Next 6: larger jumps
        elif idx < 22:
            melody_moves = [3, -3, 2, -2, 3, -3]
            harmony_moves = [-3, 3, -2, 2, -3, 3]  # Opposite tendencies
            melody_movements[char] = melody_moves[idx - 16]
            harmony_movements[char] = harmony_moves[idx - 16]
        # Rarest 4: dramatic leaps
        else:
            melody_moves = [5, -5, 4, -4]
            harmony_moves = [-5, 5, -4, 4]  # Opposite tendencies
            melody_movements[char] = melody_moves[idx - 22]
            harmony_movements[char] = harmony_moves[idx - 22]

    melody_movements[','] = -1
    melody_movements['.'] = -2
    harmony_movements[','] = 1
    harmony_movements['.'] = 2


# Initialize on load
init_letter_movements()


def get_character_action(char):
    """Map characters to musical actions"""

    if char == ' ':
        return {'type': 'rest', 'duration': 0.15}

    if char not in melody_movements:
        return None

    melody_steps = melody_movements[char]
    harmony_steps = harmony_movements[char]

    return {'type': 'note', 'melody_steps': melody_steps, 'harmony_steps': harmony_steps, 'duration': 0.15,
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


def generate_dual_tone(freq1, freq2, duration=0.15, volume=0.15):
    """Generate two-note harmony"""
    sample_rate = 44100
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)

    # Two sine waves with harmonics
    wave1 = np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(4 * np.pi * freq1 * t)
    wave2 = np.sin(2 * np.pi * freq2 * t) + 0.3 * np.sin(4 * np.pi * freq2 * t)
    wave = (wave1 + wave2) / 2

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
                    freq_melody = music_state.move_melody(action['melody_steps'], action['char'])
                    freq_harmony = music_state.move_harmony(action['harmony_steps'], action['char'])
                    tone = generate_dual_tone(freq_melody, freq_harmony, action['duration'])
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
                freq_m = music_state.get_frequency(music_state.melody_note_index)
                freq_h = music_state.get_frequency(music_state.harmony_note_index)
                print(f"{symbol} -> M:{freq_m:.1f}Hz H:{freq_h:.1f}Hz")

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
    print("Typing Music Generator - Relative Pitch Navigation")
    print("=" * 60)
    print("\nMusical mapping:")
    print("  Common letters (etaoinsh): small melodic steps")
    print("  Medium letters (rdlcumfp): larger intervals, varied rhythm")
    print("  Rare letters (gwybvk): big jumps")
    print("  Rarest (xjqz): dramatic leaps")
    print("  SPACE: rest/silence")
    print("  COMMA: quick down step")
    print("  PERIOD: longer note, resolves down")
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