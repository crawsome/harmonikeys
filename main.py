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
        self.current_note_index = 7  # Start in middle of range
        self.base_freq = 110.0  # A2

    def get_frequency(self, scale_index):
        """Convert scale index to frequency"""
        # Map index to pentatonic note (A C D E G pattern)
        octave = scale_index // 5
        position = scale_index % 5
        semitones = octave * 12 + [0, 3, 5, 7, 10][position]
        return self.base_freq * (2 ** (semitones / 12))

    def move_relative(self, steps):
        """Move by steps in the pentatonic scale"""
        # Limit to 15 notes = exactly 3 octaves of pentatonic (5 notes per octave)
        self.current_note_index = max(0, min(14, self.current_note_index + steps))
        return self.get_frequency(self.current_note_index)


# Global music state
music_state = MusicState()


def get_character_action(char):
    """Map characters to musical actions"""

    if char == ' ':
        return {'type': 'rest', 'duration': 0.15}
    elif char == ',':
        return {'type': 'note', 'steps': -1, 'duration': 0.1}
    elif char == '.':
        return {'type': 'note', 'steps': -2, 'duration': 0.4}

    if char not in LETTER_FREQ:
        return None

    idx = LETTER_FREQ.index(char)

    # Top 8 letters: small movements, medium duration
    if idx < 8:
        movements = [0, 1, -1, 1, 0, -1, 1, -1]
        return {'type': 'note', 'steps': movements[idx], 'duration': 0.15}

    # Next 8: medium movements, varied duration
    elif idx < 16:
        movements = [2, -2, 1, -1, 2, 1, -2, 0]
        durations = [0.12, 0.18, 0.12, 0.15, 0.1, 0.15, 0.2, 0.25]
        return {'type': 'note', 'steps': movements[idx - 8], 'duration': durations[idx - 8]}

    # Next 6: larger jumps
    elif idx < 22:
        movements = [3, -3, 2, -2, 3, -3]
        return {'type': 'note', 'steps': movements[idx - 16], 'duration': 0.12}

    # Rarest 4: dramatic leaps
    else:
        movements = [5, -5, 4, -4]
        return {'type': 'note', 'steps': movements[idx - 22], 'duration': 0.1}


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
                    freq = music_state.move_relative(action['steps'])
                    tone = generate_tone(freq, action['duration'])
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
                freq = music_state.get_frequency(music_state.current_note_index)
                print(f"{symbol} -> {freq:.1f}Hz (pos:{music_state.current_note_index})")

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