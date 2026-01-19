## Melotype

**Melotype** is a small Python script that turns typing into a 4‑voice harmony generator. Each keystroke maps to a set of **scale-step intervals** (one per voice) in a **3‑octave Dorian scale**, plays the resulting chord, and prints a tab-aligned “matrix” of the change.

## How it works

- **4 voices**: maintained as scale indices (`voice1_index` … `voice4_index`) spaced by octaves.
- **Scale**: 7-note Dorian mode over 3 octaves (scale indices `0..21`).
- **Keystrokes → intervals**: letters are ranked by frequency (`etaoin...`) and mapped to different step sizes per voice.
- **Audio**: generated with `numpy` and played via `pyaudio`.
- **Printing**: the audio thread prints after applying the interval(s), so the printed notes match what you hear.

## Requirements

- Python 3
- Packages:
  - `numpy`
  - `pyaudio`
  - `pynput`

## Install (PowerShell)

```powershell
python -m pip install numpy pyaudio pynput
```

If `pyaudio` fails to install on Windows, you typically need a working PortAudio/PyAudio wheel for your Python version.

## Run

```powershell
python .\main.py
```

Press keys while the script is focused/running. Press **ESC** to quit.

## Controls

- **Letters** (`a-z`): trigger different interval patterns per voice.
- **Space**: rest (no movement; printed as `·`).
- **`,` and `.`**: small fixed movements (see `init_letter_movements()` in `main.py`).
- **ESC**: exit.

## Console output (matrix)

On startup, Melotype prints the current notes:

```text
    <V1 note>    <V2 note>    <V3 note>    <V4 note>
```

Then for each keystroke it prints two lines, tab-separated:

```text
<key>    <v1 step>    <v2 step>    <v3 step>    <v4 step>
         <V1 note>    <V2 note>    <V3 note>    <V4 note>
```

- The **top line** is the “midline”: the typed key and the **interval integers** that produced the next notes.
- The **bottom line** is the resulting **note names (with octave)** for each voice.

