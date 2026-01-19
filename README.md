## HarmoniKeys

**HarmoniKeys** turns typing into a live 4‑voice harmony generator with a **PySide6 GUI** for configuring **key + scale/mode**, **octave range**, **timbre per voice**, and **per‑key interval mappings**. Changes are **live** while the app is open: the **global keyboard listener** keeps running in the background, and each keystroke plays a 4‑note voicing based on your mappings.

Coded with human ideas, and input from my guitarscalechart / pymusicgen apps. 

Cursor did the heavy lifting of the app structure and adding features I asked for. 

## Features

- **Live “typing → harmony” engine**: global keyboard listener + dedicated audio thread.
- **Configurable key + mode**:
  - Key changes the **root pitch class** (tonic) used for the scale math.
  - Mode/scale uses **step-pattern definitions** (Crawsome guitarscalechart format), so 7‑note and pentatonic/blues scales are supported.
- **Configurable octave range**: 1–5 octaves.
- **Per‑voice start notes**: pick the starting scale note for each of the 4 voices (choices update with key/mode/octaves).
- **Configurable interval mappings (live)**:
  - **Letters** (`a-z`) use letter-frequency buckets (Top 8 / Next 8 / Next 6 / Rarest 4) and every cell is editable.
  - **Punctuation + symbols + special keys** are editable per voice (including SPACE/ENTER/TAB/BACKSPACE/DELETE and many symbols).
  - **Numbers** (`0–9`) drive **voices 3–4 only** (configurable per digit).
  - **Capital letters** (`A–Z`) drive **voices 1–2 only**.
- **Silent per-voice mapping**: every mapping dropdown supports **Silent** (that voice won’t play for that key) and supports step value **0**.
- **Punctuation defaults are frequency-tiered**: punctuation columns are sorted by English-ish usage frequency and prefilled so common symbols move less and rare symbols jump more (still fully editable).
- **Timbres (per voice)**: multiple built-in synth timbres (sine/triangle/square/saw/pulse + presets like organ/strings/brass/bell/FM EP).
- **Config persistence**:
  - Save/Load `config.ini` in the repo root.
  - Older config formats are auto-migrated on load and rewritten cleanly.
- **Quality-of-life UI**:
  - File menu: Save/Load config, Open project folder, Exit
  - Help → About

## How it works

- **Voices as scale cursors**: each voice tracks a “scale index” (degree within octave + octave offset).
- **Interval mapping**: each keypress resolves to per‑voice step deltas (your mappings), applies them to each voice, then converts resulting indices to frequencies.
- **Range behavior**: voices “bounce” at the min/max range by flipping that key’s movement direction for that voice.

## Requirements

- Python 3
- Packages:
  - `numpy`
  - `pyaudio`
  - `pynput`
  - `PySide6`
  - `music21` (for chord naming)

## Install (PowerShell)

```powershell
python -m pip install numpy pyaudio pynput PySide6 music21
```

If `pyaudio` fails to install on Windows, you typically need a working PortAudio/PyAudio wheel for your Python version.

## Run

```powershell
python .\main.py
```

Press keys while the app is running (global listener). Press **ESC** to quit.

## Controls

- **Letters** (`a-z`): configurable movements for all 4 voices.
- **Capital letters** (`A-Z`): **voices 1–2 only**.
- **Digits** (`0-9`): **voices 3–4 only**.
- **SPACE/ENTER/TAB/BACKSPACE/DELETE**: configurable (default = rest if all 4 voice values are 0).
- **ESC**: exit.

## GUI

- **Basic tab** (landing):
  - Preset (tone arrangement)
  - Key
  - Tension (mode)
- **Config tab**:
  - **Scale**: key, mode (“tension”), octaves, voice starting notes
  - **Tone**: timbre per voice
- **Intervals tab**:
  - Letter buckets
  - Punctuation + special keys (sorted by English-ish usage frequency; defaults are tiered so common symbols move less and rare symbols jump more)
  - Numbers

## Menus / actions

- **File**:
  - Save config.ini
  - Load config.ini
  - Open project folder
  - Exit
- **Help**:
  - About

## Config file

- `config.ini` is created/overwritten via **File → Save config.ini**
- Load via **File → Load config.ini**
- Punctuation is stored as `v1..v4` CSV lists aligned to the current punctuation column order shown in the UI.
- Numbers are stored as `v1..v2` CSV lists aligned to digits `0..9`.
- Any **Silent** entries are saved as the token `S` in these CSV lists.
- Older `config.ini` formats are auto-migrated on load and then rewritten to the new format.

## Console output (matrix)

On startup, HarmoniKeys prints the current notes:

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

## About

Made by `github.com/crawsome`.

- Main profile: `https://github.com/crawsome`
- Main project: **Wyrm Warrior**: `https://gx.games/games/z0d76p/wyrm-warrior/`

