# VoiceTrans 🎮🌍

Real-time game voice translator. Captures voice chat, translates it, and
outputs translated speech through a virtual microphone — 100% free, 100% local.

---

## Stack (all free, all local)

| Layer       | Tool             | Why                          |
|-------------|------------------|------------------------------|
| STT         | VOSK             | Offline English STT          |
| Translation | Argos Translate  | Offline, no API key          |
| TTS         | pyttsx3 / edge-tts | Local synthetic speech     |
| Audio       | sounddevice      | Cross-platform audio I/O     |
| Virtual Mic | VB-Audio Cable   | Routes TTS to game mic input |
| GUI         | tkinter          | Built into Python            |

---

## Quick Start

### 1. Install Python dependencies

```bash
python setup.py
```

Or manually:

```bash
pip install sounddevice argostranslate pyttsx3 numpy vosk edge-tts imageio-ffmpeg
```

### 2. Install VB-Audio Virtual Cable (Windows)

Download from: https://vb-audio.com/Cable/

This is FREE. It installs two virtual devices:
- **CABLE Input** — a virtual speaker (your app writes here)
- **CABLE Output** — a virtual mic (game reads from here)

On Linux use `PulseAudio` virtual sinks instead.

### 3. Configure audio routing

```
Your real mic / game voice
        ↓
  VoiceTrans (input device = your real mic or VB-Cable)
        ↓
  VOSK → Argos → TTS
        ↓
  Output device = CABLE Input (VB-Audio Virtual Cable)
        ↓
  In-game mic = CABLE Output (VB-Audio Virtual Cable)
```

### 4. Launch

```bash
python main.py
```

---

## Expected Latency

| Stage       | Time       |
|-------------|------------|
| STT         | 300–600 ms |
| Translation | 50–200 ms  |
| TTS         | 100–300 ms |
| **Total**   | **~0.6–1.1s** |

Good enough for gaming.

---

## Supported Language Pairs (Argos Translate)

en ↔ hi, ml, ta, te, de, fr, es, ru, zh, ja, ko, ar, pt

Language packs download automatically on first use (~50MB each).

---

## Files

```
voicetrans/
├── main.py       ← GUI + app entry point
├── pipeline/      ← audio processing package
├── setup.py      ← dependency installer
├── config.json   ← auto-generated settings
└── README.md
```

---

## Troubleshooting

**"No module named vosk"**
→ `pip install vosk`

**"No module named argostranslate"**
→ `pip install argostranslate`

**Audio not playing to virtual cable**
→ Make sure VB-Audio Cable is installed, then Refresh Devices in the app.

**Game not hearing translated voice**
→ In game settings, set microphone to "CABLE Output (VB-Audio Virtual Cable)"

**High latency**
→ Reduce chunk size to 0.5s for faster response.

**Translation not working**
→ First-time use downloads language pack (~50MB). Wait for it to finish.

---

## Linux Virtual Mic Setup

```bash
# Create a virtual sink
pactl load-module module-null-sink sink_name=voicetrans_out sink_properties=device.description=VoiceTrans

# Use "Monitor of VoiceTrans" as mic source in games
```

---

## Credits

- VOSK by Alphacephei
- Argos Translate by LibreTranslate
- VB-Audio Virtual Cable by Vincent Burel
