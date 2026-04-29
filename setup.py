#!/usr/bin/env python3
"""
VoiceTrans - Dependency Installer
Run this ONCE before launching the app.
"""

import subprocess
import sys
import os


PACKAGES = [
    ("sounddevice",       "sounddevice"),
    ("argostranslate",    "argostranslate"),
    ("pyttsx3",           "pyttsx3"),
    ("edge-tts",          "edge_tts"),
    ("imageio-ffmpeg",    "imageio_ffmpeg"),
    ("vosk",              "vosk"),
    ("numpy",             "numpy"),
]

OPTIONAL = [
    ("pyaudio",           "pyaudio",        "Needed on some systems for audio"),
]


def pip_install(pkg_name: str):
    print(f"  Installing {pkg_name}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", pkg_name],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ✗ Failed: {result.stderr.strip()}")
        return False
    print(f"  ✓ {pkg_name}")
    return True


def check_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def main():
    print("=" * 56)
    print("  VoiceTrans — Dependency Setup")
    print("=" * 56)
    print()

    missing = []
    for pkg, mod in PACKAGES:
        if check_import(mod):
            print(f"  ✓ {pkg} already installed")
        else:
            missing.append(pkg)

    if missing:
        print(f"\nInstalling {len(missing)} missing package(s)...\n")
        for pkg in missing:
            pip_install(pkg)
    else:
        print("\nAll core packages already installed!")

    print("\nOptional packages:")
    for pkg, mod, note in OPTIONAL:
        if check_import(mod):
            print(f"  ✓ {pkg}")
        else:
            print(f"  - {pkg}: {note}")
            ans = input(f"    Install {pkg}? [y/N]: ").strip().lower()
            if ans == "y":
                pip_install(pkg)

    print()
    print("─" * 56)
    print("  VIRTUAL MIC SETUP (required for in-game output)")
    print("─" * 56)
    print("""
  1. Download VB-Audio Virtual Cable (free):
     https://vb-audio.com/Cable/

  2. Install it — it creates:
       • CABLE Input  (a virtual speaker)
       • CABLE Output (a virtual mic)

  3. In VoiceTrans:
       Input device  → your real mic or game voice channel
       Output device → CABLE Input (VB-Audio Virtual Cable)

  4. In your game / Discord:
       Microphone → CABLE Output (VB-Audio Virtual Cable)

  Now the game hears the translated TTS voice instead
  of your real mic.
""")
    print("─" * 56)

    print("\nLanguage pack download (Argos Translate):")
    print("  Language packs are downloaded automatically when you")
    print("  first start the pipeline. This is a one-time download.")
    print()

    print("Setup complete! Run:  python main.py")
    print()


if __name__ == "__main__":
    main()
