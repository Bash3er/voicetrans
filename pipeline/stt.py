import os
import sys
import numpy as np
from pathlib import Path

def load_stt(self):
    self._log("stt", "Loading VOSK STT engine…", "dim")
    self._stage("stt", "active")
    _register_vosk_dll_path()
    try:
        import vosk
    except ImportError:
        raise RuntimeError("vosk not installed.\nRun: pip install vosk")

    _ensure_vosk_dll_path(vosk)
    model_path = _get_vosk_model_path(self)
    if not model_path or not os.path.isdir(model_path):
        _download_vosk_model(self, model_path)

    self._vosk_model = vosk.Model(model_path)
    self._recognizer = vosk.KaldiRecognizer(self._vosk_model, self.SAMPLE_RATE)
    self._log("stt", f"VOSK loaded ✓ ({os.path.basename(model_path)})", "green")
    self._stage("stt", "done")


def _reset_recognizer(self):
    import vosk
    if self._vosk_model is None:
        return
    self._recognizer = vosk.KaldiRecognizer(self._vosk_model, self.SAMPLE_RATE)


def _register_vosk_dll_path():
    if os.name != "nt":
        return
    candidates = []
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry) / "vosk"
        if candidate.is_dir():
            candidates.append(candidate)
        elif Path(entry).is_dir() and (Path(entry) / "libvosk.dll").exists():
            candidates.append(Path(entry))
    for candidate in candidates:
        if (candidate / "libvosk.dll").exists():
            try:
                os.add_dll_directory(str(candidate))
                return
            except Exception:
                pass


def _ensure_vosk_dll_path(vosk_module):
    if os.name != "nt":
        return
    vosk_root = Path(vosk_module.__file__).resolve().parent
    if (vosk_root / "libvosk.dll").exists():
        try:
            os.add_dll_directory(str(vosk_root))
        except Exception:
            pass


def _get_vosk_model_path(self):
    app_model_dir = None
    if getattr(sys, "_MEIPASS", None):
        app_model_dir = Path(os.getenv("APPDATA") or Path.home() / "VoiceTrans") / "model"
    else:
        app_model_dir = Path(__file__).resolve().parent.parent / "model"

    app_model_dir.mkdir(parents=True, exist_ok=True)
    if self.src_lang != "en":
        self._log("stt", "VOSK only supports English in this setup; using English model.", "warn")
    return str(app_model_dir / "vosk-model-small-en-us-0.15")


def _download_vosk_model(self, model_path: str):
    import urllib.request
    import zipfile

    model_dir = os.path.dirname(model_path)
    archive = model_path + ".zip"

    self._log("stt", f"Downloading VOSK model from https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip…", "dim")
    urllib.request.urlretrieve("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip", archive)

    self._log("stt", "Extracting VOSK model…", "dim")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(model_dir)

    os.unlink(archive)
    self._log("stt", "VOSK model ready ✓", "green")


def transcribe(self, audio):
    if self._recognizer is None:
        return ""
    try:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if np.max(np.abs(audio)) < 1e-4:
            import json
            result = json.loads(self._recognizer.FinalResult()).get("text", "").strip()
            _reset_recognizer(self)
            return result

        pcm = (audio * 32767.0).astype(np.int16).tobytes()
        accepted = self._recognizer.AcceptWaveform(pcm)
        if accepted:
            import json
            return json.loads(self._recognizer.Result()).get("text", "").strip()
        return ""
    except Exception as e:
        self._log("stt", f"Transcription error: {e}", "err")
        return ""
