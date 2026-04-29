import numpy as np

def load_stt(self):
    self._log("stt", "Loading VOSK STT engine…", "dim")
    self._stage("stt", "active")
    try:
        import os
        import vosk
    except ImportError:
        raise RuntimeError("vosk not installed.\nRun: pip install vosk")

    model_path = _get_vosk_model_path(self)
    if not model_path or not os.path.isdir(model_path):
        _download_vosk_model(self, model_path)

    self._vosk_model = vosk.Model(model_path)
    self._recognizer = vosk.KaldiRecognizer(self._vosk_model, self.SAMPLE_RATE)
    self._log("stt", f"VOSK loaded ✓ ({os.path.basename(model_path)})", "green")
    self._stage("stt", "done")


def _get_vosk_model_path(self):
    import os
    base_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    if self.src_lang != "en":
        self._log("stt", "VOSK only supports English in this setup; using English model.", "warn")
    return os.path.join(base_dir, "vosk-model-small-en-us-0.15")


def _download_vosk_model(self, model_path: str):
    import os
    import urllib.request
    import zipfile

    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    archive = model_path + ".zip"

    self._log("stt", f"Downloading VOSK model from {url}…", "dim")
    urllib.request.urlretrieve(url, archive)

    self._log("stt", "Extracting VOSK model…", "dim")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(os.path.dirname(model_path))

    os.unlink(archive)
    self._log("stt", "VOSK model ready ✓", "green")


def transcribe(self, audio):
    if self._recognizer is None:
        return ""
    try:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        pcm = (audio * 32767.0).astype(np.int16).tobytes()
        accepted = self._recognizer.AcceptWaveform(pcm)
        if not accepted:
            return ""
        import json
        return json.loads(self._recognizer.Result()).get("text", "").strip()
    except Exception as e:
        self._log("stt", f"Transcription error: {e}", "err")
        return ""
