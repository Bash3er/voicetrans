import os
import tempfile
import subprocess
import asyncio
import numpy as np
from .utils import _resample


def load_tts(self):
    self._log("tts", "Loading TTS engine…", "dim")
    self._stage("tts", "active")
    self._tts = None
    self._tts_voice = None

    try:
        import edge_tts
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_path = get_ffmpeg_exe()
        if not ffmpeg_path or not os.path.exists(ffmpeg_path):
            raise RuntimeError("ffmpeg executable not found")
        self._tts = "edge-tts"
        self._tts_voice = _select_edge_voice(self)
        self._log("tts", f"edge-tts ready ✓  voice={self._tts_voice}", "green")
    except Exception as e:
        self._log("tts", f"edge-tts unavailable: {e}", "warn")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            picked = None
            for v in (voices or []):
                lid = getattr(v, "id", "").lower()
                ln = getattr(v, "name", "").lower()
                if self.tgt_lang == "en" and ("en" in lid or "english" in ln):
                    picked = v
                    break
            self._tts_voice = picked.id if picked else engine.getProperty("voice")
            engine.stop()
            self._tts = "pyttsx3"
            self._log("tts", f"pyttsx3 ready ✓  voice={self._tts_voice}", "green")
        except Exception as e2:
            self._log("tts", f"pyttsx3 unavailable: {e2} — TTS disabled", "warn")
    self._stage("tts", "done")


def tts_generate(self, text: str):
    if self._tts is None:
        return None
    if self._tts == "edge-tts":
        return _edge_tts_to_numpy(self, text)
    if self._tts == "pyttsx3":
        return _pyttsx3_to_numpy(self, text)
    return None


def _pyttsx3_to_numpy(self, text: str):
    import pyttsx3
    import wave
    if not text or not text.strip():
        return None
    engine = pyttsx3.init()
    tmp_path = None
    try:
        engine.setProperty("rate", 165)
        if self._tts_voice:
            try:
                engine.setProperty("voice", self._tts_voice)
            except Exception:
                pass
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        engine.stop()
        if not tmp_path or not os.path.exists(tmp_path):
            return None
        with wave.open(tmp_path, "rb") as wf:
            nframes = wf.getnframes()
            if nframes == 0:
                return None
            frames = wf.readframes(nframes)
            sr = wf.getframerate()
            sampwidth = wf.getsampwidth()
        dtype_map = {1: (np.uint8, 128.0, 128.0), 2: (np.int16, 0.0, 32768.0), 4: (np.int32, 0.0, 2147483648.0)}
        dt, offset, scale = dtype_map.get(sampwidth, (np.int16, 0.0, 32768.0))
        arr = (np.frombuffer(frames, dtype=dt).astype(np.float32) - offset) / scale
        if arr.size == 0:
            return None
        return _resample(arr, sr, self.SAMPLE_RATE) if sr != self.SAMPLE_RATE else arr
    except Exception as e:
        self._log("tts", f"pyttsx3 error: {e}", "err")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _select_edge_voice(self) -> str:
    mapping = {
        "en": "en-US-GuyNeural",
        "hi": "hi-IN-SwaraNeural",
        "ml": "en-IN-MadhurNeural",
        "ta": "ta-IN-ValluvarNeural",
        "te": "te-IN-SwarachitraNeural",
        "de": "de-DE-KatjaNeural",
        "fr": "fr-FR-DeniseNeural",
        "es": "es-ES-ElviraNeural",
        "ru": "ru-RU-DariyaNeural",
        "zh": "zh-CN-XiaoxiaoNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
        "ar": "ar-EG-SalmaNeural",
        "pt": "pt-BR-FranciscaNeural",
    }
    return mapping.get(self.tgt_lang, "en-US-GuyNeural")


def _edge_tts_to_numpy(self, text: str):
    if not text or not text.strip():
        return None
    try:
        import edge_tts
    except ImportError:
        self._log("tts", "edge-tts import failed", "err")
        return None
    tmp_path = None
    try:
        tmp_path = tempfile.mktemp(suffix=".mp3")
        voice = self._tts_voice or _select_edge_voice(self)

        async def save_audio():
            await edge_tts.Communicate(text, voice=voice).save(tmp_path)

        asyncio.run(save_audio())
        if not tmp_path or not os.path.exists(tmp_path):
            self._log("tts", "edge-tts output file missing", "warn")
            return None
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg_path = get_ffmpeg_exe()
        if not ffmpeg_path:
            self._log("tts", "ffmpeg not found", "err")
            return None
        proc = subprocess.run(
            [ffmpeg_path, "-y", "-loglevel", "error", "-i", tmp_path,
             "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1",
             "-ar", str(self.SAMPLE_RATE), "pipe:1"],
            capture_output=True,
        )
        if proc.returncode != 0:
            self._log("tts", f"ffmpeg error: {proc.stderr.decode(errors='ignore').strip()}", "err")
            return None
        raw = proc.stdout
        if not raw:
            self._log("tts", "ffmpeg returned empty audio", "warn")
            return None
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples if samples.size else None
    except Exception as e:
        self._log("tts", f"edge-tts error: {e}", "err")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def play_audio(self, audio):
    if audio is None or audio.size == 0:
        self._log("output", "No audio data to play", "warn")
        return
    try:
        import sounddevice as sd
        out_sr = self.SAMPLE_RATE
        try:
            info = sd.query_devices(self.output_device, kind="output")
            out_sr = int(info.get("default_samplerate", self.SAMPLE_RATE))
        except Exception as e:
            self._log("output", f"Output device query failed: {e}", "warn")
            out_sr = self.SAMPLE_RATE
        if out_sr != self.SAMPLE_RATE:
            audio = _resample(audio, self.SAMPLE_RATE, out_sr)
        if audio.size == 0:
            self._log("output", "Audio empty after resample", "warn")
            return
        sd.play(audio, samplerate=out_sr, device=self.output_device, blocking=False)
        sd.wait()
    except Exception as e:
        self._log("output", f"Playback error: {e}", "err")
