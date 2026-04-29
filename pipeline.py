"""
VoiceTrans Pipeline
Captures audio → STT → clean → translate → TTS → virtual output
"""

import queue
import time
import threading
import numpy as np
from typing import Optional


SAMPLE_RATE = 16000  # Whisper requires 16kHz


class Pipeline:
    def __init__(
        self,
        input_device: int,
        output_device: Optional[int],
        src_lang: str,
        tgt_lang: str,
        model_size: str,
        chunk_seconds: float,
        vad_filter: bool,
        log_queue: queue.Queue,
        status_queue: queue.Queue,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_size = model_size
        self.chunk_seconds = chunk_seconds
        self.vad_filter = vad_filter
        self.log_q = log_queue
        self.status_q = status_queue

        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue = queue.Queue(maxsize=20)
        self._text_queue: queue.Queue = queue.Queue(maxsize=20)
        self._output_queue: queue.Queue = queue.Queue(maxsize=20)
        self._pending_tail: str = ""
        self._speech_active = False
        self._silence_frames = 0
        self._energy_threshold = 0.006
        self._silence_hold = 2

        # Lazy-loaded modules
        self._whisper = None
        self._translator = None
        self._tts = None
        self._tts_voice = None

    # ── Public ──────────────────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            self._log("pipeline", "Initialising…", "dim")
            self._load_stt()
            self._load_translator()
            self._load_tts()

            self._log("pipeline", "All modules loaded. Listening…", "green")

            capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            stt_thread = threading.Thread(target=self._stt_loop, daemon=True)
            translate_thread = threading.Thread(target=self._translation_loop, daemon=True)
            playback_thread = threading.Thread(target=self._playback_loop, daemon=True)

            capture_thread.start()
            stt_thread.start()
            translate_thread.start()
            playback_thread.start()

            self._status("ready")

            while not self._stop_event.is_set():
                time.sleep(0.1)

            capture_thread.join(timeout=1.0)
            stt_thread.join(timeout=1.0)
            translate_thread.join(timeout=2.0)
            self._output_queue.put(None)
            playback_thread.join(timeout=2.0)

        except Exception as e:
            self._log("pipeline", f"Fatal error: {e}", "err")
            self._status("error", msg=str(e))

    # ── Module loading ───────────────────────────────────────────────────────

    def _load_stt(self):
        model_name = self.model_size
        if self.src_lang == "en" and not model_name.endswith(".en"):
            model_name = f"{model_name}.en"

        self._log("stt", f"Loading faster-whisper [{model_name}]…", "dim")
        self._stage("stt", "active")
        try:
            from faster_whisper import WhisperModel
            compute_type = "int8"
            if self.model_size in {"small", "medium", "base"}:
                compute_type = "int8_float16"
            try:
                self._whisper = WhisperModel(
                    model_name, device="cpu", compute_type=compute_type
                )
            except Exception as e:
                self._log("stt", f"compute_type={compute_type} failed, falling back to int8: {e}", "warn")
                self._whisper = WhisperModel(
                    model_name, device="cpu", compute_type="int8"
                )
            self._log("stt", f"Whisper loaded ✓ ({model_name}, {compute_type})", "green")
        except ImportError:
            raise RuntimeError(
                "faster-whisper not installed.\n"
                "Run: pip install faster-whisper"
            )
        self._stage("stt", "done")

    def _load_translator(self):
        if self.src_lang == self.tgt_lang:
            self._log("translate", "src == tgt, skipping translator", "warn")
            return

        self._log("translate", f"Loading Argos Translate [{self.src_lang}→{self.tgt_lang}]…", "dim")
        self._stage("tr", "active")
        try:
            import argostranslate.package
            import argostranslate.translate

            installed = argostranslate.translate.get_installed_languages()
            installed_codes = {lang.code for lang in installed}

            src_installed = self.src_lang in installed_codes
            tgt_installed = self.tgt_lang in installed_codes

            if not (src_installed and tgt_installed):
                self._log("translate", "Checking language pack availability…", "warn")
                argostranslate.package.update_package_index()
                available = argostranslate.package.get_available_packages()

                for pkg in available:
                    if pkg.from_code == self.src_lang and pkg.to_code == self.tgt_lang:
                        self._log("translate", f"Downloading {pkg}…", "dim")
                        path = pkg.download()
                        argostranslate.package.install_from_path(path)
                        self._log("translate", "Language pack installed ✓", "green")
                        src_installed = tgt_installed = True
                        break
                else:
                    self._log(
                        "translate",
                        f"No direct pack for {self.src_lang}→{self.tgt_lang}. Translation may fallback or be unavailable.",
                        "warn"
                    )

            self._translator = argostranslate.translate
            self._log("translate", "Argos ready ✓", "green")

        except ImportError:
            raise RuntimeError(
                "argostranslate not installed.\n"
                "Run: pip install argostranslate"
            )
        self._stage("tr", "done")

    def _load_tts(self):
        self._log("tts", "Loading TTS engine…", "dim")
        self._stage("tts", "active")
        self._tts_engine = None
        self._tts_voice = None

        try:
            import edge_tts
            import os
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg_path = get_ffmpeg_exe()
            if not ffmpeg_path or not os.path.exists(ffmpeg_path):
                raise RuntimeError("ffmpeg executable not found for MP3 decoding")

            self._log("tts", "edge-tts + MP3 decoder ready", "dim")
            self._tts = "edge-tts"
            self._tts_voice = self._select_edge_voice()
            self._log("tts", f"Selected edge-tts voice: {self._tts_voice}", "dim")
        except Exception as e:
            self._log("tts", f"edge-tts unavailable: {e}", "warn")
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 165)
                voices = engine.getProperty("voices")
                self._log("tts", f"Available voices: {len(voices)}", "dim")
                preferred_voice = None
                if voices:
                    voice_summaries = []
                    for voice in voices[:5]:
                        details = []
                        if hasattr(voice, 'name'):
                            details.append(str(voice.name))
                        if hasattr(voice, 'id'):
                            details.append(str(voice.id))
                        if hasattr(voice, 'languages') and voice.languages:
                            details.append(str(voice.languages))
                        voice_summaries.append("/".join(details))

                        lower_id = getattr(voice, 'id', '')
                        lower_name = getattr(voice, 'name', '')
                        langs = []
                        if hasattr(voice, 'languages'):
                            langs = [str(l).lower() for l in voice.languages]
                        if self.tgt_lang == "en":
                            if "en" in lower_id.lower() or "english" in lower_name.lower() or any("en" in l for l in langs):
                                preferred_voice = voice
                                break

                    self._log("tts", f"Voices: {', '.join(voice_summaries)}", "dim")

                if preferred_voice is not None:
                    self._tts_voice = preferred_voice.id
                    self._log("tts", f"Selected voice: {getattr(preferred_voice, 'name', preferred_voice.id)}", "dim")
                else:
                    default_voice = engine.getProperty('voice')
                    self._tts_voice = default_voice
                    self._log("tts", f"Selected default voice: {default_voice}", "dim")

                engine.stop()
                self._tts_engine = None
                self._log("tts", "pyttsx3 TTS ready ✓", "green")
                self._tts = "pyttsx3"
            except ImportError:
                self._log("tts", "pyttsx3 not found, TTS disabled. Run: pip install pyttsx3", "warn")
                self._tts = None
            except Exception as e:
                self._log("tts", f"TTS engine error: {e}", "err")
                self._tts = None
        except Exception as e:
            self._log("tts", f"TTS engine error: {e}", "err")
            self._tts = None

        self._stage("tts", "done")

    # ── Audio capture ─────────────────────────────────────────────────────────

    def _capture_loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            self._log("input", "sounddevice not installed. Run: pip install sounddevice", "err")
            return

        device_info = sd.query_devices(self.input_device, kind="input")
        input_sr = int(device_info.get("default_samplerate", SAMPLE_RATE))
        stream_sr = input_sr
        blocksize = int(stream_sr * self.chunk_seconds)
        self._log(
            "input",
            f"Opening device [{self.input_device}] @ {stream_sr}Hz (target {SAMPLE_RATE}Hz), chunk={self.chunk_seconds:.1f}s",
            "dim"
        )
        self._stage("input", "active")

        def callback(indata, frames, t, status):
            if status:
                self._log("input", f"Stream warning: {status}", "warn")
            chunk = indata[:, 0].copy().astype(np.float32)
            if stream_sr != SAMPLE_RATE:
                chunk = _resample(chunk, stream_sr, SAMPLE_RATE)

            if self.vad_filter:
                energy = float(np.sqrt(np.mean(chunk * chunk)))
                if energy < self._energy_threshold:
                    self._silence_frames += 1
                    if self._speech_active and self._silence_frames <= self._silence_hold:
                        pass
                    else:
                        self._speech_active = False
                        return
                else:
                    self._speech_active = True
                    self._silence_frames = 0

            if not self._audio_queue.full():
                self._audio_queue.put(chunk)
            else:
                self._log("input", "Audio queue full, dropping chunk", "warn")

        try:
            with sd.InputStream(
                device=self.input_device,
                samplerate=stream_sr,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
            ):
                self._stage("input", "done")
                while not self._stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            self._log("input", f"Audio capture error: {e}", "err")
        finally:
            self._stage("input", "idle")

    # ── Main processing loops ─────────────────────────────────────────────────

    def _stt_loop(self):
        while not self._stop_event.is_set() or not self._audio_queue.empty():
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._stage("chunk", "active")
            audio = chunk
            self._stage("chunk", "done")

            self._stage("stt", "active")
            t0 = time.perf_counter()
            text = self._transcribe(audio)
            stt_ms = (time.perf_counter() - t0) * 1000
            self._stage("stt", "done")

            if not text:
                continue

            self._log("stt", f"{text!r}  [{stt_ms:.0f}ms]", "dim")
            try:
                self._text_queue.put((text, stt_ms), timeout=0.5)
            except queue.Full:
                self._log("stt", "Text queue full, dropping transcription chunk", "warn")

        self._text_queue.put(None)

    def _translation_loop(self):
        while True:
            item = self._text_queue.get()
            if item is None:
                break

            text, stt_ms = item
            self._stage("clean", "active")
            cleaned = fix_grammar(clean_text(text))
            self._stage("clean", "done")

            if not cleaned:
                continue

            ready_segments, self._pending_tail = self._prepare_text_segments(
                self._pending_tail, cleaned
            )

            for segment in ready_segments:
                self._stage("tr", "active")
                t1 = time.perf_counter()
                translated = self._translate(segment)
                tr_ms = (time.perf_counter() - t1) * 1000
                self._stage("tr", "done")

                self._log("translate", f"{segment!r} → {translated!r}  [{tr_ms:.0f}ms]", "blue")
                self._output_queue.put((segment, translated, stt_ms, tr_ms))

        if self._pending_tail:
            final_segment = self._pending_tail.strip()
            if final_segment:
                self._stage("tr", "active")
                translated = self._translate(final_segment)
                self._stage("tr", "done")
                self._log("translate", f"{final_segment!r} → {translated!r}", "blue")
                self._output_queue.put((final_segment, translated, 0, 0))
                self._pending_tail = ""

        self._output_queue.put(None)

    def _playback_loop(self):
        while True:
            item = self._output_queue.get()
            if item is None:
                break

            src_text, translated, stt_ms, tr_ms = item
            self._stage("tts", "active")
            t2 = time.perf_counter()
            audio_out = self._tts_generate(translated)
            tts_ms = (time.perf_counter() - t2) * 1000
            self._stage("tts", "done")

            if audio_out is not None:
                self._stage("output", "active")
                self._play_audio(audio_out)
                self._stage("output", "done")

            self._status("latency", stt=stt_ms, tr=tr_ms, tts=tts_ms)
            self._status("translation", src=src_text, translated=translated)

    def _prepare_text_segments(self, pending: str, text: str):
        import re
        full = (pending + " " + text).strip() if pending else text
        if not full:
            return [], ""

        pieces = [p.strip() for p in re.split(r'(?<=[.!?])\s+', full) if p.strip()]
        if not pieces:
            return [], ""

        if full[-1] in ".!?":
            return pieces, ""

        if len(pieces) == 1:
            words = pieces[0].split()
            if len(words) >= 14:
                return [pieces[0]], ""
            return [], pieces[0]

        completed = pieces[:-1]
        tail = pieces[-1]
        if len(tail.split()) >= 18:
            completed.append(tail)
            tail = ""

        return completed, tail

    # ── STT ───────────────────────────────────────────────────────────────────

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._whisper is None:
            return ""
        try:
            beam_size = 3 if self.model_size == "tiny" else 5
            segments, info = self._whisper.transcribe(
                audio,
                language=self.src_lang if self.src_lang != "auto" else None,
                vad_filter=self.vad_filter,
                beam_size=beam_size,
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text
        except Exception as e:
            self._log("stt", f"Transcription error: {e}", "err")
            return ""

    # ── Translation ───────────────────────────────────────────────────────────

    def _translate(self, text: str) -> str:
        if self._translator is None or self.src_lang == self.tgt_lang:
            return text
        try:
            result = self._translator.translate(text, self.src_lang, self.tgt_lang)
            return result if result else text
        except Exception as e:
            self._log("translate", f"Translation error: {e}", "err")
            return text

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _tts_generate(self, text: str) -> Optional[np.ndarray]:
        """Generate audio and return numpy float32 array, or None if TTS disabled."""
        if self._tts is None:
            return None

        if self._tts == "edge-tts":
            return self._edge_tts_to_numpy(text)

        if self._tts == "pyttsx3":
            return self._pyttsx3_to_numpy(text)

        return None

    def _pyttsx3_to_numpy(self, text: str) -> Optional[np.ndarray]:
        """Save pyttsx3 output to a temp wav, load it back as numpy."""
        import tempfile, os, wave
        if not text or not text.strip():
            return None

        import pyttsx3
        engine = pyttsx3.init()
        tmp_path = None

        try:
            engine.setProperty("rate", 165)
            if self._tts_voice:
                try:
                    engine.setProperty("voice", self._tts_voice)
                except Exception:
                    self._log("tts", f"Unable to set voice id {self._tts_voice}", "warn")
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
                    self._log("tts", "Generated WAV has zero frames", "warn")
                    return None
                frames = wf.readframes(nframes)
                sr = wf.getframerate()
                sampwidth = wf.getsampwidth()

            if len(frames) == 0:
                self._log("tts", "Generated WAV file is empty", "warn")
                return None

            if sampwidth == 1:
                arr = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                arr = (arr - 128.0) / 128.0
            elif sampwidth == 2:
                arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                arr = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            if arr.size == 0:
                self._log("tts", "Converted audio array is empty", "warn")
                return None

            if sr != SAMPLE_RATE:
                arr = _resample(arr, sr, SAMPLE_RATE)

            return arr

        except Exception as e:
            self._log("tts", f"TTS error: {e}", "err")
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

    def _edge_tts_to_numpy(self, text: str) -> Optional[np.ndarray]:
        import tempfile, os, asyncio
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
            voice = self._tts_voice or self._select_edge_voice()

            async def save_audio():
                communicate = edge_tts.Communicate(text, voice=voice)
                await communicate.save(tmp_path)

            asyncio.run(save_audio())

            if not tmp_path or not os.path.exists(tmp_path):
                self._log("tts", "edge-tts output file missing", "warn")
                return None

            from imageio_ffmpeg import get_ffmpeg_exe
            import subprocess

            ffmpeg_path = get_ffmpeg_exe()
            if not ffmpeg_path:
                self._log("tts", "ffmpeg not found for MP3 decoding", "err")
                return None

            command = [
                ffmpeg_path,
                "-y",
                "-loglevel", "error",
                "-i", tmp_path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "pipe:1",
            ]
            proc = subprocess.run(command, capture_output=True)
            if proc.returncode != 0:
                self._log("tts", f"ffmpeg decode failed: {proc.stderr.decode(errors='ignore').strip()}", "err")
                return None

            raw = proc.stdout
            if not raw:
                self._log("tts", "ffmpeg returned empty audio", "warn")
                return None

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if samples.size == 0:
                self._log("tts", "Decoded audio array is empty", "warn")
                return None

            return samples
        except Exception as e:
            self._log("tts", f"edge-tts error: {e}", "err")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ── Audio output ──────────────────────────────────────────────────────────

    def _play_audio(self, audio: np.ndarray):
        """Play audio to the selected output device (virtual cable)."""
        if audio is None or audio.size == 0:
            self._log("output", "No audio data to play", "warn")
            return

        try:
            import sounddevice as sd
            out_sr = SAMPLE_RATE
            try:
                device_info = sd.query_devices(self.output_device, kind="output")
                out_sr = int(device_info.get("default_samplerate", SAMPLE_RATE))
            except Exception as e:
                self._log("output", f"Output device query failed: {e}", "warn")
                out_sr = SAMPLE_RATE

            if out_sr != SAMPLE_RATE:
                audio = _resample(audio, SAMPLE_RATE, out_sr)

            if audio.size == 0:
                self._log("output", "Audio became empty after resampling", "warn")
                return

            sd.play(audio, samplerate=out_sr,
                    device=self.output_device, blocking=True)
        except Exception as e:
            self._log("output", f"Playback error: {e}", "err")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, tag, msg, color=""):
        self.log_q.put((tag, msg, color))

    def _status(self, event_type, **kwargs):
        self.status_q.put({"type": event_type, **kwargs})

    def _stage(self, key, state):
        self.status_q.put({"type": "stage", "key": key, "state": state})


# ── Text cleaning ─────────────────────────────────────────────────────────────

FILLERS = {
    "uh", "uhh", "uhhhh", "um", "umm", "hmm", "hm", "ah", "ahh",
    "er", "erm", "like", "you know", "okay so", "so yeah",
}


def clean_text(text: str) -> str:
    import re
    t = text.strip()
    # remove repeated punctuation
    t = re.sub(r"[.!?,;]{2,}", ".", t)
    # remove filler words (whole words only)
    for filler in FILLERS:
        t = re.sub(r"\b" + re.escape(filler) + r"\b", "", t, flags=re.IGNORECASE)
    # collapse multiple spaces
    t = re.sub(r"\s{2,}", " ", t).strip()
    # skip if too short or just punctuation
    if len(t) < 2 or not any(c.isalpha() for c in t):
        return ""
    return t


def fix_grammar(text: str) -> str:
    import re
    t = text.strip()
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"([.!?])\s*([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), t)
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Simple linear resample."""
    if from_sr == to_sr:
        return audio
    ratio = to_sr / from_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


# import guard
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
