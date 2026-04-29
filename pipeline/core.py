"""
VoiceTrans Pipeline — Optimised for low-latency real-time translation.
"""

import queue
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .stt import load_stt, transcribe
from .translate import load_translator, translation_loop
from .tts import load_tts, tts_generate, play_audio
from .utils import _resample


SAMPLE_RATE = 16000  # Input sample rate for STT

# ── Tuning knobs ──────────────────────────────────────────────────────────────
TTS_BATCH_WORDS = 12   # send to TTS once we have this many words
TTS_BATCH_TIMEOUT = 0.8  # … or after this many seconds, whichever comes first
TTS_WORKERS = 3    # parallel TTS generation threads
PLAYBACK_PREBUF = 2    # concatenate this many audio chunks before sd.play()
STT_FLUSH_TIMEOUT = 0.9  # seconds of silence before flushing pending STT text
# ─────────────────────────────────────────────────────────────────────────────


class Pipeline:
    def __init__(
        self,
        input_device: int,
        output_device: Optional[int],
        src_lang: str,
        tgt_lang: str,
        chunk_seconds: float,
        vad_filter: bool,
        log_queue: queue.Queue,
        status_queue: queue.Queue,
    ):
        self.input_device = input_device
        self.output_device = output_device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.chunk_seconds = chunk_seconds
        self.vad_filter = vad_filter
        self.log_q = log_queue
        self.status_q = status_queue
        self.SAMPLE_RATE = SAMPLE_RATE
        self.TTS_BATCH_TIMEOUT = TTS_BATCH_TIMEOUT
        self.TTS_BATCH_WORDS = TTS_BATCH_WORDS

        self._stop_event = threading.Event()
        self._audio_queue: queue.Queue = queue.Queue(maxsize=20)
        self._text_queue: queue.Queue = queue.Queue(maxsize=20)

        self._pending_tail = ""
        self._speech_active = False
        self._silence_frames = 0
        self._energy_threshold = 0.006
        self._silence_hold = 2

        self._recognizer = None
        self._vosk_model = None
        self._translator = None
        self._tts = None
        self._tts_voice = None

        self.tts_input_queue = queue.Queue(maxsize=50)
        self.playback_queue = queue.Queue(maxsize=100)
        self._tts_pool = ThreadPoolExecutor(max_workers=TTS_WORKERS, thread_name_prefix="tts")
        self._play_order_lock = threading.Lock()
        self._play_next_idx = 0
        self._play_ready: dict = {}
        self._play_order_idx = 0

    def stop(self):
        self._stop_event.set()
        self._tts_pool.shutdown(wait=False)

    def run(self):
        try:
            self._log("pipeline", "Initialising…", "dim")
            self._load_stt()
            self._load_translator()
            self._load_tts()

            self._log("pipeline", "All modules loaded. Listening…", "green")

            threads = [
                threading.Thread(target=self._capture_loop, daemon=True),
                threading.Thread(target=self._stt_loop, daemon=True),
                threading.Thread(target=self._translation_loop, daemon=True),
                threading.Thread(target=self._playback_worker, daemon=True),
            ]
            for t in threads:
                t.start()

            self._status("ready")

            while not self._stop_event.is_set():
                time.sleep(0.1)

            for t in threads:
                t.join(timeout=2.0)

        except Exception as e:
            self._log("pipeline", f"Fatal error: {e}", "err")
            self._status("error", msg=str(e))

    def _load_stt(self):
        load_stt(self)

    def _load_translator(self):
        load_translator(self)

    def _load_tts(self):
        load_tts(self)

    def _capture_loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            self._log("input", "sounddevice not installed.", "err")
            return

        info = sd.query_devices(self.input_device, kind="input")
        input_sr = int(info.get("default_samplerate", SAMPLE_RATE))
        blocksize = int(input_sr * self.chunk_seconds)
        self._log("input", f"Device [{self.input_device}] @ {input_sr}Hz  chunk={self.chunk_seconds:.1f}s", "dim")
        self._stage("input", "active")

        def callback(indata, frames, t, status):
            if status:
                self._log("input", f"Stream warning: {status}", "warn")
            chunk = indata[:, 0].copy().astype(np.float32)
            if input_sr != SAMPLE_RATE:
                chunk = _resample(chunk, input_sr, SAMPLE_RATE)
            if self.vad_filter:
                energy = float(np.sqrt(np.mean(chunk * chunk)))
                if energy < self._energy_threshold:
                    self._silence_frames += 1
                    if self._speech_active and self._silence_frames <= self._silence_hold:
                        if not self._audio_queue.full():
                            self._audio_queue.put(np.zeros_like(chunk))
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
            with sd.InputStream(device=self.input_device, samplerate=input_sr,
                                channels=1, dtype="float32", blocksize=blocksize,
                                callback=callback):
                self._stage("input", "done")
                while not self._stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            self._log("input", f"Audio capture error: {e}", "err")
        finally:
            self._stage("input", "idle")

    def _stt_loop(self):
        pending_text = ""
        pending_ms = 0.0
        last_text_t = None
        flush_timeout = max(self.chunk_seconds * 1.2, STT_FLUSH_TIMEOUT)

        while not self._stop_event.is_set() or not self._audio_queue.empty():
            try:
                chunk = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                if pending_text and last_text_t and time.perf_counter() - last_text_t > flush_timeout:
                    self._enqueue_text(pending_text, pending_ms)
                    pending_text = ""
                    pending_ms = 0.0
                    last_text_t = None
                continue

            self._stage("stt", "active")
            t0 = time.perf_counter()
            text = self._transcribe(chunk)
            stt_ms = (time.perf_counter() - t0) * 1000
            self._stage("stt", "done")

            if not text:
                continue

            self._log("stt", f"{text!r}  [{stt_ms:.0f}ms]", "dim")
            now = time.perf_counter()
            if pending_text and last_text_t and now - last_text_t > flush_timeout:
                self._enqueue_text(pending_text, pending_ms)
                pending_text = text
                pending_ms = stt_ms
            else:
                pending_text = (pending_text + " " + text).strip() if pending_text else text
                pending_ms += stt_ms
            last_text_t = now

        if pending_text:
            self._enqueue_text(pending_text, pending_ms)
        self._text_queue.put(None)

    def _enqueue_text(self, text, stt_ms):
        try:
            self._text_queue.put((text, stt_ms), timeout=0.5)
        except queue.Full:
            self._log("stt", "Text queue full, dropping chunk", "warn")

    def _translation_loop(self):
        translation_loop(self)

    def _submit_tts(self, src_text: str, translated: str, stt_ms: float, tr_ms: float):
        with self._play_order_lock:
            idx = self._play_order_idx
            self._play_order_idx += 1

        def _gen():
            self._stage("tts", "active")
            t0 = time.perf_counter()
            audio = self._tts_generate(translated)
            tts_ms = (time.perf_counter() - t0) * 1000
            self._stage("tts", "done")
            return idx, audio, src_text, translated, stt_ms, tr_ms, tts_ms

        future = self._tts_pool.submit(_gen)
        threading.Thread(target=self._collect_tts_result, args=(future, idx), daemon=True).start()

    def _collect_tts_result(self, future, idx):
        try:
            result = future.result()
        except Exception as e:
            self._log("tts", f"TTS future error: {e}", "err")
            return
        with self._play_order_lock:
            self._play_ready[idx] = result
            while self._play_next_idx in self._play_ready:
                item = self._play_ready.pop(self._play_next_idx)
                self.playback_queue.put(item)
                self._play_next_idx += 1

    def _playback_worker(self):
        pending_audio = []
        pending_meta = None

        def play_pending():
            nonlocal pending_audio, pending_meta
            if not pending_audio:
                return
            combined = np.concatenate(pending_audio)
            src, translated, stt_ms, tr_ms, tts_ms = pending_meta
            self._stage("output", "active")
            self._play_audio(combined)
            self._stage("output", "done")
            self._status("latency", stt=stt_ms, tr=tr_ms, tts=tts_ms)
            self._status("translation", src=src, translated=translated)
            pending_audio = []
            pending_meta = None

        while not self._stop_event.is_set():
            try:
                item = self.playback_queue.get(timeout=0.3)
            except queue.Empty:
                play_pending()
                continue

            if item is None:
                play_pending()
                break

            idx, audio, src_text, translated, stt_ms, tr_ms, tts_ms = item
            if audio is None:
                continue

            pending_audio.append(audio)
            if pending_meta is None:
                pending_meta = (src_text, translated, stt_ms, tr_ms, tts_ms)

            if len(pending_audio) >= PLAYBACK_PREBUF or self.playback_queue.empty():
                play_pending()

    def _transcribe(self, audio: np.ndarray) -> str:
        return transcribe(self, audio)

    def _tts_generate(self, text: str):
        return tts_generate(self, text)

    def _play_audio(self, audio: np.ndarray):
        play_audio(self, audio)

    def _log(self, tag, msg, color=""):
        self.log_q.put((tag, msg, color))

    def _status(self, event_type, **kwargs):
        self.status_q.put({"type": event_type, **kwargs})

    def _stage(self, key, state):
        self.status_q.put({"type": "stage", "key": key, "state": state})
