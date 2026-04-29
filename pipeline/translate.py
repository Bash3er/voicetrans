import queue
import re
import time
from .utils import clean_text, fix_grammar


def load_translator(self):
    if self.src_lang == self.tgt_lang:
        self._log("translate", "src == tgt, skipping translator", "warn")
        return
    self._log("translate", f"Loading Argos Translate [{self.src_lang}→{self.tgt_lang}]…", "dim")
    self._stage("tr", "active")
    try:
        import argostranslate.package, argostranslate.translate
        installed = argostranslate.translate.get_installed_languages()
        installed_codes = {lang.code for lang in installed}
        if not (self.src_lang in installed_codes and self.tgt_lang in installed_codes):
            self._log("translate", "Checking language pack availability…", "warn")
            argostranslate.package.update_package_index()
            for pkg in argostranslate.package.get_available_packages():
                if pkg.from_code == self.src_lang and pkg.to_code == self.tgt_lang:
                    self._log("translate", f"Downloading {pkg}…", "dim")
                    argostranslate.package.install_from_path(pkg.download())
                    self._log("translate", "Language pack installed ✓", "green")
                    break
            else:
                self._log("translate",
                    f"No direct pack for {self.src_lang}→{self.tgt_lang}. Translation may be unavailable.",
                    "warn")
        self._translator = argostranslate.translate
        self._log("translate", "Argos ready ✓", "green")
    except ImportError:
        raise RuntimeError("argostranslate not installed.\nRun: pip install argostranslate")
    self._stage("tr", "done")


def _translate(self, text: str) -> str:
    if self._translator is None or self.src_lang == self.tgt_lang:
        return text
    try:
        result = self._translator.translate(text, self.src_lang, self.tgt_lang)
        return result if result else text
    except Exception as e:
        self._log("translate", f"Translation error: {e}", "err")
        return text


def _prepare_text_segments(self, pending: str, text: str):
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
        if len(words) >= 16:
            return [pieces[0]], ""
        if len(words) <= 5:
            return [pieces[0]], ""
        return [], pieces[0]
    completed = pieces[:-1]
    tail = pieces[-1]
    if len(tail.split()) >= 20:
        completed.append(tail)
        tail = ""
    return completed, tail


def translation_loop(self):
    tts_batch = ""
    batch_start_t = None
    batch_src = ""
    batch_stt_ms = 0.0
    batch_tr_ms = 0.0

    def flush_batch():
        nonlocal tts_batch, batch_start_t, batch_src, batch_stt_ms, batch_tr_ms
        text = tts_batch.strip()
        if text:
            self._submit_tts(batch_src, text, batch_stt_ms, batch_tr_ms)
        tts_batch = ""
        batch_start_t = None
        batch_src = ""
        batch_stt_ms = 0.0
        batch_tr_ms = 0.0

    while True:
        try:
            item = self._text_queue.get(timeout=self.TTS_BATCH_TIMEOUT * 0.5)
        except queue.Empty:
            if tts_batch and batch_start_t and time.perf_counter() - batch_start_t >= self.TTS_BATCH_TIMEOUT:
                flush_batch()
            continue

        if item is None:
            flush_batch()
            break

        text, stt_ms = item
        self._stage("clean", "active")
        cleaned = fix_grammar(clean_text(text))
        self._stage("clean", "done")
        if not cleaned:
            continue

        segments, self._pending_tail = _prepare_text_segments(self, self._pending_tail, cleaned)

        for segment in segments:
            self._stage("tr", "active")
            t1 = time.perf_counter()
            translated = _translate(self, segment)
            tr_ms = (time.perf_counter() - t1) * 1000
            self._stage("tr", "done")
            self._log("translate", f"{segment!r} → {translated!r}  [{tr_ms:.0f}ms]", "blue")

            tts_batch += (" " if tts_batch else "") + translated
            batch_src += (" " if batch_src else "") + segment
            batch_stt_ms += stt_ms
            batch_tr_ms += tr_ms
            if batch_start_t is None:
                batch_start_t = time.perf_counter()

            word_count = len(tts_batch.split())
            ends_sentence = tts_batch[-1] in ".!?" if tts_batch else False
            timeout_hit = time.perf_counter() - batch_start_t >= self.TTS_BATCH_TIMEOUT
            if word_count >= self.TTS_BATCH_WORDS or ends_sentence or timeout_hit:
                flush_batch()

    if self._pending_tail:
        seg = self._pending_tail.strip()
        if seg:
            translated = _translate(self, seg)
            self._submit_tts(seg, translated, 0, 0)
        self._pending_tail = ""

    self.playback_queue.put(None)
