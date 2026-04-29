"""
VoiceTrans - Real-time Game Voice Translator
Main GUI entry point
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import json
import os
import sys
from pathlib import Path

# ── Optional: suppress pygame welcome message ──
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

CONFIG_PATH = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "input_device": None,
    "output_device": None,
    "src_lang": "en",
    "tgt_lang": "hi",
    "stt_engine": "vosk",
    "whisper_model": "base",
    "chunk_seconds": 0.8,
    "vad_filter": True,
    "theme": "dark",
}


def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ────────────────────────────────────────────────────────────────────────────
#  Colour palette
# ────────────────────────────────────────────────────────────────────────────

DARK = {
    "bg":        "#0f1117",
    "bg2":       "#181c27",
    "bg3":       "#1e2436",
    "accent":    "#5865f2",
    "accent2":   "#45d483",
    "warn":      "#f0a500",
    "danger":    "#ed4245",
    "text":      "#e8eaf0",
    "text2":     "#8b92a5",
    "border":    "#2d3352",
    "card":      "#141824",
}

LIGHT = {
    "bg":        "#f0f2f8",
    "bg2":       "#ffffff",
    "bg3":       "#e4e8f0",
    "accent":    "#4752c4",
    "accent2":   "#2ea864",
    "warn":      "#d4870a",
    "danger":    "#d83232",
    "text":      "#1a1d2e",
    "text2":     "#5a6180",
    "border":    "#c8ccdc",
    "card":      "#ffffff",
}


# ────────────────────────────────────────────────────────────────────────────
#  Language pairs supported by Argos Translate
# ────────────────────────────────────────────────────────────────────────────

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pt": "Portuguese",
}


# ────────────────────────────────────────────────────────────────────────────
#  Main Application
# ────────────────────────────────────────────────────────────────────────────

class VoiceTransApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VoiceTrans")
        self.root.minsize(860, 640)
        self.root.configure(bg="#0f1117")

        self.cfg = load_config()
        self.C = DARK if self.cfg["theme"] == "dark" else LIGHT

        # Pipeline state
        self.pipeline = None
        self.running = False
        self.log_queue: queue.Queue = queue.Queue()
        self.status_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._refresh_devices()
        self._poll_queues()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        C = self.C
        root = self.root

        # ── top bar ──
        topbar = tk.Frame(root, bg=C["bg2"], height=48)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        tk.Label(
            topbar, text="⬡  VoiceTrans",
            font=("Consolas", 14, "bold"),
            fg=C["accent"], bg=C["bg2"]
        ).pack(side="left", padx=20, pady=12)

        # theme toggle
        self._theme_btn = tk.Button(
            topbar, text="☀  Light" if self.cfg["theme"] == "dark" else "☾  Dark",
            font=("Consolas", 10), fg=C["text2"], bg=C["bg2"],
            bd=0, cursor="hand2", activebackground=C["bg2"],
            command=self._toggle_theme
        )
        self._theme_btn.pack(side="right", padx=16)

        # version label
        tk.Label(
            topbar, text="v1.0  •  100% local  •  free",
            font=("Consolas", 9), fg=C["text2"], bg=C["bg2"]
        ).pack(side="right", padx=4)

        # ── main body ──
        body = tk.Frame(root, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # left sidebar
        sidebar = tk.Frame(body, bg=C["bg2"], width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        self._build_sidebar(sidebar)

        # right content
        content = tk.Frame(body, bg=C["bg"])
        content.pack(side="left", fill="both", expand=True, padx=0)

        self._build_content(content)

    def _section(self, parent, title):
        tk.Label(
            parent, text=title.upper(),
            font=("Consolas", 9, "bold"),
            fg=self.C["text2"], bg=self.C["bg2"]
        ).pack(anchor="w", padx=16, pady=(16, 4))

    def _card(self, parent, **kwargs):
        f = tk.Frame(parent, bg=self.C["card"],
                     highlightbackground=self.C["border"],
                     highlightthickness=1, **kwargs)
        return f

    def _build_sidebar(self, sidebar):
        C = self.C

        # ── Audio Devices ──
        self._section(sidebar, "Audio Input")

        self._input_var = tk.StringVar()
        self._input_cb = ttk.Combobox(
            sidebar, textvariable=self._input_var,
            state="readonly", font=("Consolas", 9), width=28
        )
        self._input_cb.pack(padx=12, pady=(0, 8), fill="x")
        self._input_cb.bind("<<ComboboxSelected>>", self._on_input_change)

        self._section(sidebar, "Virtual Mic Output")

        info = tk.Label(
            sidebar,
            text="Select a virtual cable as output.\nGames read this as a microphone.",
            font=("Consolas", 8), fg=C["text2"], bg=C["bg2"],
            justify="left"
        )
        info.pack(anchor="w", padx=16, pady=(0, 4))

        self._output_var = tk.StringVar()
        self._output_cb = ttk.Combobox(
            sidebar, textvariable=self._output_var,
            state="readonly", font=("Consolas", 9), width=28
        )
        self._output_cb.pack(padx=12, pady=(0, 4), fill="x")
        self._output_cb.bind("<<ComboboxSelected>>", self._on_output_change)

        tk.Button(
            sidebar, text="↻  Refresh Devices",
            font=("Consolas", 9), fg=C["text2"], bg=C["bg3"],
            bd=0, pady=4, cursor="hand2",
            activebackground=C["border"],
            command=self._refresh_devices
        ).pack(padx=12, pady=(0, 8), fill="x")

        # ── Language ──
        self._section(sidebar, "Language")

        lang_frame = tk.Frame(sidebar, bg=C["bg2"])
        lang_frame.pack(padx=12, fill="x")

        tk.Label(lang_frame, text="From", font=("Consolas", 9),
                 fg=C["text2"], bg=C["bg2"]).grid(row=0, column=0, sticky="w")
        tk.Label(lang_frame, text="To", font=("Consolas", 9),
                 fg=C["text2"], bg=C["bg2"]).grid(row=0, column=2, sticky="w")

        lang_names = [f"{v} ({k})" for k, v in LANGUAGES.items()]

        self._src_var = tk.StringVar()
        src_cb = ttk.Combobox(lang_frame, textvariable=self._src_var,
                               values=lang_names, state="readonly",
                               font=("Consolas", 9), width=11)
        src_cb.grid(row=1, column=0, sticky="ew")
        src_cb.bind("<<ComboboxSelected>>", self._on_lang_change)

        tk.Label(lang_frame, text="→", font=("Consolas", 12),
                 fg=C["accent"], bg=C["bg2"]).grid(row=1, column=1, padx=4)

        self._tgt_var = tk.StringVar()
        tgt_cb = ttk.Combobox(lang_frame, textvariable=self._tgt_var,
                               values=lang_names, state="readonly",
                               font=("Consolas", 9), width=11)
        tgt_cb.grid(row=1, column=2, sticky="ew")
        tgt_cb.bind("<<ComboboxSelected>>", self._on_lang_change)

        # set defaults
        src_key = self.cfg["src_lang"]
        tgt_key = self.cfg["tgt_lang"]
        self._src_var.set(f"{LANGUAGES.get(src_key, 'English')} ({src_key})")
        self._tgt_var.set(f"{LANGUAGES.get(tgt_key, 'Hindi')} ({tgt_key})")

        lang_frame.columnconfigure(0, weight=1)
        lang_frame.columnconfigure(2, weight=1)

        # ── STT Engine ──
        self._section(sidebar, "STT Engine")

        self._engine_var = tk.StringVar(value=self.cfg.get("stt_engine", "vosk"))
        engine_frame = tk.Frame(sidebar, bg=C["bg2"])
        engine_frame.pack(padx=12, fill="x", pady=(0, 8))

        for value, label in [("vosk", "VOSK"), ("whisper", "Whisper")]:
            rb = tk.Radiobutton(
                engine_frame, text=label,
                variable=self._engine_var, value=value,
                font=("Consolas", 9), fg=C["text"], bg=C["bg2"],
                selectcolor=C["bg3"], activebackground=C["bg2"],
                command=self._on_engine_change
            )
            rb.pack(side="left", padx=(0, 8))

        self._engine_note = tk.Label(
            sidebar,
            text="Free offline STT" if self._engine_var.get() == "vosk" else "Whisper model selection applies",
            font=("Consolas", 8), fg=C["text2"], bg=C["bg2"]
        )
        self._engine_note.pack(anchor="w", padx=16)

        # ── STT Model ──
        self._section(sidebar, "STT Model")

        models = ["tiny", "base", "small", "medium"]
        self._model_var = tk.StringVar(value=self.cfg["whisper_model"])
        model_frame = tk.Frame(sidebar, bg=C["bg2"])
        model_frame.pack(padx=12, fill="x", pady=(0, 8))

        for m in models:
            rb = tk.Radiobutton(
                model_frame, text=m,
                variable=self._model_var, value=m,
                font=("Consolas", 9), fg=C["text"], bg=C["bg2"],
                selectcolor=C["bg3"], activebackground=C["bg2"],
                command=self._on_model_change
            )
            rb.pack(side="left", padx=(0, 8))

        model_notes = {
            "tiny": "fastest, less accurate",
            "base": "fast and balanced",
            "small": "more accurate, still responsive ✓",
            "medium": "most accurate, slower",
        }
        model_note_text = (
            "VOSK ignores Whisper model selection"
            if self._engine_var.get() == "vosk"
            else model_notes[self.cfg["whisper_model"]]
        )
        self._model_note = tk.Label(
            sidebar, text=model_note_text,
            font=("Consolas", 8), fg=C["text2"], bg=C["bg2"]
        )
        self._model_note.pack(anchor="w", padx=16)

        # ── Chunk Size ──
        self._section(sidebar, "Chunk Size")

        chunk_frame = tk.Frame(sidebar, bg=C["bg2"])
        chunk_frame.pack(padx=12, fill="x", pady=(0, 16))

        self._chunk_var = tk.DoubleVar(value=self.cfg["chunk_seconds"])
        self._chunk_label = tk.Label(
            chunk_frame, text=f"{self.cfg['chunk_seconds']:.1f}s",
            font=("Consolas", 10, "bold"), fg=C["accent"], bg=C["bg2"],
            width=4
        )
        self._chunk_label.pack(side="right")

        sl = tk.Scale(
            chunk_frame, from_=0.3, to=2.0, resolution=0.1,
            orient="horizontal", variable=self._chunk_var,
            showvalue=False, bg=C["bg2"], fg=C["text2"],
            troughcolor=C["bg3"], highlightthickness=0, bd=0,
            command=self._on_chunk_change
        )
        sl.pack(side="left", fill="x", expand=True)

        # ── VAD ──
        self._vad_var = tk.BooleanVar(value=self.cfg["vad_filter"])
        tk.Checkbutton(
            sidebar, text="VAD Filter (skip silence)",
            variable=self._vad_var,
            font=("Consolas", 9), fg=C["text"], bg=C["bg2"],
            selectcolor=C["bg3"], activebackground=C["bg2"],
            command=self._on_vad_change
        ).pack(anchor="w", padx=16, pady=(0, 16))

    def _build_content(self, parent):
        C = self.C

        # ── big start button ──
        btn_frame = tk.Frame(parent, bg=C["bg"])
        btn_frame.pack(pady=20)

        self._start_btn = tk.Button(
            btn_frame,
            text="▶  START TRANSLATION",
            font=("Consolas", 13, "bold"),
            fg="#ffffff", bg=C["accent"],
            activebackground=C["accent2"],
            bd=0, padx=30, pady=14,
            cursor="hand2",
            command=self._toggle_pipeline
        )
        self._start_btn.pack()

        self._status_lbl = tk.Label(
            btn_frame, text="idle — configure settings and press start",
            font=("Consolas", 9), fg=C["text2"], bg=C["bg"]
        )
        self._status_lbl.pack(pady=(6, 0))

        # ── pipeline stages ──
        pipe_frame = tk.Frame(parent, bg=C["bg"])
        pipe_frame.pack(padx=20, fill="x")

        self._stages = {}
        stage_names = [
            ("input",  "Audio In"),
            ("chunk",  "Chunking"),
            ("stt",    "STT"),
            ("clean",  "Clean"),
            ("tr",     "Translate"),
            ("tts",    "TTS"),
            ("output", "Output"),
        ]

        for i, (key, label) in enumerate(stage_names):
            sf = tk.Frame(pipe_frame, bg=C["bg"])
            sf.pack(side="left", expand=True)

            box = tk.Label(
                sf, text=label,
                font=("Consolas", 9), fg=C["text2"], bg=C["bg3"],
                padx=8, pady=5,
                relief="flat",
                highlightbackground=C["border"],
                highlightthickness=1
            )
            box.pack(expand=True, fill="x")
            self._stages[key] = box

            if i < len(stage_names) - 1:
                tk.Label(pipe_frame, text="→", font=("Consolas", 11),
                         fg=C["text2"], bg=C["bg"]).pack(side="left")

        # ── latency cards ──
        lat_frame = tk.Frame(parent, bg=C["bg"])
        lat_frame.pack(padx=20, pady=16, fill="x")

        self._lat = {}
        for key, label in [("stt", "STT"), ("tr", "Translate"), ("tts", "TTS"), ("total", "Total")]:
            card = tk.Frame(lat_frame, bg=C["card"],
                            highlightbackground=C["border"], highlightthickness=1)
            card.pack(side="left", expand=True, fill="x", padx=(0, 8))

            val_lbl = tk.Label(card, text="—", font=("Consolas", 18, "bold"),
                               fg=C["accent2"] if key != "total" else C["accent"],
                               bg=C["card"])
            val_lbl.pack(pady=(8, 2))
            tk.Label(card, text=f"{label} ms", font=("Consolas", 8),
                     fg=C["text2"], bg=C["card"]).pack(pady=(0, 8))
            self._lat[key] = val_lbl

        # ── last translation display ──
        trans_frame = tk.Frame(parent, bg=C["bg"])
        trans_frame.pack(padx=20, fill="x", pady=(0, 12))

        tk.Label(trans_frame, text="LAST TRANSLATION",
                 font=("Consolas", 9, "bold"), fg=C["text2"], bg=C["bg"]
                 ).pack(anchor="w", pady=(0, 4))

        trans_card = tk.Frame(trans_frame, bg=C["card"],
                              highlightbackground=C["border"], highlightthickness=1)
        trans_card.pack(fill="x")

        row1 = tk.Frame(trans_card, bg=C["card"])
        row1.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(row1, text="SRC", font=("Consolas", 8), fg=C["text2"], bg=C["card"]).pack(side="left")
        self._src_text = tk.Label(row1, text="—", font=("Consolas", 11),
                                  fg=C["text"], bg=C["card"])
        self._src_text.pack(side="left", padx=8)

        row2 = tk.Frame(trans_card, bg=C["card"])
        row2.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(row2, text="OUT", font=("Consolas", 8), fg=C["accent2"], bg=C["card"]).pack(side="left")
        self._tr_text = tk.Label(row2, text="—", font=("Consolas", 12, "bold"),
                                 fg=C["accent2"], bg=C["card"])
        self._tr_text.pack(side="left", padx=8)

        # ── log ──
        log_frame = tk.Frame(parent, bg=C["bg"])
        log_frame.pack(padx=20, pady=(0, 16), fill="both", expand=True)

        log_header = tk.Frame(log_frame, bg=C["bg"])
        log_header.pack(fill="x", pady=(0, 4))
        tk.Label(log_header, text="LOG", font=("Consolas", 9, "bold"),
                 fg=C["text2"], bg=C["bg"]).pack(side="left")
        tk.Button(log_header, text="clear", font=("Consolas", 8),
                  fg=C["text2"], bg=C["bg"], bd=0, cursor="hand2",
                  command=self._clear_log).pack(side="right")

        self._log_text = tk.Text(
            log_frame, font=("Consolas", 9),
            bg=C["card"], fg=C["text"],
            insertbackground=C["text"],
            relief="flat", bd=0,
            highlightbackground=C["border"], highlightthickness=1,
            state="disabled", wrap="word"
        )
        self._log_text.pack(fill="both", expand=True)

        # configure tag colors
        self._log_text.tag_config("green", foreground=self.C["accent2"])
        self._log_text.tag_config("blue", foreground=self.C["accent"])
        self._log_text.tag_config("warn", foreground=self.C["warn"])
        self._log_text.tag_config("err", foreground=self.C["danger"])
        self._log_text.tag_config("dim", foreground=self.C["text2"])

        self._log("system", "VoiceTrans ready.", "green")
        self._log("system", "Install deps: pip install faster-whisper sounddevice argostranslate pyttsx3 numpy vosk", "dim")
        self._log("system", "For virtual mic output: install VB-Audio Cable (free)", "dim")

    # ── Device management ────────────────────────────────────────────────

    def _refresh_devices(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
        except Exception as e:
            self._log("device", f"sounddevice not installed: {e}", "err")
            return

        self._all_devices = devices
        input_names = []
        output_names = []

        for i, d in enumerate(devices):
            name = f"[{i}] {d['name']}"
            if d["max_input_channels"] > 0:
                input_names.append(name)
            if d["max_output_channels"] > 0:
                output_names.append(name)

        self._input_cb["values"] = input_names
        self._output_cb["values"] = output_names

        # restore saved selections
        if self.cfg["input_device"] is not None:
            idx = self.cfg["input_device"]
            matching = [n for n in input_names if n.startswith(f"[{idx}]")]
            if matching:
                self._input_var.set(matching[0])

        if self.cfg["output_device"] is not None:
            idx = self.cfg["output_device"]
            matching = [n for n in output_names if n.startswith(f"[{idx}]")]
            if matching:
                self._output_var.set(matching[0])

        # auto-select VB-Cable if present
        vb_inputs = [n for n in input_names if "cable" in n.lower() or "vb-audio" in n.lower()]
        vb_outputs = [n for n in output_names if "cable" in n.lower() or "vb-audio" in n.lower()]

        if vb_inputs and not self._input_var.get():
            self._input_var.set(vb_inputs[0])
            self._log("device", f"Auto-selected input: {vb_inputs[0]}", "green")
        if vb_outputs and not self._output_var.get():
            self._output_var.set(vb_outputs[0])
            self._log("device", f"Auto-selected output: {vb_outputs[0]}", "green")

        self._log("device", f"Found {len(input_names)} input, {len(output_names)} output devices.", "dim")

    def _get_device_index(self, var: tk.StringVar):
        val = var.get()
        if not val:
            return None
        try:
            return int(val.split("]")[0].lstrip("["))
        except Exception:
            return None

    # ── Settings callbacks ───────────────────────────────────────────────

    def _on_input_change(self, *_):
        self.cfg["input_device"] = self._get_device_index(self._input_var)
        save_config(self.cfg)

    def _on_output_change(self, *_):
        self.cfg["output_device"] = self._get_device_index(self._output_var)
        save_config(self.cfg)

    def _on_lang_change(self, *_):
        def extract_code(s):
            return s.split("(")[-1].rstrip(")")
        self.cfg["src_lang"] = extract_code(self._src_var.get())
        self.cfg["tgt_lang"] = extract_code(self._tgt_var.get())
        save_config(self.cfg)

    def _on_model_change(self):
        m = self._model_var.get()
        self.cfg["whisper_model"] = m
        if self.cfg.get("stt_engine", "vosk") == "vosk":
            self._model_note.config(text="VOSK ignores Whisper model selection")
        else:
            notes = {"tiny": "fastest, less accurate", "base": "best balance ✓",
                     "small": "accurate, slower", "medium": "most accurate"}
            self._model_note.config(text=notes.get(m, ""))
        save_config(self.cfg)

    def _on_engine_change(self):
        engine = self._engine_var.get()
        self.cfg["stt_engine"] = engine
        note = "Free offline STT" if engine == "vosk" else "Whisper model selection applies"
        self._engine_note.config(text=note)
        save_config(self.cfg)

    def _on_chunk_change(self, *_):
        v = round(self._chunk_var.get(), 1)
        self._chunk_label.config(text=f"{v:.1f}s")
        self.cfg["chunk_seconds"] = v
        save_config(self.cfg)

    def _on_vad_change(self):
        self.cfg["vad_filter"] = self._vad_var.get()
        save_config(self.cfg)

    def _toggle_theme(self):
        self.cfg["theme"] = "light" if self.cfg["theme"] == "dark" else "dark"
        save_config(self.cfg)
        messagebox.showinfo("VoiceTrans", "Restart the app to apply theme change.")

    # ── Pipeline control ─────────────────────────────────────────────────

    def _toggle_pipeline(self):
        if not self.running:
            self._start()
        else:
            self._stop()

    def _start(self):
        from pipeline import Pipeline

        in_dev = self._get_device_index(self._input_var)
        out_dev = self._get_device_index(self._output_var)

        if in_dev is None:
            messagebox.showwarning("VoiceTrans", "Please select an input device.")
            return

        if out_dev is None:
            if not messagebox.askyesno(
                "VoiceTrans",
                "No output device selected. Audio will play to the default output device. Continue?"
            ):
                return

        self.running = True
        self._start_btn.config(text="■  STOP TRANSLATION", bg=self.C["danger"])
        engine_label = self.cfg.get("stt_engine", "vosk")
        status_text = "loading VOSK model…" if engine_label == "vosk" else "loading Whisper model…"
        self._status_lbl.config(text=status_text)
        self._set_stage_all("idle")

        self.pipeline = Pipeline(
            input_device=in_dev,
            output_device=out_dev,
            src_lang=self.cfg["src_lang"],
            tgt_lang=self.cfg["tgt_lang"],
            model_size=self.cfg["whisper_model"],
            stt_engine=self.cfg.get("stt_engine", "vosk"),
            chunk_seconds=self.cfg["chunk_seconds"],
            vad_filter=self.cfg["vad_filter"],
            log_queue=self.log_queue,
            status_queue=self.status_queue,
        )

        t = threading.Thread(target=self.pipeline.run, daemon=True)
        t.start()

    def _stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
        self._start_btn.config(text="▶  START TRANSLATION", bg=self.C["accent"])
        self._status_lbl.config(text="stopped")
        self._set_stage_all("idle")
        self._log("system", "Pipeline stopped.", "warn")

    # ── Stage indicator ───────────────────────────────────────────────────

    def _set_stage(self, key, state):
        """state: idle | active | done"""
        C = self.C
        box = self._stages.get(key)
        if not box:
            return
        if state == "active":
            box.config(bg=C["accent"], fg="#ffffff",
                       highlightbackground=C["accent"])
        elif state == "done":
            box.config(bg=C["accent2"], fg="#000000",
                       highlightbackground=C["accent2"])
        else:
            box.config(bg=C["bg3"], fg=C["text2"],
                       highlightbackground=C["border"])

    def _set_stage_all(self, state):
        for k in self._stages:
            self._set_stage(k, state)

    # ── Log helpers ───────────────────────────────────────────────────────

    def _log(self, tag, msg, color=""):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{tag}] {msg}\n"
        t = self._log_text
        t.config(state="normal")
        t.insert("end", line, color or "")
        t.see("end")
        t.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    # ── Queue polling ─────────────────────────────────────────────────────

    def _poll_queues(self):
        # process log queue
        while not self.log_queue.empty():
            try:
                tag, msg, color = self.log_queue.get_nowait()
                self._log(tag, msg, color)
            except Exception:
                pass

        # process status queue
        while not self.status_queue.empty():
            try:
                event = self.status_queue.get_nowait()
                self._handle_status(event)
            except Exception:
                pass

        self.root.after(80, self._poll_queues)

    def _handle_status(self, event: dict):
        kind = event.get("type")

        if kind == "stage":
            self._set_stage(event["key"], event["state"])

        elif kind == "latency":
            for k in ("stt", "tr", "tts"):
                v = event.get(k)
                if v is not None:
                    self._lat[k].config(text=str(int(v)))
            total = sum(event.get(k, 0) for k in ("stt", "tr", "tts"))
            self._lat["total"].config(text=str(int(total)))

        elif kind == "translation":
            self._src_text.config(text=event.get("src", ""))
            self._tr_text.config(text=event.get("translated", ""))
            self._status_lbl.config(text=f"last: {event.get('translated','')[:60]}")

        elif kind == "ready":
            self._status_lbl.config(text="pipeline running — listening…")
            self._set_stage_all("done")

        elif kind == "error":
            self._status_lbl.config(text=f"error: {event.get('msg','')}")


def main():
    root = tk.Tk()
    app = VoiceTransApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
