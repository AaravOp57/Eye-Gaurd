from __future__ import annotations
from i18n import t
"""
voice.py

--------
Wraps pyttsx3 text-to-speech in a background worker thread so that
speaking a message never blocks the GUI thread. A simple queue feeds
phrases to a dedicated worker; pyttsx3's engine is created once inside
that worker thread (pyttsx3 engines are not reliably thread-safe if
shared across threads that also touch the GUI).
"""


import queue
import threading
from typing import Optional


class VoiceManager:
    """Non-blocking text-to-speech announcer."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._engine_ready = False
        self._start_worker()

    def _start_worker(self) -> None:
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            self._engine_ready = True
        except Exception:
            # No TTS backend available on this system - fail silently,
            # the app must keep working without voice.
            engine = None
            self._engine_ready = False

        while not self._stop_flag.is_set():
            try:
                phrase = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if phrase is None:  # sentinel used to shut the worker down
                break
            if engine is not None and self.enabled:
                try:
                    engine.say(phrase)
                    engine.runAndWait()
                except Exception:
                    # Swallow TTS errors so a driver hiccup never crashes the app
                    pass

    def speak(self, phrase: str) -> None:
        """Queue a phrase to be spoken. Safe to call from any thread."""
        if not self.enabled:
            return
        self._queue.put(phrase)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def shutdown(self) -> None:
        self._stop_flag.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
