from __future__ import annotations
"""
voice.py

--------
Wraps pyttsx3 text-to-speech in a background worker thread so that
speaking a message never blocks the GUI thread. A simple queue feeds
phrases to a dedicated worker; pyttsx3's engine is created once inside
that worker thread (pyttsx3 engines are not reliably thread-safe if
shared across threads that also touch the GUI).
"""


import logging
import queue
import threading
import time
from typing import Dict, Optional


logger = logging.getLogger(__name__)

VOICE_ALERT_REPEAT_SECONDS = 10.0
VOICE_LANGUAGE_TAGS = {
    "bn": "bn",
    "de": "de",
    "en": "en",
    "fr": "fr",
    "gu": "gu",
    "hi": "hi",
    "it": "it",
    "ja": "ja",
    "ka": "kn",
    "ma": "ml",
    "mr": "mr",
    "od": "or",
    "pu": "pa",
    "ru": "ru",
    "ta": "ta",
    "tu": "te",
    "ud": "ur",
    "zn_ch": "zh",
}


class VoiceManager:
    """Non-blocking text-to-speech announcer."""

    def __init__(self, enabled: bool = True, language: str = "en") -> None:
        self.enabled = enabled
        self._language = language.casefold()
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._engine_ready = False
        self._last_spoken: Dict[str, float] = {}
        self._voice_fallback_logged: set[str] = set()
        self._speech_lock = threading.Lock()
        self._start_worker()

    def _select_language_voice(self, engine) -> None:
        with self._speech_lock:
            language = self._language
        language_tag = VOICE_LANGUAGE_TAGS.get(language)
        if language_tag is None:
            return

        try:
            voices = engine.getProperty("voices")
        except Exception:
            logger.exception("Could not inspect installed speech voices.")
            return

        for voice in voices:
            for voice_language in getattr(voice, "languages", ()):
                if isinstance(voice_language, bytes):
                    voice_language = voice_language.decode("utf-8", errors="ignore")
                if str(voice_language).replace("_", "-").casefold().startswith(language_tag):
                    try:
                        engine.setProperty("voice", voice.id)
                    except Exception:
                        logger.exception("Could not select the installed %s speech voice.", language)
                    return

        if language not in self._voice_fallback_logged:
            logger.warning(
                "No installed speech voice matches language %s; using the system default.",
                language,
            )
            self._voice_fallback_logged.add(language)

    def _start_worker(self) -> None:
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self) -> None:
        engine = None
        while not self._stop_flag.is_set():
            try:
                phrase = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if phrase is None:  # sentinel used to shut the worker down
                break
            if not self.enabled:
                continue
            if engine is None:
                try:
                    import pyttsx3
                    engine = pyttsx3.init()
                    self._select_language_voice(engine)
                    self._engine_ready = True
                except Exception:
                    self._engine_ready = False
                    logger.exception("Could not initialize the voice alert engine.")
                    continue
            try:
                engine.say(phrase)
                engine.runAndWait()
            except Exception:
                logger.exception("Voice alert playback failed; the engine will be restarted.")
                self._engine_ready = False
                try:
                    engine.stop()
                except Exception:
                    logger.exception("Could not stop the failed voice alert engine.")
                engine = None
            else:
                try:
                    engine.stop()
                except Exception:
                    logger.exception("Could not reset the voice alert engine after playback.")
                engine = None
                self._engine_ready = False

    def speak(self, phrase: str, *, cooldown_key: Optional[str] = None,
              cooldown_seconds: float = 0.0) -> None:
        """Queue a phrase to be spoken. Safe to call from any thread."""
        if not self.enabled:
            return
        with self._speech_lock:
            if cooldown_key is not None:
                now = time.monotonic()
                last_spoken = self._last_spoken.get(cooldown_key)
                if last_spoken is not None and now - last_spoken < cooldown_seconds:
                    return
                self._last_spoken[cooldown_key] = now
            self._queue.put(phrase)

    def speak_once(self, phrase: str, *, cooldown_key: str) -> None:
        """Speak once until this alert key's cooldown is reset."""
        if not self.enabled:
            return
        with self._speech_lock:
            if cooldown_key in self._last_spoken:
                return
            self._last_spoken[cooldown_key] = time.monotonic()
            self._queue.put(phrase)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def set_language(self, language: str) -> None:
        with self._speech_lock:
            self._language = language.casefold()

    def reset_cooldown(self, cooldown_key: str) -> None:
        """Allow the next occurrence of a cleared alert to speak immediately."""
        with self._speech_lock:
            self._last_spoken.pop(cooldown_key, None)

    def shutdown(self) -> None:
        self._stop_flag.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
