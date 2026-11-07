from __future__ import annotations
"""
notifications.py
-----------------
Sends desktop notifications using `plyer`, which on Windows delegates to
the native Windows toast notification system (win10toast backend) and
degrades gracefully on other platforms. Notifications run in a background
thread and are rate-limited per "topic" so the same alert can't spam the
user repeatedly within a short window.
"""


import logging
import threading
import time
from typing import Dict

from i18n import t


logger = logging.getLogger(__name__)


class NotificationManager:
    """Sends rate-limited desktop notifications without blocking the caller."""

    APP_NAME = "EyeGuard AI"

    def __init__(self, enabled: bool = True, cooldown_seconds: float = 60.0) -> None:
        self.enabled = enabled
        self.focus_mode = False
        self.cooldown_seconds = cooldown_seconds
        self._last_sent: Dict[str, float] = {}
        self._history = []
        self._lock = threading.Lock()
        self._tray = None

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def set_focus_mode(self, enabled: bool) -> None:
        self.focus_mode = enabled

    def set_tray(self, tray) -> None:
        self._tray = tray

    def _can_send(self, topic: str) -> bool:
        with self._lock:
            now = time.time()
            last = self._last_sent.get(topic, 0.0)
            if now - last >= self.cooldown_seconds:
                self._last_sent[topic] = now
                return True
            return False

    def notify(self, title: str, message: str, topic: str = "general", timeout: int = 8) -> bool:
        """
        Send a desktop notification, respecting the per-topic cooldown so
        the same kind of alert doesn't spam the user. Runs in a background
        thread so a slow OS notification call never freezes the GUI.

        Returns True if a notification was actually dispatched (i.e. it was
        not suppressed by the cooldown), so callers can use this to decide
        whether to count the event in their own statistics.
        """
        if not self.enabled:
            return False
        if not self._can_send(topic):
            return False
        with self._lock:
            self._history.append({"timestamp": time.time(), "topic": topic, "title": title, "message": message})
            self._history = self._history[-200:]
        threading.Thread(
            target=self._send, args=(title, message, timeout), daemon=True
        ).start()
        return True

    def get_history(self):
        with self._lock:
            return list(self._history)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def _send(self, title: str, message: str, timeout: int) -> None:
        try:
            from plyer import notification
            notification.notify(
                title=title,
                message=message,
                app_name=self.APP_NAME,
                timeout=timeout,
            )
            return
        except Exception:
            logger.warning("Desktop notification backend failed; trying the system tray.", exc_info=True)

        if self._tray is not None and hasattr(self._tray, "notify"):
            try:
                if self._tray.notify(title, message):
                    return
            except Exception:
                logger.warning("System-tray notification failed.", exc_info=True)

        logger.error("No notification backend could display the alert: %s", title)

    def notify_low_blink_rate(self) -> bool:
        if self.focus_mode:
            return False
        return self.notify(
            title=t("EyeGuard AI"),
            message=t("Please blink more frequently."),
            topic="low_blink_rate",
        )

    def notify_too_close(self) -> bool:
        return self.notify(
            title=t("EyeGuard AI"),
            message=t("Please move away from the screen."),
            topic="too_close",
        )

    def notify_posture_adjustment(self, guidance: str) -> bool:
        return self.notify(
            title=t("EyeGuard AI"),
            message=t(guidance),
            topic="posture_adjustment",
        )

    def notify_break_reminder(self) -> None:
        self.notify(
            title=f"{t('EyeGuard AI')} — {t('20-20-20 Break Timer')}",
            message=t("Look at something 20 feet away for 20 seconds."),
            topic="break_reminder",
            timeout=12,
        )

    def notify_exercise(self, exercise_name: str, instructions: str) -> None:
        if self.focus_mode:
            return
        self.notify(
            title=f"{t('EyeGuard AI')} — {t('Recommended Exercise')}",
            message=f"{exercise_name}: {instructions}",
            topic="exercise_reminder",
            timeout=12,
        )
