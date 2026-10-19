from __future__ import annotations
"""Windows system-tray integration for EyeGuard AI."""
from i18n import t


import threading
from typing import Callable, Optional


class TrayController:
    """Owns the tray icon on its own thread and forwards menu actions safely."""

    def __init__(
        self,
        on_open: Callable[[], None],
        on_pause: Callable[[], None],
        on_resume: Callable[[], None],
        on_ai_coach: Callable[[], None],
        on_settings: Callable[[], None],
        on_exit: Callable[[], None],
    ) -> None:
        self._callbacks = {
            "open": on_open,
            "pause": on_pause,
            "resume": on_resume,
            "ai_coach": on_ai_coach,
            "settings": on_settings,
            "exit": on_exit,
        }
        self._icon = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def notify(self, title: str, message: str) -> None:
        if self._icon is not None and hasattr(self._icon, "notify"):
            try:
                self._icon.notify(message, title=title)
            except Exception:
                pass

    def start(self) -> None:
        try:
            import pystray
            from PIL import Image, ImageDraw
        except Exception:
            return

        def build_icon():
            image = Image.new("RGB", (64, 64), "#101820")
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((8, 8, 56, 56), radius=12, fill="#2fa572")
            draw.ellipse((22, 17, 42, 47), fill="#f4f7f5")
            draw.ellipse((26, 23, 30, 28), fill="#101820")
            draw.ellipse((34, 23, 38, 28), fill="#101820")
            return image

        def run() -> None:
            menu = pystray.Menu(
                pystray.MenuItem("Open EyeGuard", lambda: self._callbacks["open"](), default=True),
                pystray.MenuItem("Pause Monitoring", lambda: self._callbacks["pause"]()),
                pystray.MenuItem("Resume Monitoring", lambda: self._callbacks["resume"]()),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("AI Coach", lambda: self._callbacks["ai_coach"]()),
                pystray.MenuItem("Settings", lambda: self._callbacks["settings"]()),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Exit EyeGuard", lambda: self._callbacks["exit"]()),
            )
            self._icon = pystray.Icon("EyeGuard AI", build_icon(), "EyeGuard AI", menu)
            self._icon.run()

        self._thread = threading.Thread(target=run, name="EyeGuardTray", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass
        self._stop_event.set()
        self._icon = None
