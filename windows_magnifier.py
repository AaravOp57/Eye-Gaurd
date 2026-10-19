"""Optional Windows desktop magnification control."""

from __future__ import annotations

import ctypes
import os


class WindowsMagnifier:
    """Adjusts the Windows Magnification API zoom when the face is too close."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and os.name == "nt"
        self._initialized = False
        self._current = 1.0
        if self.enabled:
            try:
                self._dll = ctypes.WinDLL("Magnification.dll")
                self._initialized = bool(self._dll.MagInitialize())
            except Exception:
                self.enabled = False

    def update(self, distance_status: str) -> None:
        if not self.enabled or not self._initialized:
            return
        target = 1.5 if distance_status == "TOO CLOSE" else 1.0
        if target == self._current:
            return
        try:
            # The system magnifier uses a desktop-wide transform. Passing the
            # identity matrix at 100% restores the normal desktop view.
            matrix = (ctypes.c_float * 9)(
                target, 0, 0, 0, target, 0, 0, 0, 1,
            )
            self._dll.MagSetFullscreenTransform(ctypes.c_float(target), 0, 0)
            self._current = target
        except Exception:
            self.enabled = False

    def close(self) -> None:
        if self._initialized:
            try:
                self._dll.MagSetFullscreenTransform(ctypes.c_float(1.0), 0, 0)
                self._dll.MagUninitialize()
            except Exception:
                pass
            self._initialized = False
