"""
settings.py
------------
Handles loading, saving, and providing access to user-configurable
settings for EyeGuard AI. Settings are persisted locally as JSON so
no external database is required.

Never hardcode a real API key here - the default is an empty string,
and the user supplies their own key through the Settings panel in the UI.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict


SETTINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")


@dataclass
class AppSettings:
    """Typed container for every user-configurable setting."""

    gemini_api_key: str = ""            # User supplies their own key in Settings
    voice_alerts_enabled: bool = True
    notifications_enabled: bool = True
    focus_mode: bool = False
    blink_rate_threshold: int = 12      # Blinks per minute considered "too low"
    distance_too_close_ratio: float = 0.20   # Higher value allows sitting closer
    distance_too_far_ratio: float = 0.055
    theme: str = "dark"                 # "dark" or "light"
    accent_color: str = "blue"          # "blue" or "pink"
    language: str = "en"
    break_interval_minutes: int = 20    # 20-20-20 rule: look away every 20 minutes
    break_look_away_seconds: int = 20   # ...for at least 20 seconds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SettingsManager:
    """Loads settings from disk on startup and persists changes back to disk."""

    def __init__(self, path: str = SETTINGS_PATH) -> None:
        self._path = path
        self._settings = AppSettings()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self.load()

    def load(self) -> None:
        """Load settings from JSON, falling back to defaults on any error."""
        if not os.path.exists(self._path):
            self.save()  # create the file with defaults on first run
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            defaults = AppSettings()
            merged = {**defaults.to_dict(), **raw}
            self._settings = AppSettings(**{
                key: merged.get(key, getattr(defaults, key))
                for key in defaults.to_dict().keys()
            })
        except (json.JSONDecodeError, OSError, TypeError):
            # Corrupted or unreadable settings file - fall back to safe defaults
            self._settings = AppSettings()

    def save(self) -> None:
        """Persist the current settings to disk."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._settings.to_dict(), f, indent=2)
        except OSError:
            pass  # Non-fatal: app continues running with in-memory settings

    def get_all(self) -> AppSettings:
        return self._settings

    def get(self, key: str) -> Any:
        return getattr(self._settings, key)

    def set(self, key: str, value: Any) -> None:
        """Update a single setting and persist immediately."""
        if not hasattr(self._settings, key):
            raise AttributeError(f"Unknown setting: {key}")
        setattr(self._settings, key, value)
        self.save()

    def update_many(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        self.save()

    def has_gemini_key(self) -> bool:
        return bool(self._settings.gemini_api_key and self._settings.gemini_api_key.strip())
