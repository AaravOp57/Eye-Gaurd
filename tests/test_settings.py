import json
import tempfile
import unittest
from pathlib import Path

from settings import SettingsManager


class SettingsLanguageTests(unittest.TestCase):
    def test_language_setting_is_saved_and_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            manager = SettingsManager(path=str(path))

            manager.set("language", "mr")
            loaded = SettingsManager(path=str(path))

            self.assertEqual(loaded.get("language"), "mr")

    def test_existing_settings_default_to_english(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")

            manager = SettingsManager(path=str(path))

            self.assertEqual(manager.get("language"), "en")


if __name__ == "__main__":
    unittest.main()
