import json
import tempfile
import unittest
from pathlib import Path

from i18n import Translator


class TranslatorTests(unittest.TestCase):
    def test_set_language_and_fallback_to_key(self):
        with tempfile.TemporaryDirectory() as directory:
            locale_dir = Path(directory)
            (locale_dir / "en.json").write_text(
                json.dumps({"greeting": "Hello"}), encoding="utf-8"
            )
            (locale_dir / "hi.json").write_text(
                json.dumps({"greeting": "नमस्ते"}), encoding="utf-8"
            )
            translator = Translator(locale_dir)

            self.assertTrue(translator.set_language("hi"))
            self.assertEqual(translator("greeting"), "नमस्ते")
            self.assertEqual(translator("missing"), "missing")
            self.assertFalse(translator.set_language("fr"))
            self.assertEqual(translator.current_lang, "hi")

    def test_locale_codes_are_case_insensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            locale_dir = Path(directory)
            (locale_dir / "Ma.json").write_text(
                json.dumps({"greeting": "നമസ്കാരം"}), encoding="utf-8"
            )
            translator = Translator(locale_dir)

            self.assertEqual(list(translator.translations), ["ma"])
            self.assertTrue(translator.set_language("ma"))
            self.assertEqual(translator("greeting"), "നമസ്കാരം")

    def test_next_language_cycles_with_default_first(self):
        with tempfile.TemporaryDirectory() as directory:
            locale_dir = Path(directory)
            for code in ("mr", "bn", "en", "hi"):
                (locale_dir / f"{code}.json").write_text("{}", encoding="utf-8")
            translator = Translator(locale_dir)

            self.assertEqual(translator.current_lang, "en")
            translator.next_language()
            self.assertEqual(translator.current_lang, "bn")
            translator.next_language()
            self.assertEqual(translator.current_lang, "hi")
            translator.next_language()
            self.assertEqual(translator.current_lang, "mr")
            translator.next_language()
            self.assertEqual(translator.current_lang, "en")

    def test_next_language_discovers_added_and_removed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            locale_dir = Path(directory)
            (locale_dir / "en.json").write_text("{}", encoding="utf-8")
            (locale_dir / "hi.json").write_text("{}", encoding="utf-8")
            translator = Translator(locale_dir)

            (locale_dir / "xx.json").write_text(
                json.dumps({"__language_name__": "Example", "greeting": "Hello"}),
                encoding="utf-8",
            )
            translator.next_language()
            self.assertEqual(translator.current_lang, "hi")
            translator.next_language()
            self.assertEqual(translator.current_lang, "xx")
            self.assertEqual(translator.language_name(), "Example")

            (locale_dir / "xx.json").unlink()
            translator.next_language()
            self.assertEqual(translator.current_lang, "en")
            self.assertNotIn("xx", translator.translations)

    def test_partial_locale_falls_back_to_default_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            locale_dir = Path(directory)
            (locale_dir / "en.json").write_text(
                json.dumps({"welcome": "Welcome"}), encoding="utf-8"
            )
            (locale_dir / "xx.json").write_text(
                json.dumps({"__language_name__": "Example"}), encoding="utf-8"
            )
            translator = Translator(locale_dir)
            translator.set_language("xx")

            self.assertEqual(translator.language_name(), "Example")
            self.assertEqual(translator("welcome"), "Welcome")

    def test_added_locales_translate_core_app_ui(self):
        locale_dir = Path(__file__).parents[1] / "locales"
        translator = Translator(locale_dir)
        base_ui = json.loads((locale_dir / "bn.json").read_text(encoding="utf-8"))
        ui_keys = tuple(key for key in base_ui if not key.startswith("coach_"))
        all_locale_keys = set()
        for locale_file in locale_dir.glob("*.json"):
            data = json.loads(locale_file.read_text(encoding="utf-8"))
            all_locale_keys.update(
                key for key in data if key != "__language_name__"
            )

        for language in ("gu", "ka", "ma", "od", "pu", "ta", "tu", "ud"):
            with self.subTest(language=language):
                self.assertTrue(translator.set_language(language))
                for key in (*ui_keys, *all_locale_keys):
                    with self.subTest(key=key):
                        if key in translator.translations[language]:
                            self.assertNotEqual(translator(key), key)

        self.assertTrue(translator.set_language("tu"))
        self.assertTrue(all(key in translator.translations["tu"] for key in all_locale_keys))


if __name__ == "__main__":
    unittest.main()
