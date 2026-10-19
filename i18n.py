import json
import re
from pathlib import Path

class Translator:
    def __init__(self, locales_dir=".", default="en"):
        # We default to "." because your JSON files are in the root folder
        self.locales_dir = Path(locales_dir)
        self.default_lang = default
        self.current_lang = default
        self.translations = {}
        self.refresh_locales()

    def refresh_locales(self):
        translations = {}
        for json_file in self.locales_dir.glob("*.json"):
            lang_code = json_file.stem.casefold()
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    translations[lang_code] = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not load locale file {json_file}: {exc}") from exc
        self.translations = translations

    def set_language(self, lang_code):
        lang_code = lang_code.casefold()
        if lang_code in self.translations:
            self.current_lang = lang_code
            return True
        return False

    def language_name(self, lang_code=None):
        code = (lang_code or self.current_lang).casefold()
        language_name = self.translations.get(code, {}).get("__language_name__")
        return language_name if isinstance(language_name, str) and language_name else code.upper()

    def next_language(self):
        self.refresh_locales()
        languages = sorted(self.translations)
        if not languages:
            return
        if self.default_lang in languages:
            languages.remove(self.default_lang)
            languages.insert(0, self.default_lang)
        try:
            idx = languages.index(self.current_lang)
        except ValueError:
            self.current_lang = self.default_lang if self.default_lang in languages else languages[0]
            return
        self.current_lang = languages[(idx + 1) % len(languages)]

    def __call__(self, key, **kwargs):
        lang_dict = self.translations.get(self.current_lang, {})
        default_dict = self.translations.get(self.default_lang, {})
        text = lang_dict.get(key, default_dict.get(key, key))
        if kwargs:
            try:
                return text.format(**kwargs)
            except KeyError:
                return text
        return text
    
t = Translator(Path(__file__).parent / "locales")
