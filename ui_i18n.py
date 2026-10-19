"""
ui_i18n.py - translates the labels and buttons of a CustomTkinter window.

How it works
------------
The ENGLISH text of a widget is used as the translation key. Any widget whose
text appears in locales/hi.json, mr.json, bn.json, ... is remembered, and
refresh() sets its text to the current language. Text that has no translation
is simply left in English, so nothing ever breaks.

If the app changes a widget's text itself (a live value, a status message),
refresh() leaves that widget alone - the app is responsible for those and
should call t("...") when it sets them.
"""
import customtkinter as ctk

from i18n import t

TEXT_WIDGETS = (ctk.CTkLabel, ctk.CTkButton, ctk.CTkSwitch, ctk.CTkCheckBox, ctk.CTkRadioButton)


class UITranslator:
    def __init__(self, root):
        self.root = root
        self.items = {}  # widget -> [english_text, text_we_last_set]

    def _walk(self, widget):
        yield widget
        try:
            children = widget.winfo_children()
        except Exception:
            return
        for child in children:
            yield from self._walk(child)

    def refresh(self):
        known = set()
        for table in t.translations.values():
            known.update(table)

        # 1) register new widgets (including ones created after the last refresh)
        for w in self._walk(self.root):
            if isinstance(w, TEXT_WIDGETS) and w not in self.items:
                try:
                    text = w.cget("text")
                except Exception:
                    continue
                if text in known:
                    self.items[w] = [text, text]

        # 2) apply the current language
        for w in list(self.items):
            english, last = self.items[w]
            try:
                if not w.winfo_exists():
                    del self.items[w]
                    continue
                if w.cget("text") != last:
                    continue  # the app changed this text itself - leave it
                new = t(english)
                if new != last:
                    w.configure(text=new)
                self.items[w][1] = new
            except Exception:
                self.items.pop(w, None)
