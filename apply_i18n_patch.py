"""
Run this ONCE, from the folder that contains eyeguard_ui.py:

    python apply_i18n_patch.py

It makes a backup (eyeguard_ui.py.bak), adds the translation hooks, and checks
that the result is valid Python before saving. If anything looks wrong it
changes nothing and tells you why.
"""
import sys
from pathlib import Path

TARGET = Path(__file__).parent / "eyeguard_ui.py"

NEW_METHODS = '''        elif key == "privacy":
            self._refresh_privacy_tab()
        self._ui_tr.refresh()

    # ---- Language switching ----

    LANGUAGE_NAMES = {"en": "English", "hi": "हिन्दी", "mr": "मराठी", "bn": "বাংলা"}

    def _language_button_text(self) -> str:
        return f"{t('Language')}: {self.LANGUAGE_NAMES.get(t.current_lang, t.current_lang)}"

    def _next_language(self) -> None:
        t.next_language()
        self.language_btn.configure(text=self._language_button_text())
        self._relabel_everything()

    def _relabel_everything(self) -> None:
        self.title(t("EyeGuard AI - Digital Eye Wellness Assistant"))
        key, values = self._status
        self._set_status(key, **values)
        if self.monitoring_active:
            self.toggle_btn.configure(text=t("Stop Monitoring"))
        else:
            self._configure_start_monitor_button()
        self._refresh_privacy_tab()
        self._ui_tr.refresh()

    def _set_status(self, key: str, **values) -> None:
        self._status = (key, values)
        self.status_label.configure(text=t(key, **values))

    def _show_from_tray(self) -> None:
'''

LANGUAGE_BUTTON = '''        self.quick_accent_selector.pack(fill="x", pady=(8, 0))
        self.language_btn = ctk.CTkButton(
            appearance_controls, text=self._language_button_text(), command=self._next_language
        )
        self.language_btn.pack(fill="x", pady=(8, 0))
'''

# (old, new) - each "old" must appear exactly once in your file
EDITS = [
    # the stray import above the docstring breaks "from __future__" - remove it
    ('from i18n import t\n"""\neyeguard_ui.py\n', '"""\neyeguard_ui.py\n'),
    ('from windows_magnifier import WindowsMagnifier\n',
     'from windows_magnifier import WindowsMagnifier\nfrom i18n import t\nfrom ui_i18n import UITranslator\n'),
    ('        self._build_layout()\n        self._show_tab("dashboard")\n',
     '        self._status = ("Status: Idle", {})\n        self._build_layout()\n'
     '        self._ui_tr = UITranslator(self)\n        self._show_tab("dashboard")\n'),
    ('        self.quick_accent_selector.pack(fill="x", pady=(8, 0))\n', LANGUAGE_BUTTON),
    ('        elif key == "privacy":\n            self._refresh_privacy_tab()\n\n    def _show_from_tray(self) -> None:\n',
     NEW_METHODS),

    # status line
    ('self.status_label.configure(text="Status: Starting camera...")', 'self._set_status("Status: Starting camera...")'),
    ('self.status_label.configure(text="Status: Idle")', 'self._set_status("Status: Idle")'),
    ('self.status_label.configure(text="Status: Monitoring")', 'self._set_status("Status: Monitoring")'),
    ('self.status_label.configure(text=f"Status: Error - {payload}")',
     'self._set_status("Status: Error - {payload}", payload=payload)'),

    # start/stop button
    ('self.toggle_btn.configure(text="Start Monitoring", fg_color=palette["fg"], hover_color=palette["hover"])',
     'self.toggle_btn.configure(text=t("Start Monitoring"), fg_color=palette["fg"], hover_color=palette["hover"])'),
    ('self.toggle_btn.configure(text="Stop Monitoring", fg_color="#c0392b", hover_color="#a93226")',
     'self.toggle_btn.configure(text=t("Stop Monitoring"), fg_color="#c0392b", hover_color="#a93226")'),

    # live dashboard values
    ('text="Closed" if metrics.get("is_eye_closed") else ("Open" if metrics.get("face_detected") else "No face detected")',
     'text=t("Closed") if metrics.get("is_eye_closed") else (t("Open") if metrics.get("face_detected") else t("No face detected"))'),
    ('self.distance_status_label.configure(text=distance_status)',
     'self.distance_status_label.configure(text=t(distance_status))'),
    ('self.posture_status_label.configure(text=metrics.get("posture_status", PostureStatus.UNKNOWN.value))',
     'self.posture_status_label.configure(text=t(metrics.get("posture_status", PostureStatus.UNKNOWN.value)))'),
    ('self.posture_guidance_label.configure(text=metrics.get("posture_guidance", "Keep your head and shoulders in view"))',
     'self.posture_guidance_label.configure(text=t(metrics.get("posture_guidance", "Keep your head and shoulders in view")))'),
    ('self.tracking_confidence_label.configure(text=metrics.get("tracking_confidence", "Low confidence"))',
     'self.tracking_confidence_label.configure(text=t(metrics.get("tracking_confidence", "Low confidence")))'),
    ('self.lighting_label.configure(text=metrics.get("lighting_quality", "—"))',
     'self.lighting_label.configure(text=t(metrics.get("lighting_quality", "—")))'),
    ('if posture_balance is not None else "Collecting"', 'if posture_balance is not None else t("Collecting")'),

    # break timer
    ('self.next_break_label.configure(text=f"Next break in {minutes:02d}:{secs:02d}")',
     'self.next_break_label.configure(text=t("Next break in {time}", time=f"{minutes:02d}:{secs:02d}"))'),
    ('text=f"TIME FOR A BREAK - look away for {settings.break_look_away_seconds}s"',
     'text=t("TIME FOR A BREAK - look away for {seconds}s", seconds=settings.break_look_away_seconds)'),
    ('self.break_label.configure(text=f"TIME FOR A BREAK - {self._break_remaining}s remaining")',
     'self.break_label.configure(text=t("TIME FOR A BREAK - {seconds}s remaining", seconds=self._break_remaining))'),
    ('self.break_start_btn.configure(state="normal", text="Start Break")',
     'self.break_start_btn.configure(state="normal", text=t("Start Break"))'),
    ('self.break_start_btn.configure(state="disabled", text="In progress")',
     'self.break_start_btn.configure(state="disabled", text=t("In progress"))'),

    # AI coach tab
    ('self.ai_analyze_btn.configure(state="disabled", text="Analyzing...")',
     'self.ai_analyze_btn.configure(state="disabled", text=t("Analyzing..."))'),
    ('self.ai_analyze_btn.configure(state="normal", text="Get AI Analysis")',
     'self.ai_analyze_btn.configure(state="normal", text=t("Get AI Analysis"))'),
    ('text=f"Top priority: {report.top_priority}"', 'text=t("Top priority: {text}", text=report.top_priority)'),
    ('text=f"Secondary: {report.secondary_priority}"', 'text=t("Secondary: {text}", text=report.secondary_priority)'),
    ('text=f"Going well: {report.going_well}"', 'text=t("Going well: {text}", text=report.going_well)'),
    ('text=f"Duration: {ex.duration_seconds} seconds"',
     'text=t("Duration: {seconds} seconds", seconds=ex.duration_seconds)'),
    ('text=f"Next reminder: in {ex.remind_after_minutes} minutes"',
     'text=t("Next reminder: in {minutes} minutes", minutes=ex.remind_after_minutes)'),
    ('self._append_chat(f"You: {question}")', 'self._append_chat(f"{t(\'You\')}: {question}")'),
    ('self._append_chat(f"Coach: {result[1]}")', 'self._append_chat(f"{t(\'Coach\')}: {result[1]}")'),
]

# Texts set with  .configure(text="...")  ->  .configure(text=t("..."))
WRAP = [
    "Start monitoring, sit comfortably, then calibrate",
    "Capturing your comfortable posture reference...",
    "Posture reference saved for this session",
    "Keep your head and shoulders in view, then calibrate again",
    "Break complete.",
    "Break reminder active",
    "Camera: On - local processing",
    "Pause Camera",
    "Camera: Off",
    "Start Camera",
]


def main() -> int:
    if not TARGET.exists():
        print(f"Could not find {TARGET}. Put this script in the same folder as eyeguard_ui.py.")
        return 1

    raw = TARGET.read_bytes()
    crlf = b"\r\n" in raw
    src = raw.decode("utf-8").replace("\r\n", "\n")

    if "ui_i18n" in src:
        print("eyeguard_ui.py is already patched. Nothing to do.")
        return 0

    applied, skipped = 0, 0
    for old, new in EDITS:
        count = src.count(old)
        if count != 1:
            print(f"  skipped ({count} matches): {old.strip().splitlines()[0][:70]}")
            skipped += 1
            continue
        src = src.replace(old, new)
        applied += 1

    for text in WRAP:
        old, new = f'configure(text="{text}")', f'configure(text=t("{text}"))'
        if old not in src:
            print(f"  skipped (not found): {text[:60]}")
            skipped += 1
            continue
        src = src.replace(old, new)
        applied += 1

    try:
        compile(src, "eyeguard_ui.py", "exec")
    except SyntaxError as exc:
        print(f"The patched file would not run (line {exc.lineno}: {exc.msg}).")
        print("Nothing was changed. Send me this message and I will fix it.")
        return 1

    TARGET.with_name(TARGET.name + ".bak").write_bytes(raw)
    TARGET.write_bytes((src.replace("\n", "\r\n") if crlf else src).encode("utf-8"))
    print(f"Done: {applied} edits applied, {skipped} skipped. Backup saved as eyeguard_ui.py.bak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
