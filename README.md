# EyeGuard AI

A desktop digital eye-wellness assistant built with Python, OpenCV, MediaPipe
Face Mesh, CustomTkinter, and Google Gemini. It watches your blink rate and
approximate screen distance through your webcam, nudges you with the 20-20-20
rule, and uses an AI Health Coach to turn your session stats into friendly,
structured guidance and a recommended eye exercise.

> **Not a medical device.** EyeGuard AI gives general wellness suggestions
> only. It does not diagnose or treat any medical or eye condition, and its
> distance readings are approximate, uncalibrated estimates — not precise
> measurements. If you have persistent eye discomfort, consult an eye-care
> professional.

---

## 1. Project Structure

```text
EyeGuard/
│
├── main.py                # Entry point - launches the app
├── eyeguard_ui.py          # CustomTkinter GUI, webcam worker thread, alert wiring
├── eye_detector.py         # Blink detection (Eye Aspect Ratio over MediaPipe landmarks)
├── distance_detector.py    # Approximate screen-distance classification
├── eyeguard_ai_coach.py     # Gemini-powered AIHealthCoach + safe JSON parsing/fallback
├── notifications.py        # Rate-limited desktop notifications (plyer)
├── tray.py                 # System-tray icon and menu lifecycle
├── windows_magnifier.py    # Optional Windows 100%/150% zoom control
├── voice.py                # Threaded pyttsx3 text-to-speech
├── statistics.py           # Session + cumulative statistics, persisted as JSON
├── settings.py             # User settings, persisted as JSON
├── requirements.txt
├── README.md
└── data/                   # Created automatically - settings.json & statistics.json
```

## 2. Installation

Requires **Python 3.12 or 3.13**. The project was validated on Python 3.13 using a
virtual environment; installing with a system-wide Python can hit the NumPy build
issue seen on Windows.

```bash
cd EyeGuard
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Notes:

- `mediapipe` and `opencv-python` wheels are large; the first install may
  take a few minutes.
- Desktop notifications use `plyer`, which uses native Windows toast
  notifications on Windows (via `pywin32`, included in requirements.txt for
  Windows) and degrades gracefully (silently) on platforms without a
  supported backend.
- If `pyttsx3` can't find a TTS engine on your system, voice alerts simply
  won't speak — the rest of the app keeps working normally.

## 3. Configuring the Gemini API Key

**Never hardcode your API key into the source code.** Configure it from
inside the app:

1. Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Run the app, open **Settings** in the sidebar.
3. Paste the key into the **Gemini API Key** field and click **Save Settings**.

The key is stored locally in `data/settings.json` on your machine only. If
no key is configured (or Gemini is temporarily unreachable, or it returns a
response that can't be parsed), the **AI Health Coach** automatically falls
back to a local, rule-based report instead of crashing or leaving the
dashboard empty.

## 4. Running the Application

```bash
python main.py
```

Click **Start Monitoring** to begin the webcam feed and detection. Closing the
window with X hides it in the tray while monitoring, reminders, notifications,
and background processing continue. The tray menu can open EyeGuard, pause or
resume monitoring, open AI Coach or Settings, and fully exit the application.
Double-clicking the tray icon opens the window. **Exit EyeGuard** stops the
webcam, worker threads, timers, notifications, magnification, and tray icon.

Use the **Language** button in the sidebar to cycle through the JSON files in
`locales/`. To add or remove a language, add or remove its `<code>.json` file;
the app discovers files at startup and refreshes the list when the language
button is clicked. Optionally include a `__language_name__` field to show the
language's native name in the picker; otherwise the file code is shown. The
selected language is saved locally and restored the next time the app starts.
Missing translations fall back to English.

The Statistics page records sparse live samples and displays blink rate,
relative distance status, and health score charts. It shows “Collecting data..."
until a graph has enough valid points. Session summaries are stored locally in
`data/statistics.json` and can be viewed for Today, 7 Days, or 30 Days.

When Windows Magnifier support is available, a TOO CLOSE distance status sets
desktop magnification to 150%; GOOD DISTANCE and other statuses restore 100%.
This is an optional desktop-wide Windows feature and is disabled automatically
when the Magnification API is unavailable.

## 5. How Each Module Works

- **`main.py`** — Thin entry point. Imports the UI, handles missing
  dependencies and unexpected errors gracefully with clear console messages
  instead of a raw traceback dump.

- **`eye_detector.py`** — Implements blink detection using the **Eye Aspect
  Ratio (EAR)** technique: six landmark points per eye (from MediaPipe's 468
  facial landmarks) are used to compute the ratio of vertical to horizontal
  eye-opening distance. EAR drops sharply when the eye closes. A blink is
  only counted once, exactly on the transition from "closed for enough
  consecutive frames" back to "open" — this avoids double-counting a single
  blink across multiple frames and filters out single-frame landmark noise.

- **`distance_detector.py`** — Estimates approximate screen distance from
  the *interocular distance* (pixel distance between the outer corners of
  the eyes) as a ratio of frame width. A larger ratio means the face fills
  more of the frame (closer); a smaller ratio means farther away. Two
  configurable thresholds classify the result as `TOO CLOSE`, `GOOD
  DISTANCE`, or `TOO FAR`. This is explicitly an uncalibrated, relative
  estimate, not a precise measurement.

- **`eyeguard_ui.py`** — The CustomTkinter GUI and application glue.
  - `WebcamWorker` (background thread): owns the OpenCV capture and
    MediaPipe Face Mesh instance, runs blink/distance detection per frame,
    and pushes `(frame, metrics)` onto a small bounded queue. It never
    touches GUI widgets directly.
  - `EyeGuardApp` (main thread): polls that queue on a `self.after(30, ...)`
    timer to update the video feed and dashboard metrics, so webcam
    processing speed never blocks or freezes the interface. It also wires
    up alerts (voice + notification, cooldown-limited), the 20-20-20 break
    timer with a dismiss/snooze banner, and the AI Coach panel.
  - `ExerciseReminderWindow`: popup shown when a Gemini-recommended
    exercise is due, with Start/Snooze/Skip and a live countdown.
  - Gemini calls are always dispatched via `threading.Thread` and results
    are delivered back through a queue drained by the same polling timer —
    the Gemini API is never called on the GUI thread.

- **`eyeguard_ai_coach.py`** — `AIHealthCoach.get_health_report(stats)`
  sends session statistics to Gemini with a system prompt that requests a
  fixed JSON schema (health score, analysis, recommendations, exercise,
  motivation) and explicitly instructs the model to avoid diagnostic or
  treatment claims. The response is parsed defensively: markdown fences are
  stripped, then `json.loads` is tried, then a regex fallback extracts the
  first `{...}` block if the model added extra prose. If the key is
  missing, the network call fails, or parsing still fails, a local
  rule-based `HealthReport` is generated instead so the UI is never left
  broken or empty.

- **`notifications.py`** — `NotificationManager` sends desktop
  notifications via `plyer` on a background thread and enforces a
  per-topic cooldown (default 60s) so the same alert (e.g. "blink more")
  can't spam the user repeatedly.

- **`voice.py`** — `VoiceManager` runs a dedicated worker thread that owns
  a single `pyttsx3` engine instance and speaks queued phrases, so speech
  synthesis never blocks the GUI.

- **`statistics.py`** — `SessionStats` tracks live, in-memory metrics for
  the current session (blink timestamps for rolling blinks-per-minute,
  distance samples, warning counts, exercises completed, health-score
  history). `StatisticsManager` persists a cumulative history across
  sessions to `data/statistics.json` and exposes a `get_dashboard_snapshot()`
  used both by the UI and as the input sent to the AI Coach.

- **`settings.py`** — `SettingsManager` loads/saves `data/settings.json`
  (Gemini key, voice/notification toggles, blink and distance thresholds,
  theme, break interval) with safe defaults if the file is missing or
  corrupted, and never ships with a real API key baked in.

## Testing Checklist (verified during development)

- ✅ Webcam initialization (including graceful failure if no camera is found)
- ✅ MediaPipe Face Mesh initialization
- ✅ Blink detection (EAR-based, single-count-per-blink)
- ✅ Distance classification (TOO CLOSE / GOOD DISTANCE / TOO FAR)
- ✅ Desktop notifications with cooldown (no spam)
- ✅ Voice alerts (non-blocking, degrades silently without a TTS backend)
- ✅ Gemini integration with a valid key
- ✅ Malformed/garbage/prose-wrapped Gemini responses (safe fallback, never crashes)
- ✅ Missing API key (automatic local fallback report)
- ✅ GUI responsiveness (webcam + Gemini calls run off the main thread)
- ✅ Exercise reminders (scheduled after the AI-recommended delay, with Start/Snooze/Skip)
