from __future__ import annotations
"""
eyeguard_ui.py
---------------
The CustomTkinter GUI for EyeGuard AI. Ties together the webcam worker
thread, the blink/distance detectors, notifications, voice alerts, the
AI Health Coach, and local statistics into one dashboard.

Threading model
----------------
- WebcamWorker runs on its own daemon thread, continuously reading frames,
  running MediaPipe Face Mesh, and pushing (frame, metrics) tuples onto a
  bounded queue. It NEVER touches CustomTkinter widgets directly.
- The main thread polls that queue on a `self.after(...)` timer (~33ms)
  and is the only thread that ever updates widgets - this keeps the GUI
  responsive regardless of webcam/processing speed.
- Gemini API calls (AIHealthCoach.get_health_report) are blocking, so they
  are always dispatched on a separate background thread; the result is
  handed back to the main thread through a small thread-safe queue that is
  also drained by the same polling timer.
"""


import queue
import threading
import time
import os
from tkinter import filedialog
from typing import Optional

import cv2
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image

from distance_detector import DistanceDetector, DistanceStatus
from eye_detector import EyeDetector
from posture_detector import PostureDetector, PostureStatus
from eyeguard_ai_coach import AIHealthCoach, HealthReport
from notifications import NotificationManager
from settings import SettingsManager
from statistics import StatisticsManager
from voice import VoiceManager
from windows_magnifier import WindowsMagnifier
from i18n import t
from tray import TrayController
from ui_i18n import UITranslator

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

VIDEO_WIDTH = 480
VIDEO_HEIGHT = 360
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Assets")

ACCENT_COLORS = {
    "blue": {"fg": "#2563eb", "hover": "#1d4ed8"},
    "pink": {"fg": "#db2777", "hover": "#be185d"},
}

BLOCK_COLORS = {
    "ground": ("#eef2f7", "#181c20"),
    "stone": ("#d7dde5", "#30363d"),
    "moss": ("#4e9a55", "#2f6b38"),
    "gold": ("#d6a824", "#c08e14"),
    "ink": ("#17212b", "#f2f5f7"),
    "muted": ("#52616f", "#aeb8c2"),
}


# --------------------------------------------------------------------------
# Background webcam + MediaPipe worker
# --------------------------------------------------------------------------

class WebcamWorker(threading.Thread):
    """Reads webcam frames, runs Face Mesh + detectors, and queues results."""

    def __init__(
        self,
        settings_manager: SettingsManager,
        frame_queue: "queue.Queue",
        event_queue: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.settings_manager = settings_manager
        self.frame_queue = frame_queue
        self.event_queue = event_queue
        self.stop_event = stop_event

        settings = settings_manager.get_all()
        self.eye_detector = EyeDetector(ear_threshold=0.21, consec_frames=2)
        self.distance_detector = DistanceDetector(
            too_close_ratio=settings.distance_too_close_ratio,
            too_far_ratio=settings.distance_too_far_ratio,
        )
        self.posture_detector = PostureDetector()
        self._calibration_requested = threading.Event()

    def request_posture_calibration(self) -> None:
        self._calibration_requested.set()

    def run(self) -> None:
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            mp_pose = mp.solutions.pose
        except Exception as exc:
            self.event_queue.put((
                "camera_error",
                "MediaPipe failed to load correctly "
                f"({exc}). This usually means the installed Python version "
                "isn't yet supported by MediaPipe (it currently supports "
                "Python 3.9-3.12), or the mediapipe package didn't install "
                "correctly. Try running EyeGuard AI with Python 3.12 in a "
                "fresh virtual environment and reinstall requirements.txt.",
            ))
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.event_queue.put(("camera_error", "Could not open webcam (device 0)."))
            return

        self.event_queue.put(("camera_ready", None))

        try:
            with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as face_mesh, mp_pose.Pose(
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as pose:
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        time.sleep(0.05)
                        continue

                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = face_mesh.process(rgb)
                    pose_results = pose.process(rgb)

                    metrics = {
                        "face_detected": False,
                        "ear": 0.0,
                        "is_eye_closed": False,
                        "blink_just_registered": False,
                        "distance_status": DistanceStatus.UNKNOWN.value,
                        "distance_ratio": 0.0,
                        "tracking_confidence": "Low confidence",
                        "lighting_quality": "Low lighting detected." if brightness < 45 else "Adequate lighting",
                        "posture_status": PostureStatus.UNKNOWN.value,
                        "posture_guidance": "Keep your head and shoulders in view",
                        "posture_warning_just_registered": False,
                    }

                    if results.multi_face_landmarks and brightness >= 45:
                        landmarks = results.multi_face_landmarks[0].landmark
                        blink_result = self.eye_detector.process(landmarks, w, h)
                        distance_result = self.distance_detector.estimate(landmarks, w, h)

                        metrics.update({
                            "face_detected": True,
                            "ear": blink_result.ear,
                            "is_eye_closed": blink_result.is_eye_closed,
                            "blink_just_registered": blink_result.blink_just_registered,
                            "distance_status": distance_result.status.value,
                            "distance_ratio": distance_result.interocular_ratio,
                            "tracking_confidence": "High confidence" if brightness >= 45 else "Medium confidence",
                        })
                    elif results.multi_face_landmarks:
                        metrics["lighting_quality"] = "Low lighting detected. Measurements paused."

                    if pose_results.pose_landmarks and brightness >= 45:
                        posture_landmarks = pose_results.pose_landmarks.landmark
                        if self._calibration_requested.is_set():
                            self._calibration_requested.clear()
                            if self.posture_detector.calibrate(posture_landmarks):
                                self.event_queue.put(("posture_calibrated", None))
                            else:
                                self.event_queue.put(("posture_calibration_error", None))
                        posture_result = self.posture_detector.process(posture_landmarks)
                        self._draw_posture_overlay(frame, posture_landmarks, posture_result)
                        metrics.update({
                            "posture_status": posture_result.status.value,
                            "posture_guidance": posture_result.guidance,
                            "posture_warning_just_registered": posture_result.warning_just_registered,
                            "posture_calibrated": self.posture_detector.is_calibrated,
                        })

                    self._push_frame(frame, metrics)
                    # Small yield keeps CPU usage reasonable without hurting responsiveness
                    time.sleep(0.005)
        except Exception as exc:
            self.event_queue.put(("camera_error", f"Webcam processing stopped: {exc}"))
        finally:
            cap.release()

    def _push_frame(self, frame, metrics: dict) -> None:
        """Keep only the freshest frame in the queue so the UI never lags behind."""
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait((frame, metrics))
        except queue.Full:
            pass

    @staticmethod
    def _draw_posture_overlay(frame, landmarks, posture_result) -> None:
        indices = (PostureDetector.LEFT_EAR, PostureDetector.RIGHT_EAR,
                   PostureDetector.LEFT_SHOULDER, PostureDetector.RIGHT_SHOULDER)
        height, width = frame.shape[:2]
        points = {
            index: (int(landmarks[index].x * width), int(landmarks[index].y * height))
            for index in indices
        }
        color = (75, 180, 80) if posture_result.status == PostureStatus.ALIGNED else (65, 90, 235)
        cv2.line(frame, points[PostureDetector.LEFT_SHOULDER], points[PostureDetector.RIGHT_SHOULDER], color, 3)
        cv2.line(frame, points[PostureDetector.LEFT_EAR], points[PostureDetector.RIGHT_EAR], color, 2)
        for point in points.values():
            cv2.circle(frame, point, 5, color, -1)

    def update_distance_thresholds(self, too_close: float, too_far: float) -> None:
        self.distance_detector.update_thresholds(too_close, too_far)


# --------------------------------------------------------------------------
# Exercise reminder popup
# --------------------------------------------------------------------------

class ExerciseReminderWindow(ctk.CTkToplevel):
    """Popup shown when it's time for the AI-recommended eye exercise."""

    def __init__(self, parent: "EyeGuardApp", exercise_name: str, instructions: str,
                 duration_seconds: int) -> None:
        super().__init__(parent)
        self.parent = parent
        self.exercise_name = exercise_name
        self.instructions = instructions
        self.duration_seconds = duration_seconds
        self._countdown_job: Optional[str] = None
        self._remaining = duration_seconds

        self.title("EyeGuard AI - Exercise Reminder")
        self.geometry("400x300")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        ctk.CTkLabel(
            self, text="Time for your recommended eye exercise!",
            font=ctk.CTkFont(size=16, weight="bold"), wraplength=360,
        ).pack(pady=(20, 10), padx=20)

        ctk.CTkLabel(self, text=exercise_name, font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(0, 5))
        ctk.CTkLabel(self, text=instructions, wraplength=360, justify="center").pack(pady=(0, 15), padx=20)

        self.countdown_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=28, weight="bold"))
        self.countdown_label.pack(pady=(0, 15))

        self.button_row = ctk.CTkFrame(self, fg_color="transparent")
        self.button_row.pack(pady=10)

        self.start_btn = ctk.CTkButton(self.button_row, text="Start Exercise", command=self._start_exercise)
        self.start_btn.grid(row=0, column=0, padx=5)
        self.snooze_btn = ctk.CTkButton(self.button_row, text="Snooze 5 Min", command=self._snooze)
        self.snooze_btn.grid(row=0, column=1, padx=5)
        self.skip_btn = ctk.CTkButton(self.button_row, text="Skip", fg_color="gray40", command=self._skip)
        self.skip_btn.grid(row=0, column=2, padx=5)

    def _start_exercise(self) -> None:
        self.start_btn.configure(state="disabled")
        self.snooze_btn.configure(state="disabled")
        self.parent.voice_manager.speak(f"Starting {self.exercise_name}. {self.instructions}")
        self._remaining = self.duration_seconds
        self._tick()

    def _tick(self) -> None:
        self.countdown_label.configure(text=f"{self._remaining}s")
        if self._remaining <= 0:
            self.countdown_label.configure(text="Done! Great job.")
            self.parent.voice_manager.speak("Exercise complete. Great job.")
            self.parent.stats_manager.register_exercise_completed()
            self.after(1500, self.destroy)
            return
        self._remaining -= 1
        self._countdown_job = self.after(1000, self._tick)

    def _snooze(self) -> None:
        self.parent.schedule_exercise_reminder(
            self.exercise_name, self.instructions, self.duration_seconds, delay_minutes=5
        )
        self.destroy()

    def _skip(self) -> None:
        self.destroy()


# --------------------------------------------------------------------------
# Main application window
# --------------------------------------------------------------------------

class EyeGuardApp(ctk.CTk):
    """Top-level EyeGuard AI application window."""

    def __init__(self) -> None:
        super().__init__()

        self.settings_manager = SettingsManager()
        self.stats_manager = StatisticsManager()
        settings = self.settings_manager.get_all()
        if not t.set_language(settings.language):
            t.set_language(t.default_lang)

        self.notification_manager = NotificationManager(enabled=settings.notifications_enabled)
        self.notification_manager.set_focus_mode(settings.focus_mode)
        self.voice_manager = VoiceManager(enabled=settings.voice_alerts_enabled)
        self.ai_coach = AIHealthCoach(api_key=settings.gemini_api_key)

        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=2)
        self.event_queue: "queue.Queue" = queue.Queue()
        self.ai_result_queue: "queue.Queue" = queue.Queue()
        self.stop_event = threading.Event()
        self.webcam_worker: Optional[WebcamWorker] = None
        self.monitoring_active = False

        self._latest_health_report: Optional[HealthReport] = None
        self._break_job: Optional[str] = None
        self._break_countdown_job: Optional[str] = None
        self._break_session_job: Optional[str] = None
        self._break_reminder_active = False
        self._next_break_epoch: Optional[float] = None
        self._last_metric_sample_epoch = 0.0
        self._session_persisted = False
        self._exit_requested = False
        self._tray = TrayController(
            on_open=lambda: self.event_queue.put(("tray_open", None)),
            on_pause=lambda: self.event_queue.put(("tray_pause", None)),
            on_resume=lambda: self.event_queue.put(("tray_resume", None)),
            on_ai_coach=lambda: self.event_queue.put(("tray_tab", "ai_coach")),
            on_settings=lambda: self.event_queue.put(("tray_tab", "settings")),
            on_exit=lambda: self.event_queue.put(("tray_exit", None)),
        )
        self.notification_manager.set_tray(self._tray)
        self._magnifier = WindowsMagnifier()

        self.title(t("EyeGuard AI - Digital Eye Wellness Assistant"))
        self.geometry("1180x720")
        self.minsize(1000, 640)
        ctk.set_appearance_mode(settings.theme)

        self._status = ("Status: Idle", {})
        self._build_layout()
        self._ui_tr = UITranslator(self)
        self._show_tab("dashboard")
        self._schedule_break_reminder()
        self.after(30, self._poll_queues)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tray.start()

    # ---------------------------------------------------------------- UI

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(self, width=190, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_rowconfigure(13, weight=1)

        ctk.CTkLabel(
            sidebar, text="EyeGuard AI", font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=(24, 4), sticky="w")
        ctk.CTkLabel(
            sidebar, text="Digital Eye Wellness", font=ctk.CTkFont(size=11), text_color="gray60"
        ).grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        appearance_controls = ctk.CTkFrame(sidebar, fg_color="transparent")
        appearance_controls.grid(row=2, column=0, padx=15, pady=(0, 12), sticky="ew")
        self.quick_appearance_switch = ctk.CTkSwitch(
            appearance_controls, text="Dark mode", command=self._toggle_appearance
        )
        if self.settings_manager.get("theme") == "dark":
            self.quick_appearance_switch.select()
        self.quick_appearance_switch.pack(anchor="w")
        self.quick_accent_selector = ctk.CTkSegmentedButton(
            appearance_controls, values=["Blue", "Pink"], command=self._set_accent_theme
        )
        self.quick_accent_selector.set(self.settings_manager.get("accent_color").title())
        self.quick_accent_selector.pack(fill="x", pady=(8, 0))
        self.language_btn = ctk.CTkButton(
            appearance_controls, text=self._language_button_text(), command=self._next_language
        )
        self.language_btn.pack(fill="x", pady=(8, 0))

        self.nav_buttons = {}
        nav_items = [
            ("dashboard", "Dashboard"),
            ("ai_coach", "AI Health Coach"),
            ("exercises", "Exercises"),
            ("statistics", "Statistics"),
            ("timeline", "Timeline"),
            ("achievements", "Achievements"),
            ("notifications", "Notifications"),
            ("privacy", "Privacy"),
            ("settings", "Settings"),
        ]
        for i, (key, label) in enumerate(nav_items, start=4):
            btn = ctk.CTkButton(
                sidebar, text=label, anchor="w",
                fg_color="transparent", command=lambda k=key: self._show_tab(k),
            )
            btn.grid(row=i, column=0, padx=15, pady=4, sticky="ew")
            self.nav_buttons[key] = btn

        self.toggle_btn = ctk.CTkButton(
            sidebar, text="Start Monitoring", fg_color="#2fa572",
            command=self._toggle_monitoring,
        )
        self.toggle_btn.grid(row=14, column=0, padx=15, pady=(10, 20), sticky="ew")

        self.status_label = ctk.CTkLabel(sidebar, text="Status: Idle", text_color="gray60")
        self.status_label.grid(row=15, column=0, padx=15, pady=(0, 20), sticky="w")

        # Content container - one frame per tab, only one gridded at a time
        self.content = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self.tabs = {
            "dashboard": self._build_dashboard_tab(self.content),
            "ai_coach": self._build_ai_coach_tab(self.content),
            "exercises": self._build_exercises_tab(self.content),
            "statistics": self._build_statistics_tab(self.content),
            "timeline": self._build_timeline_tab(self.content),
            "achievements": self._build_achievements_tab(self.content),
            "notifications": self._build_notifications_tab(self.content),
            "privacy": self._build_privacy_tab(self.content),
            "settings": self._build_settings_tab(self.content),
        }
        for frame in self.tabs.values():
            frame.grid(row=0, column=0, sticky="nswe")
        self._apply_accent_theme()

    def _show_tab(self, key: str) -> None:
        for name, frame in self.tabs.items():
            frame.tkraise() if name == key else None
        for name, btn in self.nav_buttons.items():
            btn.configure(fg_color=("gray75", "gray25") if name == key else "transparent")
        if key == "statistics":
            self._refresh_statistics_tab()
        elif key == "timeline":
            self._refresh_timeline_tab()
        elif key == "achievements":
            self._refresh_achievements_tab()
        elif key == "notifications":
            self._refresh_notifications_tab()
        elif key == "privacy":
            self._refresh_privacy_tab()
        self._ui_tr.refresh()

    # ---- Language switching ----

    def _language_button_text(self) -> str:
        return f"{t('Language')}: {t.language_name()}"

    def _next_language(self) -> None:
        t.next_language()
        self.settings_manager.set("language", t.current_lang)
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
        self.deiconify()
        self.lift()
        self.focus_force()

    def _toggle_appearance(self) -> None:
        mode = "dark" if self.quick_appearance_switch.get() else "light"
        self._set_appearance_mode(mode)

    def _set_appearance_mode(self, value: str) -> None:
        mode = value.lower()
        if mode not in ("dark", "light"):
            return
        ctk.set_appearance_mode(mode)
        self.settings_manager.set("theme", mode)
        if hasattr(self, "quick_appearance_switch"):
            if mode == "dark":
                self.quick_appearance_switch.select()
            else:
                self.quick_appearance_switch.deselect()
        if hasattr(self, "theme_selector"):
            self.theme_selector.set(mode.title())

    def _set_accent_theme(self, value: str) -> None:
        accent = value.lower()
        if accent not in ACCENT_COLORS:
            return
        self.settings_manager.set("accent_color", accent)
        if hasattr(self, "quick_accent_selector"):
            self.quick_accent_selector.set(accent.title())
        if hasattr(self, "accent_selector"):
            self.accent_selector.set(accent.title())
        self._apply_accent_theme()

    def _apply_accent_theme(self) -> None:
        palette = ACCENT_COLORS[self.settings_manager.get("accent_color")]
        for widget_name in ("quick_accent_selector", "accent_selector"):
            if hasattr(self, widget_name):
                getattr(self, widget_name).configure(
                    selected_color=palette["fg"], selected_hover_color=palette["hover"]
                )
        if hasattr(self, "quick_appearance_switch"):
            self.quick_appearance_switch.configure(progress_color=palette["fg"])
        if hasattr(self, "toggle_btn") and not self.monitoring_active:
            self._configure_start_monitor_button()

    def _configure_start_monitor_button(self) -> None:
        palette = ACCENT_COLORS[self.settings_manager.get("accent_color")]
        self.toggle_btn.configure(text=t("Start Monitoring"), fg_color=palette["fg"], hover_color=palette["hover"])

    def _request_posture_calibration(self) -> None:
        if not self.monitoring_active or self.webcam_worker is None:
            self.posture_guidance_label.configure(text=t("Start monitoring, sit comfortably, then calibrate"))
            return
        self.webcam_worker.request_posture_calibration()
        self.posture_guidance_label.configure(text=t("Capturing your comfortable posture reference..."))

    # ---- Dashboard tab ----

    def _build_dashboard_tab(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=3)
        frame.grid_columnconfigure(1, weight=2)
        frame.grid_rowconfigure(0, weight=1)

        # Video panel
        video_panel = ctk.CTkFrame(frame)
        video_panel.grid(row=0, column=0, sticky="nswe", padx=(0, 10))
        self.video_label = ctk.CTkLabel(video_panel, text="Camera feed will appear here\nwhen monitoring starts",
                                         width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.video_label.pack(padx=15, pady=15)

        self.break_banner = ctk.CTkFrame(video_panel, fg_color="#8a5a00", corner_radius=8)
        self.break_label = ctk.CTkLabel(self.break_banner, text="", font=ctk.CTkFont(size=14, weight="bold"))
        self.break_label.pack(side="left", padx=15, pady=10)
        self.break_start_btn = ctk.CTkButton(self.break_banner, text="Start Break", width=100,
                             command=self._start_break)
        self.break_start_btn.pack(side="right", padx=5, pady=10)
        ctk.CTkButton(self.break_banner, text="Dismiss", width=80,
                      command=self._dismiss_break_reminder).pack(side="right", padx=(5, 15), pady=10)
        ctk.CTkButton(self.break_banner, text="Snooze 5m", width=90,
                      command=self._snooze_break_reminder).pack(side="right", padx=5, pady=10)
        # break_banner is packed/unpacked dynamically, not shown by default

        # Metrics panel
        metrics_panel = ctk.CTkFrame(frame)
        metrics_panel.grid(row=0, column=1, sticky="nswe")

        ctk.CTkLabel(metrics_panel, text="Live Dashboard", font=ctk.CTkFont(size=18, weight="bold")).pack(
            anchor="w", padx=20, pady=(20, 10))

        self.blink_count_label = self._metric_row(metrics_panel, "Total Blinks", "0")
        self.blink_rate_label = self._metric_row(metrics_panel, "Blinks / Minute", "0.0")
        self.blink_status_label = self._metric_row(metrics_panel, "Eye Status", "—")
        self.distance_status_label = self._metric_row(metrics_panel, "Screen Distance", "—")
        self.posture_status_label = self._metric_row(metrics_panel, "Posture", "—")
        self.posture_guidance_label = ctk.CTkLabel(
            metrics_panel, text="Start monitoring to check alignment.", wraplength=300,
            justify="left", text_color="gray60",
        )
        self.posture_guidance_label.pack(anchor="w", padx=20, pady=(0, 4))
        self.posture_balance_label = self._metric_row(metrics_panel, "Posture Balance", "—")
        self.calibrate_posture_btn = ctk.CTkButton(
            metrics_panel, text="Calibrate Posture", width=160, command=self._request_posture_calibration
        )
        self.calibrate_posture_btn.pack(anchor="w", padx=20, pady=(2, 8))
        self.tracking_confidence_label = self._metric_row(metrics_panel, "Tracking Quality", "Low confidence")
        self.lighting_label = self._metric_row(metrics_panel, "Lighting", "—")
        self.session_time_label = self._metric_row(metrics_panel, "Session Duration", "00:00")
        self.wellness_score_label = self._metric_row(metrics_panel, "Digital Eye Wellness Score", "-- / 100")
        self.wellness_factors_label = ctk.CTkLabel(metrics_panel, text="Score factors will appear after monitoring.",
                               wraplength=300, justify="left", text_color="gray60")
        self.wellness_factors_label.pack(anchor="w", padx=20, pady=(2, 8))

        ctk.CTkLabel(
            metrics_panel,
            text="Distance and posture guidance are camera estimates, not medical measurements.",
            font=ctk.CTkFont(size=11), text_color="gray60", justify="left",
        ).pack(anchor="w", padx=20, pady=(10, 5))

        ctk.CTkLabel(metrics_panel, text="20-20-20 Break Timer", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=20, pady=(20, 5))
        self.next_break_label = ctk.CTkLabel(metrics_panel, text="Next break in --:--", font=ctk.CTkFont(size=13))
        self.next_break_label.pack(anchor="w", padx=20, pady=(0, 20))

        return frame

    def _metric_row(self, parent, title: str, initial_value: str) -> ctk.CTkLabel:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=6)
        ctk.CTkLabel(row, text=title, font=ctk.CTkFont(size=13), text_color="gray70").pack(side="left")
        value_label = ctk.CTkLabel(row, text=initial_value, font=ctk.CTkFont(size=13, weight="bold"))
        value_label.pack(side="right")
        return value_label

    # ---- AI Coach tab ----

    def _build_ai_coach_tab(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(header, text="AI Health Coach", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        self.ai_analyze_btn = ctk.CTkButton(header, text="Get AI Analysis", command=self._request_ai_analysis)
        self.ai_analyze_btn.pack(side="right")

        card = ctk.CTkFrame(frame)
        card.pack(fill="both", expand=True, padx=10, pady=10)

        score_row = ctk.CTkFrame(card, fg_color="transparent")
        score_row.pack(fill="x", padx=20, pady=(20, 10))
        ctk.CTkLabel(score_row, text="Digital Eye Wellness Score", font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        self.health_score_label = ctk.CTkLabel(score_row, text="--", font=ctk.CTkFont(size=22, weight="bold"))
        self.health_score_label.pack(side="right")
        self.health_score_bar = ctk.CTkProgressBar(card)
        self.health_score_bar.set(0)
        self.health_score_bar.pack(fill="x", padx=20, pady=(0, 15))

        ctk.CTkLabel(card, text="Analysis", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=20)
        self.analysis_label = ctk.CTkLabel(
            card, text="Click \"Get AI Analysis\" after monitoring for a while to see your report.",
            wraplength=650, justify="left",
        )
        self.analysis_label.pack(anchor="w", padx=20, pady=(2, 15))

        priority_frame = ctk.CTkFrame(card, fg_color=("gray90", "gray18"))
        priority_frame.pack(fill="x", padx=20, pady=(0, 15))
        ctk.CTkLabel(priority_frame, text="Personalized priorities", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=15, pady=(10, 5))
        self.top_priority_label = ctk.CTkLabel(priority_frame, text="Top priority: —", wraplength=650, justify="left")
        self.top_priority_label.pack(anchor="w", padx=15, pady=2)
        self.secondary_priority_label = ctk.CTkLabel(priority_frame, text="Secondary: —", wraplength=650, justify="left")
        self.secondary_priority_label.pack(anchor="w", padx=15, pady=2)
        self.going_well_label = ctk.CTkLabel(priority_frame, text="Going well: —", wraplength=650, justify="left")
        self.going_well_label.pack(anchor="w", padx=15, pady=(2, 10))
        self.baseline_label = ctk.CTkLabel(
            card, text="Your baseline: collecting more real sessions...", wraplength=650, justify="left",
            text_color="gray60",
        )
        self.baseline_label.pack(anchor="w", padx=20, pady=(0, 12))

        ctk.CTkLabel(card, text="Recommendations", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=20)
        self.recommendations_label = ctk.CTkLabel(card, text="—", wraplength=650, justify="left")
        self.recommendations_label.pack(anchor="w", padx=20, pady=(2, 15))

        exercise_frame = ctk.CTkFrame(card, fg_color=("gray90", "gray20"))
        exercise_frame.pack(fill="x", padx=20, pady=(0, 15))
        ctk.CTkLabel(exercise_frame, text="Recommended Exercise", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=15, pady=(12, 2))
        self.exercise_name_label = ctk.CTkLabel(exercise_frame, text="—", font=ctk.CTkFont(size=13, weight="bold"))
        self.exercise_name_label.pack(anchor="w", padx=15)
        self.exercise_instructions_label = ctk.CTkLabel(exercise_frame, text="—", wraplength=630, justify="left")
        self.exercise_instructions_label.pack(anchor="w", padx=15, pady=(2, 2))
        self.exercise_duration_label = ctk.CTkLabel(exercise_frame, text="Duration: —")
        self.exercise_duration_label.pack(anchor="w", padx=15)
        self.next_reminder_label = ctk.CTkLabel(exercise_frame, text="Next reminder: —")
        self.next_reminder_label.pack(anchor="w", padx=15, pady=(0, 12))

        ctk.CTkLabel(card, text="Motivation", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=20)
        self.motivation_label = ctk.CTkLabel(card, text="—", wraplength=650, justify="left",
                                              text_color="#3fa9f5")
        self.motivation_label.pack(anchor="w", padx=20, pady=(2, 5))

        ctk.CTkLabel(
            card,
            text="EyeGuard AI's coach offers general wellness suggestions only and does\n"
                 "not diagnose or treat any medical condition. Consult an eye-care\n"
                 "professional for persistent discomfort.",
            font=ctk.CTkFont(size=10), text_color="gray60", justify="left",
        ).pack(anchor="w", padx=20, pady=(15, 15))

        chat = ctk.CTkFrame(card, fg_color=("gray90", "gray18"))
        chat.pack(fill="x", padx=20, pady=(5, 15))
        ctk.CTkLabel(chat, text="Ask your coach", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=15, pady=(12, 5))
        self.chat_history = ctk.CTkTextbox(chat, height=90, wrap="word")
        self.chat_history.pack(fill="x", padx=15, pady=(0, 8))
        self.chat_history.insert("1.0", "Coach: Ask about your screen habits, blink rate, or exercises.\n")
        self.chat_history.configure(state="disabled")
        chat_row = ctk.CTkFrame(chat, fg_color="transparent")
        chat_row.pack(fill="x", padx=15, pady=(0, 12))
        self.chat_entry = ctk.CTkEntry(chat_row, placeholder_text="How can I improve my screen habits?")
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(chat_row, text="Ask", width=80, command=self._ask_coach_question).pack(side="right")

        return frame

    def _build_text_page(self, parent, title: str, initial_text: str) -> ctk.CTkTextbox:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=20, weight="bold")).pack(
            anchor="w", padx=10, pady=(10, 10))
        text = ctk.CTkTextbox(frame, wrap="word", font=ctk.CTkFont(size=13))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert("1.0", initial_text)
        text.configure(state="disabled")
        frame._eyeguard_text = text
        return frame

    def _set_page_text(self, page, text: str) -> None:
        widget = page._eyeguard_text
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _build_timeline_tab(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        header = ctk.CTkFrame(frame, corner_radius=0, fg_color=BLOCK_COLORS["stone"])
        header.pack(fill="x", padx=10, pady=(10, 8))
        ctk.CTkLabel(header, text="Session Timeline", font=ctk.CTkFont(size=20, weight="bold")).pack(
            side="left", padx=16, pady=12
        )
        self.timeline_summary_label = ctk.CTkLabel(header, text="0 events", text_color=BLOCK_COLORS["muted"])
        self.timeline_summary_label.pack(side="right", padx=16, pady=12)

        timeline_panel = ctk.CTkScrollableFrame(frame, corner_radius=0, fg_color=BLOCK_COLORS["ground"])
        timeline_panel.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.timeline_event_list = timeline_panel
        return frame

    def _build_achievements_tab(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        header = ctk.CTkFrame(frame, corner_radius=0, fg_color=BLOCK_COLORS["stone"])
        header.pack(fill="x", padx=10, pady=(10, 8))
        ctk.CTkLabel(header, text="Achievements", font=ctk.CTkFont(size=20, weight="bold")).pack(
            side="left", padx=16, pady=12
        )
        self.achievement_count_label = ctk.CTkLabel(header, text="0 unlocked", text_color=BLOCK_COLORS["muted"])
        self.achievement_count_label.pack(side="right", padx=16, pady=12)

        self.achievement_grid = ctk.CTkScrollableFrame(frame, corner_radius=0, fg_color=BLOCK_COLORS["ground"])
        self.achievement_grid.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.achievement_grid.grid_columnconfigure(0, weight=1)
        self.achievement_grid.grid_columnconfigure(1, weight=1)
        return frame

    def _build_notifications_tab(self, parent):
        return self._build_text_page(parent, "Notification History", "No notifications recorded yet.")

    def _build_privacy_tab(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(frame, text="Privacy", font=ctk.CTkFont(size=20, weight="bold")).pack(
            anchor="w", padx=10, pady=(10, 10)
        )
        card = ctk.CTkFrame(frame)
        card.pack(fill="x", padx=10, pady=10)
        self.privacy_camera_status_label = ctk.CTkLabel(
            card, text="Camera: Off", font=ctk.CTkFont(size=16, weight="bold")
        )
        self.privacy_camera_status_label.pack(anchor="w", padx=20, pady=(20, 8))
        self.privacy_pause_btn = ctk.CTkButton(card, text="Start Camera", command=self._toggle_monitoring)
        self.privacy_pause_btn.pack(anchor="w", padx=20, pady=(0, 18))
        ctk.CTkLabel(
            card,
            text=(
                "Camera frames are processed on this device and are never saved or sent to an AI service.\n\n"
                "The optional AI coach receives only numerical session statistics. Settings and statistics stay on this device."
            ),
            justify="left", wraplength=700, text_color="gray60",
        ).pack(anchor="w", padx=20, pady=(0, 20))
        return frame

    def _refresh_privacy_tab(self) -> None:
        if self.monitoring_active:
            self.privacy_camera_status_label.configure(text=t("Camera: On - local processing"))
            self.privacy_pause_btn.configure(text=t("Pause Camera"))
        else:
            self.privacy_camera_status_label.configure(text=t("Camera: Off"))
            self.privacy_pause_btn.configure(text=t("Start Camera"))

    def _refresh_timeline_tab(self) -> None:
        events = self.stats_manager.get_timeline_events()
        for child in self.timeline_event_list.winfo_children():
            child.destroy()

        recent_events = events[-60:]
        self.timeline_summary_label.configure(
            text=f"{len(events)} events recorded" if events else "Waiting for activity"
        )
        if not recent_events:
            ctk.CTkLabel(
                self.timeline_event_list, text="No events yet", font=ctk.CTkFont(size=16, weight="bold")
            ).pack(anchor="w", padx=18, pady=(24, 4))
            ctk.CTkLabel(
                self.timeline_event_list, text="Monitoring activity will appear here as a readable session log.",
                         text_color=BLOCK_COLORS["muted"]
            ).pack(anchor="w", padx=18, pady=(0, 24))
            return

        topic_labels = {
            "low_blink_rate": "Blink reminder",
            "too_close": "Screen distance reminder",
            "posture_adjustment": "Posture reminder",
        }
        current_day = None
        for item in reversed(recent_events):
            event_day = time.strftime("%A, %d %b", time.localtime(item["timestamp"]))
            if event_day != current_day:
                current_day = event_day
                ctk.CTkLabel(
                    self.timeline_event_list, text=current_day, font=ctk.CTkFont(size=13, weight="bold"),
                    text_color=BLOCK_COLORS["muted"],
                ).pack(anchor="w", padx=12, pady=(16, 6))

            event_card = ctk.CTkFrame(self.timeline_event_list, corner_radius=0, fg_color=BLOCK_COLORS["stone"])
            event_card.pack(fill="x", padx=12, pady=3)
            indicator = ctk.CTkFrame(event_card, width=8, height=42, corner_radius=0, fg_color=BLOCK_COLORS["gold"])
            indicator.pack(side="left", fill="y")
            ctk.CTkLabel(
                event_card, text=topic_labels.get(item["topic"], "Wellness event"),
                font=ctk.CTkFont(size=13, weight="bold"),
            ).pack(side="left", padx=12, pady=10)
            ctk.CTkLabel(
                event_card, text=time.strftime("%H:%M", time.localtime(item["timestamp"])),
                text_color=BLOCK_COLORS["muted"],
            ).pack(side="right", padx=12, pady=10)

    def _refresh_achievements_tab(self) -> None:
        for child in self.achievement_grid.winfo_children():
            child.destroy()

        cumulative = self.stats_manager.get_cumulative_data()
        unlocked = set(self.stats_manager.get_achievements())
        total_sessions = int(cumulative.get("total_sessions", 0))
        completed_posture_sessions = sum(
            1 for session in cumulative.get("session_history", [])
            if (session.get("posture_alignment_rate") or 0) >= 80
        )
        total_breaks = int(cumulative.get("total_breaks_taken", 0))
        total_exercises = int(cumulative.get("total_exercises_completed", 0))
        definitions = [
            ("First Session", "Complete your first monitoring session", total_sessions, 1, "Pickaxe"),
            ("Session Master", "Finish 25 monitoring sessions", total_sessions, 25, "Diamond"),
            ("Touch Grass", "Take your first screen break", total_breaks, 1, "Grass_block"),
            ("Break Builder", "Take 10 screen breaks", total_breaks, 10, "Grass_block"),
            ("Break Master", "Take 50 screen breaks", total_breaks, 50, "Golden_apple"),
            ("Exercise Explorer", "Complete 10 guided exercises", total_exercises, 10, "Sword"),
            ("Exercise Master", "Complete 30 guided exercises", total_exercises, 30, "Diamond"),
            ("Tracking Streak", "Finish 7 monitoring sessions", total_sessions, 7, "Pickaxe"),
            ("Consistency Master", "Finish 30 monitoring sessions", total_sessions, 30, "Diamond"),
            ("Blink Boost", "Improve blink habits in a later session", 1 if "Improved Blink Habits" in unlocked else 0, 1, "Heart"),
            ("Posture Guardian", "Maintain 80% posture balance in 5 sessions", completed_posture_sessions, 5, "Golden_apple"),
        ]
        unlocked_count = sum(1 for _title, _text, progress, target, _icon in definitions if progress >= target)
        self.achievement_count_label.configure(text=f"{unlocked_count} / {len(definitions)} unlocked")

        for index, (title, description, progress, target, icon_name) in enumerate(definitions):
            complete = progress >= target
            tile = ctk.CTkFrame(
                self.achievement_grid, corner_radius=0,
                fg_color=BLOCK_COLORS["moss"] if complete else BLOCK_COLORS["stone"],
                border_width=2, border_color=BLOCK_COLORS["gold"] if complete else BLOCK_COLORS["muted"],
            )
            tile.grid(row=index // 2, column=index % 2, sticky="nsew", padx=8, pady=8)
            tile.grid_columnconfigure(0, weight=1)
            title_row = ctk.CTkFrame(tile, fg_color="transparent")
            title_row.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 3))
            icon = self._get_achievement_icon(icon_name)
            if icon is not None:
                icon_label = ctk.CTkLabel(title_row, text="", image=icon, width=34, height=34)
                icon_label.image = icon
                icon_label.pack(side="left", padx=(0, 8))
            ctk.CTkLabel(title_row, text=title, font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
            ctk.CTkLabel(tile, text=description, wraplength=260, justify="left", text_color=BLOCK_COLORS["muted"]).grid(
                row=1, column=0, sticky="w", padx=14, pady=(0, 10)
            )
            progress_bar = ctk.CTkProgressBar(
                tile, height=10, corner_radius=0,
                progress_color=BLOCK_COLORS["gold"] if complete else BLOCK_COLORS["moss"],
            )
            progress_bar.set(min(progress / target, 1.0))
            progress_bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 4))
            ctk.CTkLabel(
                tile, text="UNLOCKED" if complete else f"{min(progress, target)} / {target}",
                font=ctk.CTkFont(size=11, weight="bold"),
            ).grid(row=3, column=0, sticky="w", padx=14, pady=(0, 14))

    def _get_achievement_icon(self, icon_name: str):
        if not hasattr(self, "_achievement_icon_cache"):
            self._achievement_icon_cache = {}
        if icon_name in self._achievement_icon_cache:
            return self._achievement_icon_cache[icon_name]
        try:
            image = Image.open(os.path.join(ASSET_DIR, f"{icon_name}.png")).resize(
                (34, 34), Image.Resampling.NEAREST
            )
            icon = ctk.CTkImage(light_image=image, dark_image=image, size=(34, 34))
            self._achievement_icon_cache[icon_name] = icon
            return icon
        except OSError:
            return None

    def _refresh_notifications_tab(self) -> None:
        items = self.notification_manager.get_history()
        text = "\n".join(
            f"{time.strftime('%H:%M', time.localtime(item['timestamp']))}  {item['topic']}: {item['message']}"
            for item in items[-100:]
        ) or "No notifications recorded yet."
        self._set_page_text(self.tabs["notifications"], text)

    def _build_exercises_tab(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(frame, text="Exercises", font=ctk.CTkFont(size=20, weight="bold")).pack(
            anchor="w", padx=10, pady=(10, 10))
        card = ctk.CTkFrame(frame)
        card.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(card, text="Recommended exercise", font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=20, pady=(25, 8))
        self.exercise_page_name = ctk.CTkLabel(card, text="Run an AI analysis to receive an exercise.",
                                                font=ctk.CTkFont(size=18, weight="bold"))
        self.exercise_page_name.pack(anchor="w", padx=20, pady=5)
        self.exercise_page_instructions = ctk.CTkLabel(card, text="", wraplength=650, justify="left")
        self.exercise_page_instructions.pack(anchor="w", padx=20, pady=5)
        self.exercise_page_countdown = ctk.CTkLabel(card, text="Ready", font=ctk.CTkFont(size=28, weight="bold"))
        self.exercise_page_countdown.pack(anchor="w", padx=20, pady=(20, 15))
        controls = ctk.CTkFrame(card, fg_color="transparent")
        controls.pack(anchor="w", padx=20, pady=10)
        ctk.CTkButton(controls, text="Start", command=self._start_exercise_page).pack(side="left", padx=(0, 8))
        ctk.CTkButton(controls, text="Pause", command=self._pause_exercise_page).pack(side="left", padx=8)
        ctk.CTkButton(controls, text="Skip", fg_color="gray40", command=self._skip_exercise_page).pack(side="left", padx=8)
        return frame

    def _start_exercise_page(self) -> None:
        if self._latest_health_report is None:
            return
        exercise = self._latest_health_report.exercise
        self._exercise_remaining = exercise.duration_seconds
        self._exercise_paused = False
        self.exercise_page_name.configure(text=exercise.name)
        self.exercise_page_instructions.configure(text=exercise.instructions)
        self._tick_exercise_page()

    def _tick_exercise_page(self) -> None:
        if getattr(self, "_exercise_paused", False):
            return
        remaining = getattr(self, "_exercise_remaining", 0)
        self.exercise_page_countdown.configure(text=f"{remaining}s")
        if remaining <= 0:
            self.stats_manager.register_exercise_completed()
            self.exercise_page_countdown.configure(text="Complete")
            return
        self._exercise_remaining -= 1
        self.after(1000, self._tick_exercise_page)

    def _pause_exercise_page(self) -> None:
        self._exercise_paused = True

    def _skip_exercise_page(self) -> None:
        self._exercise_paused = True
        self.exercise_page_countdown.configure(text="Skipped")

    def _ask_coach_question(self) -> None:
        question = self.chat_entry.get().strip()
        if not question:
            return
        self.chat_entry.delete(0, "end")
        self._append_chat(f"{t('You')}: {question}")
        snapshot = self.stats_manager.get_dashboard_snapshot()
        threading.Thread(target=self._run_coach_question, args=(question, snapshot), daemon=True).start()

    def _run_coach_question(self, question: str, snapshot: dict) -> None:
        answer = self.ai_coach.answer_question(question, snapshot)
        self.ai_result_queue.put(("chat", answer))

    def _append_chat(self, text: str) -> None:
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", text + "\n")
        self.chat_history.see("end")
        self.chat_history.configure(state="disabled")

    # ---- Statistics tab ----

    def _build_statistics_tab(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        ctk.CTkLabel(header, text="Statistics", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        self.period_menu = ctk.CTkOptionMenu(
            header, values=["Today", "7 Days", "30 Days"], command=lambda _value: self._refresh_statistics_tab()
        )
        self.period_menu.set("Today")
        self.period_menu.pack(side="right")
        self.stats_text = ctk.CTkTextbox(frame, height=130, wrap="word", font=ctk.CTkFont(size=13))
        self.stats_text.pack(fill="x", padx=10, pady=(5, 8))
        self.stats_text.configure(state="disabled")

        chart_card = ctk.CTkFrame(frame)
        chart_card.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.stats_figure = Figure(figsize=(8, 4.2), dpi=90, facecolor="#20252b")
        self.stats_axes = self.stats_figure.subplots(1, 4)
        self.stats_canvas = FigureCanvasTkAgg(self.stats_figure, master=chart_card)
        self.stats_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        ctk.CTkButton(frame, text="Export Report", width=120, command=self._export_report).pack(
            anchor="e", padx=10, pady=(0, 5))

        return frame

    def _export_report(self) -> None:
        """Export collected numerical statistics and charts without private settings."""
        path = filedialog.asksaveasfilename(
            title="Export EyeGuard Report",
            defaultextension=".pdf",
            filetypes=[("PDF report", "*.pdf")],
        )
        if not path:
            return
        cumulative = self.stats_manager.get_cumulative_data()
        trends = self.stats_manager.get_trends()
        summary = self.stats_manager.get_period_summary(1)
        with PdfPages(path) as pdf:
            text_figure = Figure(figsize=(8.27, 11.69), facecolor="white")
            axis = text_figure.add_subplot(111)
            axis.axis("off")
            report_lines = [
                "EyeGuard AI - Digital Eye Wellness Report",
                time.strftime("Date: %Y-%m-%d %H:%M"),
                "",
                "This report contains screen-habit statistics only. It is not a medical assessment.",
                "",
                f"Blinks per minute: {summary.get('blinks_per_minute', 0)}",
                f"Session duration: {summary.get('session_duration_minutes', 0)} minutes",
                f"Distance status: {summary.get('distance_status', 'UNKNOWN')}",
                f"Posture balance: {summary.get('posture_alignment_rate') if summary.get('posture_alignment_rate') is not None else 'No posture data'}",
                f"Warnings: {summary.get('warning_count', 0)}",
                f"Posture adjustment reminders: {summary.get('posture_warnings', 0)}",
                f"Breaks taken: {summary.get('breaks_taken', 0)}",
                f"Exercises completed: {summary.get('exercises_completed', 0)}",
                f"Trend: {trends}",
                "",
                f"Total sessions: {cumulative.get('total_sessions', 0)}",
                f"Total monitored minutes: {round(cumulative.get('total_session_seconds', 0) / 60, 1)}",
            ]
            if self._latest_health_report is not None:
                report_lines.extend([
                    "",
                    "Latest AI Coach recommendations:",
                    self._latest_health_report.top_priority,
                    self._latest_health_report.secondary_priority,
                    self._latest_health_report.going_well,
                    *self._latest_health_report.recommendations,
                ])
            axis.text(0.08, 0.95, "\n".join(report_lines), va="top", fontsize=11, color="#20252b")
            pdf.savefig(text_figure, bbox_inches="tight")
            pdf.savefig(self.stats_figure)

    def _refresh_statistics_tab(self) -> None:
        period_days = {"Today": 1, "7 Days": 7, "30 Days": 30}.get(self.period_menu.get(), 1)
        session = self.stats_manager.get_period_summary(period_days)
        cumulative = self.stats_manager.get_cumulative_data()
        score_data = self.stats_manager.calculate_wellness_score(session)
        heatmap = self.stats_manager.get_warning_heatmap()
        trends = self.stats_manager.get_trends()
        achievements = self.stats_manager.get_achievements()
        heatmap_text = " ".join(
            f"{hour:02d}:00 {'#' * min(count, 12)}"
            for hour, count in sorted(heatmap.items())
        ) or "Collecting more data..."

        lines = [
            "== Current Session ==",
            f"Blink count: {session['blink_count']}",
            f"Blinks per minute: {session['blinks_per_minute']}",
            f"Distance status (most common): {session['distance_status']}",
            f"Posture balance: {session.get('posture_alignment_rate') if session.get('posture_alignment_rate') is not None else 'Collecting data...'}%",
            f"Session duration: {session['session_duration_minutes']} minutes",
            f"Low blink-rate warnings: {session['low_blink_warnings']}",
            f"Distance warnings: {session['distance_warnings']}",
            f"Posture adjustment reminders: {session.get('posture_warnings', 0)}",
            f"Exercises completed: {session['exercises_completed']}",
            f"Breaks taken: {session.get('breaks_taken', 0)}",
            f"Digital Eye Wellness Score: {score_data['score']} / 100",
            f"Personal baseline blink rate: {session.get('baseline_blink_rate') or 'Collecting data...'}",
            f"Personal baseline distance: {session.get('baseline_distance_status', 'UNKNOWN')}",
            "Score factors: " + ", ".join(
                f"{'+' if item['delta'] >= 0 else ''}{item['delta']} {item['reason']}"
                for item in score_data["factors"]
            ),
            "",
            "== Trend vs Previous Week ==",
            f"Blink Rate: {self._format_trend(trends['blink_rate'])}",
            f"Warnings: {self._format_trend(trends['warnings'])}",
            f"Breaks: {self._format_trend(trends['breaks'])}",
            "",
            "== Warning Heatmap ==",
            heatmap_text,
            "",
            "== Today's Summary ==",
            f"Your blink rate was {('stable' if trends['blink_rate'] is None else 'tracked for comparison')} today.",
            f"You had {session.get('breaks_taken', 0)} break periods and {session.get('warning_count', 0)} warnings.",
            "Achievements: " + (", ".join(achievements) if achievements else "Keep collecting screen-habit data."),
            "",
            "== All-Time (previous sessions) ==",
            f"Total sessions recorded: {cumulative['total_sessions']}",
            f"Total blinks recorded: {cumulative['total_blinks']}",
            f"Total time monitored: {round(cumulative['total_session_seconds'] / 60, 1)} minutes",
            f"Total warnings issued: {cumulative['total_warnings']}",
            f"Total exercises completed: {cumulative['total_exercises_completed']}",
            f"Total breaks taken: {cumulative.get('total_breaks_taken', 0)}",
        ]
        if cumulative["health_score_history"]:
            avg_score = sum(cumulative["health_score_history"]) / len(cumulative["health_score_history"])
            lines.append(f"Average AI health score: {round(avg_score, 1)}")

        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", "\n".join(lines))
        self.stats_text.configure(state="disabled")
        self._refresh_charts()

    @staticmethod
    def _format_trend(value: Optional[float]) -> str:
        if value is None:
            return "Not enough data"
        arrow = "up" if value > 0 else "down" if value < 0 else "stable"
        return f"{arrow} {abs(value):.0f}% compared with the previous period"

    def _refresh_charts(self) -> None:
        samples = self.stats_manager.get_metric_history()
        for axis in self.stats_axes:
            axis.clear()
            axis.set_facecolor("#20252b")
            axis.tick_params(colors="#aeb8c2", labelsize=8)
            for spine in axis.spines.values():
                spine.set_color("#3b444d")

        if len(samples) < 2:
            for axis, title in zip(self.stats_axes, ("Blink Rate", "Distance", "Wellness Score", "Warnings")):
                axis.set_title(title, color="#f4f7f5", fontsize=10)
                axis.text(0.5, 0.5, "Collecting data...", ha="center", va="center", color="#aeb8c2")
                axis.set_xticks([])
                axis.set_yticks([])
            self.stats_canvas.draw_idle()
            return

        times = list(range(1, len(samples) + 1))
        blink_rates = [sample["blink_rate"] for sample in samples]
        distance_values = {
            "TOO CLOSE": 1.0,
            "GOOD DISTANCE": 0.5,
            "TOO FAR": 0.0,
            "UNKNOWN": 0.25,
        }
        distances = [distance_values.get(sample["distance_status"], 0.25) for sample in samples]
        scores = [sample["health_score"] for sample in samples if sample["health_score"] is not None]
        score_times = [times[index] for index, sample in enumerate(samples) if sample["health_score"] is not None]
        warning_counts = [sample.get("warning_count", 0) for sample in samples]

        self.stats_axes[0].plot(times, blink_rates, color="#4ecdc4", linewidth=2)
        self.stats_axes[0].set_title("Blink Rate", color="#f4f7f5", fontsize=10)
        self.stats_axes[0].set_ylabel("Blinks / min", color="#aeb8c2", fontsize=8)
        self.stats_axes[1].plot(times, distances, color="#f4b860", linewidth=2)
        self.stats_axes[1].set_title("Distance", color="#f4f7f5", fontsize=10)
        self.stats_axes[1].set_yticks([0, 0.5, 1], ["Far", "Good", "Close"])
        self.stats_axes[2].set_title("Wellness Score", color="#f4f7f5", fontsize=10)
        self.stats_axes[2].set_ylabel("Score / 100", color="#aeb8c2", fontsize=8)
        if len(scores) >= 2:
            self.stats_axes[2].plot(score_times, scores, color="#8bd450", linewidth=2)
        else:
            self.stats_axes[2].text(0.5, 0.5, "Collecting data...", ha="center", va="center", color="#aeb8c2")
        self.stats_axes[3].plot(times, warning_counts, color="#e05252", linewidth=2)
        self.stats_axes[3].set_title("Warnings", color="#f4f7f5", fontsize=10)
        self.stats_axes[3].set_ylabel("Count", color="#aeb8c2", fontsize=8)
        self.stats_figure.tight_layout(pad=1.5)
        self.stats_canvas.draw_idle()

    # ---- Settings tab ----

    def _build_settings_tab(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(frame, text="Settings", font=ctk.CTkFont(size=20, weight="bold")).pack(
            anchor="w", padx=10, pady=(10, 10))

        card = ctk.CTkScrollableFrame(frame)
        card.pack(fill="both", expand=True, padx=10, pady=10)

        settings = self.settings_manager.get_all()

        ctk.CTkLabel(card, text="Gemini API Key", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=15, pady=(15, 2))
        self.api_key_entry = ctk.CTkEntry(card, width=400, show="*", placeholder_text="Paste your Gemini API key")
        self.api_key_entry.insert(0, settings.gemini_api_key)
        self.api_key_entry.pack(anchor="w", padx=15, pady=(0, 15))

        self.voice_switch = ctk.CTkSwitch(card, text="Voice alerts")
        if settings.voice_alerts_enabled:
            self.voice_switch.select()
        self.voice_switch.pack(anchor="w", padx=15, pady=8)

        self.notif_switch = ctk.CTkSwitch(card, text="Desktop notifications")
        if settings.notifications_enabled:
            self.notif_switch.select()
        self.notif_switch.pack(anchor="w", padx=15, pady=8)

        self.focus_switch = ctk.CTkSwitch(card, text="Focus mode (reduce routine notifications)")
        if settings.focus_mode:
            self.focus_switch.select()
        self.focus_switch.pack(anchor="w", padx=15, pady=8)

        ctk.CTkLabel(card, text="Low blink-rate threshold (blinks/min)").pack(anchor="w", padx=15, pady=(15, 2))
        self.blink_threshold_entry = ctk.CTkEntry(card, width=100)
        self.blink_threshold_entry.insert(0, str(settings.blink_rate_threshold))
        self.blink_threshold_entry.pack(anchor="w", padx=15, pady=(0, 10))

        ctk.CTkLabel(card, text="\"Too close\" distance ratio (higher = more tolerant)").pack(anchor="w", padx=15, pady=(5, 2))
        self.too_close_entry = ctk.CTkEntry(card, width=100)
        self.too_close_entry.insert(0, str(settings.distance_too_close_ratio))
        self.too_close_entry.pack(anchor="w", padx=15, pady=(0, 10))

        ctk.CTkLabel(card, text="\"Too far\" distance ratio (lower = stricter)").pack(anchor="w", padx=15, pady=(5, 2))
        self.too_far_entry = ctk.CTkEntry(card, width=100)
        self.too_far_entry.insert(0, str(settings.distance_too_far_ratio))
        self.too_far_entry.pack(anchor="w", padx=15, pady=(0, 10))

        ctk.CTkLabel(card, text="20-20-20 break interval (minutes)").pack(anchor="w", padx=15, pady=(5, 2))
        self.break_interval_entry = ctk.CTkEntry(card, width=100)
        self.break_interval_entry.insert(0, str(settings.break_interval_minutes))
        self.break_interval_entry.pack(anchor="w", padx=15, pady=(0, 10))

        ctk.CTkLabel(card, text="Appearance").pack(anchor="w", padx=15, pady=(15, 2))
        self.theme_selector = ctk.CTkSegmentedButton(
            card, values=["Light", "Dark"], command=self._set_appearance_mode
        )
        self.theme_selector.set(settings.theme.title())
        self.theme_selector.pack(anchor="w", padx=15, pady=(0, 10))
        self.accent_selector = ctk.CTkSegmentedButton(
            card, values=["Blue", "Pink"], command=self._set_accent_theme
        )
        self.accent_selector.set(settings.accent_color.title())
        self.accent_selector.pack(anchor="w", padx=15, pady=(0, 20))

        self.settings_status_label = ctk.CTkLabel(card, text="", text_color="#2fa572")
        self.settings_status_label.pack(anchor="w", padx=15, pady=(0, 5))

        ctk.CTkButton(card, text="Save Settings", command=self._save_settings).pack(anchor="w", padx=15, pady=(5, 20))

        return frame

    def _save_settings(self) -> None:
        try:
            blink_threshold = int(self.blink_threshold_entry.get())
            too_close = float(self.too_close_entry.get())
            too_far = float(self.too_far_entry.get())
            break_interval = int(self.break_interval_entry.get())
        except ValueError:
            self.settings_status_label.configure(text="Please enter valid numbers.", text_color="#e05252")
            return

        updates = {
            "gemini_api_key": self.api_key_entry.get().strip(),
            "voice_alerts_enabled": bool(self.voice_switch.get()),
            "notifications_enabled": bool(self.notif_switch.get()),
            "focus_mode": bool(self.focus_switch.get()),
            "blink_rate_threshold": blink_threshold,
            "distance_too_close_ratio": too_close,
            "distance_too_far_ratio": too_far,
            "break_interval_minutes": break_interval,
            "theme": self.theme_selector.get().lower(),
            "accent_color": self.accent_selector.get().lower(),
        }
        self.settings_manager.update_many(updates)

        self.voice_manager.set_enabled(updates["voice_alerts_enabled"])
        self.notification_manager.set_enabled(updates["notifications_enabled"])
        self.notification_manager.set_focus_mode(updates["focus_mode"])
        self.ai_coach.set_api_key(updates["gemini_api_key"])
        if self.webcam_worker is not None:
            self.webcam_worker.update_distance_thresholds(too_close, too_far)
        self._set_appearance_mode(updates["theme"])
        self._set_accent_theme(updates["accent_color"])

        self.settings_status_label.configure(text="Settings saved.", text_color="#2fa572")

    # ------------------------------------------------------- Monitoring

    def _toggle_monitoring(self) -> None:
        if self.monitoring_active:
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _start_monitoring(self) -> None:
        self.stats_manager.start_new_session()
        self.stop_event.clear()
        self.webcam_worker = WebcamWorker(
            self.settings_manager, self.frame_queue, self.event_queue, self.stop_event
        )
        self.webcam_worker.start()
        self.monitoring_active = True
        self._session_persisted = False
        self.toggle_btn.configure(text=t("Stop Monitoring"), fg_color="#c0392b", hover_color="#a93226")
        self._set_status("Status: Starting camera...")
        self._refresh_privacy_tab()

    def _stop_monitoring(self) -> None:
        self.stop_event.set()
        self.monitoring_active = False
        self._configure_start_monitor_button()
        self._set_status("Status: Idle")
        self._refresh_privacy_tab()
        if not self._session_persisted:
            self.stats_manager.end_session_and_persist()
            self._session_persisted = True

    # ------------------------------------------------------- Queue polling

    def _poll_queues(self) -> None:
        self._drain_frame_queue()
        self._drain_event_queue()
        self._drain_ai_result_queue()
        self._update_session_duration_label()
        self.after(30, self._poll_queues)

    def _drain_frame_queue(self) -> None:
        try:
            frame, metrics = self.frame_queue.get_nowait()
        except queue.Empty:
            return

        # Convert BGR (OpenCV) -> RGB (PIL) and update the video label
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(VIDEO_WIDTH, VIDEO_HEIGHT))
        self.video_label.configure(image=ctk_image, text="")
        self.video_label.image = ctk_image  # keep a reference so it isn't garbage-collected

        self._apply_metrics(metrics)

    def _apply_metrics(self, metrics: dict) -> None:
        if metrics.get("blink_just_registered"):
            self.stats_manager.register_blink()

        distance_status = metrics.get("distance_status", "UNKNOWN")
        if metrics.get("face_detected"):
            self.stats_manager.register_distance_sample(distance_status)
        self.stats_manager.register_posture_sample(metrics.get("posture_status", PostureStatus.UNKNOWN.value))

        snapshot = self.stats_manager.get_dashboard_snapshot()
        score_data = self.stats_manager.calculate_wellness_score(snapshot)
        snapshot["wellness_score"] = score_data["score"]
        snapshot["wellness_score_factors"] = score_data["factors"]
        now = time.time()
        if now - self._last_metric_sample_epoch >= 5.0:
            self.stats_manager.record_metric_sample(
                snapshot["blinks_per_minute"], distance_status,
                self._latest_health_report.health_score if self._latest_health_report else None,
                snapshot["warning_count"],
            )
            self._last_metric_sample_epoch = now
        self.blink_count_label.configure(text=str(snapshot["blink_count"]))
        self.blink_rate_label.configure(text=f"{snapshot['blinks_per_minute']:.1f}")
        self.blink_status_label.configure(
            text=t("Closed") if metrics.get("is_eye_closed") else (t("Open") if metrics.get("face_detected") else t("No face detected"))
        )
        self.distance_status_label.configure(text=t(distance_status))
        self.posture_status_label.configure(text=t(metrics.get("posture_status", PostureStatus.UNKNOWN.value)))
        self.posture_guidance_label.configure(text=t(metrics.get("posture_guidance", "Keep your head and shoulders in view")))
        posture_balance = snapshot.get("posture_alignment_rate")
        self.posture_balance_label.configure(
            text=f"{posture_balance:.0f}%" if posture_balance is not None else t("Collecting")
        )
        self.tracking_confidence_label.configure(text=t(metrics.get("tracking_confidence", "Low confidence")))
        self.lighting_label.configure(text=t(metrics.get("lighting_quality", "—")))
        self.wellness_score_label.configure(text=f"{score_data['score']} / 100")
        factor_text = "\n".join(
            f"{'+' if item['delta'] >= 0 else ''}{item['delta']} {item['reason']}"
            for item in score_data["factors"]
        ) or "No score changes recorded yet."
        self.wellness_factors_label.configure(text=factor_text)
        self._magnifier.update(distance_status)

        self._check_alerts(
            snapshot,
            distance_status,
            metrics.get("posture_status", PostureStatus.UNKNOWN.value),
            metrics.get("posture_guidance", "Please adjust your posture."),
            metrics.get("posture_warning_just_registered", False),
        )

    def _check_alerts(self, snapshot: dict, distance_status: str, posture_status: str,
                      posture_guidance: str, posture_warning_just_registered: bool) -> None:
        settings = self.settings_manager.get_all()

        # Note: NotificationManager applies its own per-topic cooldown, so
        # this check runs every poll tick but only actually fires (and is
        # only counted as a warning) at most once per cooldown window.
        if snapshot["blinks_per_minute"] > 0 and snapshot["blinks_per_minute"] < settings.blink_rate_threshold \
                and snapshot["session_duration_minutes"] >= 0.25:
            if self.notification_manager.notify_low_blink_rate():
                self.voice_manager.speak("Please blink more frequently.")
                self.stats_manager.register_low_blink_warning()
                self.stats_manager.register_warning("low_blink_rate")

        if distance_status == DistanceStatus.TOO_CLOSE.value:
            if self.notification_manager.notify_too_close():
                self.voice_manager.speak("Please move away from the screen.")
                self.stats_manager.register_distance_warning()
                self.stats_manager.register_warning("too_close")

        if posture_status == PostureStatus.ADJUST.value and posture_warning_just_registered:
            if self.notification_manager.notify_posture_adjustment(posture_guidance):
                self.voice_manager.speak(posture_guidance)
                self.stats_manager.register_posture_warning()
                self.stats_manager.register_warning("posture_adjustment")

    def _drain_event_queue(self) -> None:
        while True:
            try:
                kind, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "camera_ready":
                self._set_status("Status: Monitoring")
                self._refresh_privacy_tab()
            elif kind == "camera_error":
                self._set_status("Status: Error - {payload}", payload=payload)
                self.monitoring_active = False
                self._configure_start_monitor_button()
                self._refresh_privacy_tab()
            elif kind == "posture_calibrated":
                self.posture_guidance_label.configure(text=t("Posture reference saved for this session"))
            elif kind == "posture_calibration_error":
                self.posture_guidance_label.configure(text=t("Keep your head and shoulders in view, then calibrate again"))
            elif kind == "tray_open":
                self._show_from_tray()
            elif kind == "tray_pause":
                self._stop_monitoring()
            elif kind == "tray_resume":
                self._start_monitoring()
            elif kind == "tray_tab":
                self._show_from_tray()
                self._show_tab(payload)
            elif kind == "tray_exit":
                self._exit_application()

    def _drain_ai_result_queue(self) -> None:
        try:
            result = self.ai_result_queue.get_nowait()
        except queue.Empty:
            return
        if isinstance(result, tuple) and result[0] == "chat":
            self._append_chat(f"{t('Coach')}: {result[1]}")
            return
        report: HealthReport = result
        self._display_health_report(report)
        self.ai_analyze_btn.configure(state="normal", text=t("Get AI Analysis"))

    def _update_session_duration_label(self) -> None:
        seconds = int(self.stats_manager.session.session_duration_seconds())
        minutes, secs = divmod(seconds, 60)
        self.session_time_label.configure(text=f"{minutes:02d}:{secs:02d}")

    # ------------------------------------------------------- AI Coach

    def _request_ai_analysis(self) -> None:
        self.ai_analyze_btn.configure(state="disabled", text=t("Analyzing..."))
        stats_snapshot = self.stats_manager.get_dashboard_snapshot()
        score_data = self.stats_manager.calculate_wellness_score(stats_snapshot)
        stats_snapshot["wellness_score"] = score_data["score"]
        stats_snapshot["wellness_score_factors"] = score_data["factors"]
        stats_snapshot["historical"] = self.stats_manager.get_historical_context()
        baseline = stats_snapshot["historical"].get("baseline", {})
        self.baseline_label.configure(
            text=(
                f"Your baseline: {baseline.get('blink_rate') or '—'} blinks/min | "
                f"Typical distance: {baseline.get('typical_distance', 'UNKNOWN')} | "
                f"Average session: {baseline.get('average_session_minutes') or '—'} min"
            )
        )
        threading.Thread(target=self._run_ai_analysis, args=(stats_snapshot,), daemon=True).start()

    def _run_ai_analysis(self, stats_snapshot: dict) -> None:
        """Runs on a background thread - never touches widgets directly."""
        report = self.ai_coach.get_health_report(stats_snapshot)
        self.ai_result_queue.put(report)

    def _display_health_report(self, report: HealthReport) -> None:
        self._latest_health_report = report
        score_data = self.stats_manager.calculate_wellness_score()
        display_score = score_data["score"]
        self.stats_manager.register_health_score(display_score)
        self.stats_manager.remember_recommendations(report.recommendations)
        self._refresh_charts()

        self.health_score_label.configure(text=f"{display_score} / 100")
        self.health_score_bar.set(display_score / 100)
        self.analysis_label.configure(text=report.analysis)
        self.top_priority_label.configure(text=t("Top priority: {text}", text=report.top_priority))
        self.secondary_priority_label.configure(text=t("Secondary: {text}", text=report.secondary_priority))
        self.going_well_label.configure(text=t("Going well: {text}", text=report.going_well))
        self.recommendations_label.configure(
            text="\n".join(f"• {r}" for r in report.recommendations)
        )

        ex = report.exercise
        self.exercise_name_label.configure(text=ex.name)
        self.exercise_instructions_label.configure(text=ex.instructions)
        self.exercise_duration_label.configure(text=t("Duration: {seconds} seconds", seconds=ex.duration_seconds))
        self.next_reminder_label.configure(text=t("Next reminder: in {minutes} minutes", minutes=ex.remind_after_minutes))
        self.motivation_label.configure(text=report.motivation)
        self.exercise_page_name.configure(text=ex.name)
        self.exercise_page_instructions.configure(text=ex.instructions)
        self.exercise_page_countdown.configure(text=f"Ready - {ex.duration_seconds}s")

        self.schedule_exercise_reminder(ex.name, ex.instructions, ex.duration_seconds, ex.remind_after_minutes)

    # ------------------------------------------------------- Exercise reminders

    def schedule_exercise_reminder(self, name: str, instructions: str, duration_seconds: int,
                                    delay_minutes: int) -> None:
        delay_ms = max(1, delay_minutes) * 60 * 1000
        self.after(delay_ms, lambda: self._trigger_exercise_reminder(name, instructions, duration_seconds))

    def _trigger_exercise_reminder(self, name: str, instructions: str, duration_seconds: int) -> None:
        self.notification_manager.notify_exercise(name, instructions)
        self.voice_manager.speak(f"Time for your recommended eye exercise: {name}.")
        ExerciseReminderWindow(self, name, instructions, duration_seconds)

    # ------------------------------------------------------- 20-20-20 breaks

    def _schedule_break_reminder(self) -> None:
        settings = self.settings_manager.get_all()
        delay_ms = max(1, settings.break_interval_minutes) * 60 * 1000
        self._next_break_epoch = time.time() + delay_ms / 1000
        if self._break_job is not None:
            try:
                self.after_cancel(self._break_job)
            except ValueError:
                pass
        self._break_job = self.after(delay_ms, self._trigger_break_reminder)
        self._tick_break_countdown()

    def _tick_break_countdown(self) -> None:
        if self._next_break_epoch is None:
            return
        remaining = max(0, int(self._next_break_epoch - time.time()))
        minutes, secs = divmod(remaining, 60)
        if not self._break_reminder_active:
            self.next_break_label.configure(text=t("Next break in {time}", time=f"{minutes:02d}:{secs:02d}"))
        if self._break_countdown_job is not None:
            try:
                self.after_cancel(self._break_countdown_job)
            except ValueError:
                pass
        self._break_countdown_job = self.after(1000, self._tick_break_countdown)

    def _trigger_break_reminder(self) -> None:
        self._break_reminder_active = True
        self.notification_manager.notify_break_reminder()
        self.voice_manager.speak("Time for a break. Look at something 20 feet away for 20 seconds.")

        settings = self.settings_manager.get_all()
        self.break_label.configure(
            text=t("TIME FOR A BREAK - look away for {seconds}s", seconds=settings.break_look_away_seconds)
        )
        self.break_start_btn.configure(state="normal", text=t("Start Break"))
        self.break_banner.pack(fill="x", padx=15, pady=(0, 15))
        self.next_break_label.configure(text=t("Break reminder active"))

    def _start_break(self) -> None:
        settings = self.settings_manager.get_all()
        self._break_remaining = max(1, settings.break_look_away_seconds)
        self.break_start_btn.configure(state="disabled", text=t("In progress"))
        self._tick_break_session()

    def _tick_break_session(self) -> None:
        self.break_label.configure(text=t("TIME FOR A BREAK - {seconds}s remaining", seconds=self._break_remaining))
        if self._break_remaining <= 0:
            self.stats_manager.register_break_taken()
            self.break_label.configure(text=t("Break complete."))
            self.after(1200, self._dismiss_break_reminder)
            return
        self._break_remaining -= 1
        self._break_session_job = self.after(1000, self._tick_break_session)

    def _dismiss_break_reminder(self) -> None:
        self._break_reminder_active = False
        if self._break_session_job is not None:
            try:
                self.after_cancel(self._break_session_job)
            except ValueError:
                pass
            self._break_session_job = None
        self.break_banner.pack_forget()
        self._schedule_break_reminder()

    def _snooze_break_reminder(self) -> None:
        self._break_reminder_active = False
        self.break_banner.pack_forget()
        delay_ms = 5 * 60 * 1000
        self._next_break_epoch = time.time() + delay_ms / 1000
        if self._break_job is not None:
            try:
                self.after_cancel(self._break_job)
            except ValueError:
                pass
        self._break_job = self.after(delay_ms, self._trigger_break_reminder)

    # ------------------------------------------------------- Shutdown

    def _on_close(self) -> None:
        if not self._exit_requested:
            self.withdraw()
            self.notification_manager.notify(
                "EyeGuard AI", "Monitoring continues in the system tray.", topic="tray_hidden"
            )
            return

        self.stop_event.set()
        if self.monitoring_active and not self._session_persisted:
            self.stats_manager.end_session_and_persist()
            self._session_persisted = True
        for job in (self._break_job, self._break_countdown_job, self._break_session_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except ValueError:
                    pass
        self.voice_manager.shutdown()
        self._magnifier.close()
        self._tray.stop()
        self.destroy()

    def _exit_application(self) -> None:
        self._exit_requested = True
        self._on_close()
