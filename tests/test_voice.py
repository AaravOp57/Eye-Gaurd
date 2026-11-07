import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from i18n import t
from voice import VOICE_ALERT_REPEAT_SECONDS, VoiceManager


class VoiceManagerTests(unittest.TestCase):
    def test_alerts_repeat_after_cooldown_without_repeating_during_cooldown(self):
        with patch.object(VoiceManager, "_start_worker"):
            manager = VoiceManager()

        with patch("voice.time.monotonic", side_effect=[10.0, 10.5, 25.0]):
            manager.speak(
                "First alert",
                cooldown_key="too_close",
                cooldown_seconds=VOICE_ALERT_REPEAT_SECONDS,
            )
            manager.speak(
                "Suppressed alert",
                cooldown_key="too_close",
                cooldown_seconds=VOICE_ALERT_REPEAT_SECONDS,
            )
            manager.speak(
                "Repeated alert",
                cooldown_key="too_close",
                cooldown_seconds=VOICE_ALERT_REPEAT_SECONDS,
            )

        self.assertEqual(manager._queue.get_nowait(), "First alert")
        self.assertEqual(manager._queue.get_nowait(), "Repeated alert")
        self.assertTrue(manager._queue.empty())

    def test_voice_alerts_repeat_after_ten_seconds(self):
        self.assertEqual(VOICE_ALERT_REPEAT_SECONDS, 10.0)

    def test_speak_once_repeats_only_after_cooldown_reset(self):
        with patch.object(VoiceManager, "_start_worker"):
            manager = VoiceManager()

        with patch("voice.time.monotonic", side_effect=[10.0, 10.5]):
            manager.speak_once("First alert", cooldown_key="low_blink_rate")
            manager.speak_once("Suppressed alert", cooldown_key="low_blink_rate")
            manager.reset_cooldown("low_blink_rate")
            manager.speak_once("Second alert", cooldown_key="low_blink_rate")

        self.assertEqual(manager._queue.get_nowait(), "First alert")
        self.assertEqual(manager._queue.get_nowait(), "Second alert")
        self.assertTrue(manager._queue.empty())

    def test_new_alert_episode_speaks_immediately_after_cooldown_reset(self):
        with patch.object(VoiceManager, "_start_worker"):
            manager = VoiceManager()

        with patch("voice.time.monotonic", side_effect=[10.0, 10.5]):
            manager.speak(
                "First approach",
                cooldown_key="too_close",
                cooldown_seconds=VOICE_ALERT_REPEAT_SECONDS,
            )
            manager.reset_cooldown("too_close")
            manager.speak(
                "Second approach",
                cooldown_key="too_close",
                cooldown_seconds=VOICE_ALERT_REPEAT_SECONDS,
            )

        self.assertEqual(manager._queue.get_nowait(), "First approach")
        self.assertEqual(manager._queue.get_nowait(), "Second approach")
        self.assertTrue(manager._queue.empty())

    def test_distance_alert_speaks_again_after_user_moves_away_and_returns(self):
        from distance_detector import DistanceStatus
        from eyeguard_ui import EyeGuardApp
        from posture_detector import PostureStatus

        with patch.object(VoiceManager, "_start_worker"):
            manager = VoiceManager()

        app = EyeGuardApp.__new__(EyeGuardApp)
        app.settings_manager = SimpleNamespace(
            get_all=lambda: SimpleNamespace(blink_rate_threshold=12)
        )
        app.notification_manager = SimpleNamespace(
            notify_low_blink_rate=lambda: False,
            notify_too_close=lambda: False,
            notify_posture_adjustment=lambda guidance: False,
        )
        app.stats_manager = SimpleNamespace(
            register_low_blink_warning=lambda: None,
            register_warning=lambda topic: None,
            register_distance_warning=lambda: None,
            register_posture_warning=lambda: None,
        )
        app.voice_manager = manager
        snapshot = {"blinks_per_minute": 0, "session_duration_minutes": 1}

        with patch("voice.time.monotonic", side_effect=[10.0, 10.5]):
            app._check_alerts(
                snapshot, DistanceStatus.TOO_CLOSE.value,
                PostureStatus.UNKNOWN.value, "Adjust", False,
            )
            app._check_alerts(
                snapshot, DistanceStatus.GOOD.value,
                PostureStatus.UNKNOWN.value, "Adjust", False,
            )
            app._check_alerts(
                snapshot, DistanceStatus.TOO_CLOSE.value,
                PostureStatus.UNKNOWN.value, "Adjust", False,
            )

        self.assertEqual(manager._queue.get_nowait(), "Please move away from the screen.")
        self.assertEqual(manager._queue.get_nowait(), "Please move away from the screen.")
        self.assertTrue(manager._queue.empty())

    def test_low_blink_alert_speaks_once_until_blink_rate_recovers(self):
        from eyeguard_ui import EyeGuardApp
        from posture_detector import PostureStatus

        with patch.object(VoiceManager, "_start_worker"):
            manager = VoiceManager()

        app = EyeGuardApp.__new__(EyeGuardApp)
        app.settings_manager = SimpleNamespace(
            get_all=lambda: SimpleNamespace(blink_rate_threshold=12)
        )
        app.notification_manager = SimpleNamespace(
            notify_low_blink_rate=lambda: False,
            notify_too_close=lambda: False,
            notify_posture_adjustment=lambda guidance: False,
        )
        app.stats_manager = SimpleNamespace(
            register_low_blink_warning=lambda: None,
            register_warning=lambda topic: None,
            register_distance_warning=lambda: None,
            register_posture_warning=lambda: None,
        )
        app.voice_manager = manager
        low_rate = {"blinks_per_minute": 5, "session_duration_minutes": 1}
        recovered_rate = {"blinks_per_minute": 12, "session_duration_minutes": 1}

        t.set_language("hi")
        try:
            app._check_alerts(
                low_rate, "GOOD DISTANCE", PostureStatus.UNKNOWN.value, "Adjust", False,
            )
            app._check_alerts(
                low_rate, "GOOD DISTANCE", PostureStatus.UNKNOWN.value, "Adjust", False,
            )
            app._check_alerts(
                recovered_rate, "GOOD DISTANCE", PostureStatus.UNKNOWN.value, "Adjust", False,
            )
            app._check_alerts(
                low_rate, "GOOD DISTANCE", PostureStatus.UNKNOWN.value, "Adjust", False,
            )

            self.assertEqual(manager._queue.get_nowait(), "कृपया अधिक बार पलकें झपकाएँ।")
            self.assertEqual(manager._queue.get_nowait(), "कृपया अधिक बार पलकें झपकाएँ।")
            self.assertTrue(manager._queue.empty())
        finally:
            t.set_language("en")

    def test_alert_cooldowns_are_independent_by_topic(self):
        with patch.object(VoiceManager, "_start_worker"):
            manager = VoiceManager()

        with patch("voice.time.monotonic", side_effect=[10.0, 10.5]):
            manager.speak("First alert", cooldown_key="too_close", cooldown_seconds=60)
            manager.speak("Other alert", cooldown_key="low_blink_rate", cooldown_seconds=60)

        self.assertEqual(manager._queue.get_nowait(), "First alert")
        self.assertEqual(manager._queue.get_nowait(), "Other alert")
        self.assertTrue(manager._queue.empty())

    def test_worker_reinitializes_engine_for_each_alert(self):
        playback_complete = threading.Event()
        played_phrases = []
        init_count = 0

        class FakeEngine:
            def getProperty(self, name):
                return [SimpleNamespace(id="english", languages=["en-US"])]

            def say(self, phrase):
                played_phrases.append(phrase)

            def runAndWait(self):
                if len(played_phrases) == 2:
                    playback_complete.set()

            def setProperty(self, name, value):
                pass

            def stop(self):
                pass

        def init_engine():
            nonlocal init_count
            init_count += 1
            return FakeEngine()

        fake_pyttsx3 = types.ModuleType("pyttsx3")
        fake_pyttsx3.init = init_engine
        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            manager = VoiceManager()
            try:
                manager.speak("First alert")
                manager.speak("Second alert")
                self.assertTrue(playback_complete.wait(timeout=2))
                self.assertEqual(played_phrases, ["First alert", "Second alert"])
                self.assertEqual(init_count, 2)
            finally:
                manager.shutdown()

    def test_worker_uses_voice_for_selected_language_when_available(self):
        playback_complete = threading.Event()
        selected_voices = []

        class FakeEngine:
            def getProperty(self, name):
                return [
                    SimpleNamespace(id="english", languages=["en-US"]),
                    SimpleNamespace(id="french", languages=["fr-FR"]),
                ]

            def setProperty(self, name, value):
                if name == "voice":
                    selected_voices.append(value)

            def say(self, phrase):
                pass

            def runAndWait(self):
                playback_complete.set()

            def stop(self):
                pass

        fake_pyttsx3 = types.ModuleType("pyttsx3")
        fake_pyttsx3.init = FakeEngine
        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            manager = VoiceManager()
            try:
                manager.set_language("fr")
                manager.speak("Bonjour")
                self.assertTrue(playback_complete.wait(timeout=2))
                self.assertEqual(selected_voices, ["french"])
            finally:
                manager.shutdown()

    def test_worker_restarts_engine_after_playback_failure(self):
        playback_recovered = threading.Event()
        run_count = 0
        init_count = 0

        class FakeEngine:
            def getProperty(self, name):
                return [SimpleNamespace(id="english", languages=["en-US"])]

            def say(self, phrase):
                pass

            def setProperty(self, name, value):
                pass

            def runAndWait(self):
                nonlocal run_count
                run_count += 1
                if run_count == 1:
                    raise RuntimeError("simulated playback failure")
                playback_recovered.set()

            def stop(self):
                pass

        def init_engine():
            nonlocal init_count
            init_count += 1
            return FakeEngine()

        fake_pyttsx3 = types.ModuleType("pyttsx3")
        fake_pyttsx3.init = init_engine
        with patch.dict(sys.modules, {"pyttsx3": fake_pyttsx3}):
            manager = VoiceManager()
            try:
                manager.speak("First alert")
                manager.speak("Repeated alert")
                self.assertTrue(playback_recovered.wait(timeout=2))
                self.assertEqual(init_count, 2)
            finally:
                manager.shutdown()


if __name__ == "__main__":
    unittest.main()
