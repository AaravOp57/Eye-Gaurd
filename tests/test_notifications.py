import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

from i18n import t
from notifications import NotificationManager


class FakeTray:
    def __init__(self):
        self.notified = []
        self.event = threading.Event()

    def notify(self, title, message):
        self.notified.append((title, message))
        self.event.set()
        return True


class NotificationManagerTests(unittest.TestCase):
    def tearDown(self):
        t.set_language("en")

    def test_notify_prefers_desktop_backend(self):
        manager = NotificationManager(enabled=True, cooldown_seconds=0)
        tray = FakeTray()
        manager.set_tray(tray)
        desktop_notify = Mock()
        plyer = types.ModuleType("plyer")
        plyer.notification = types.SimpleNamespace(notify=desktop_notify)

        with patch.dict(sys.modules, {"plyer": plyer}):
            manager._send("Title", "Body", timeout=5)

        desktop_notify.assert_called_once_with(
            title="Title", message="Body", app_name="EyeGuard AI", timeout=5
        )
        self.assertEqual(tray.notified, [])

    def test_notify_uses_tray_when_desktop_backend_fails(self):
        manager = NotificationManager(enabled=True, cooldown_seconds=0)
        tray = FakeTray()
        manager.set_tray(tray)
        desktop_notify = Mock(side_effect=RuntimeError("backend unavailable"))
        plyer = types.ModuleType("plyer")
        plyer.notification = types.SimpleNamespace(notify=desktop_notify)

        with patch.dict(sys.modules, {"plyer": plyer}):
            manager._send("Title", "Body", timeout=5)

        self.assertTrue(tray.event.is_set())
        self.assertEqual(tray.notified, [("Title", "Body")])

    def test_alert_notifications_use_the_selected_language(self):
        manager = NotificationManager(enabled=True, cooldown_seconds=0)
        manager.notify = Mock(return_value=True)
        self.assertTrue(t.set_language("fr"))

        manager.notify_low_blink_rate()
        manager.notify_too_close()
        manager.notify_posture_adjustment("Level your shoulders")
        manager.notify_break_reminder()
        manager.notify_exercise("Relaxation", "Close and open your eyes.")

        calls = manager.notify.call_args_list
        self.assertEqual(calls[0].kwargs["message"], "Clignez des yeux plus fréquemment.")
        self.assertEqual(calls[1].kwargs["message"], "Éloignez-vous un peu de l’écran.")
        self.assertEqual(calls[2].kwargs["message"], "Gardez les épaules bien droites")
        self.assertEqual(calls[3].kwargs["message"], "Regardez un objet situé à environ 6 mètres pendant 20 secondes.")
        self.assertEqual(calls[4].kwargs["title"], "EyeGuard AI — Exercice recommandé")


if __name__ == "__main__":
    unittest.main()
