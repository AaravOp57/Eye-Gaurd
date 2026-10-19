import threading
import unittest

from notifications import NotificationManager


class FakeTray:
    def __init__(self):
        self.notified = []
        self.event = threading.Event()

    def notify(self, title, message):
        self.notified.append((title, message))
        self.event.set()


class NotificationManagerTests(unittest.TestCase):
    def test_notify_uses_tray_fallback(self):
        manager = NotificationManager(enabled=True, cooldown_seconds=0)
        tray = FakeTray()
        manager.set_tray(tray)

        manager._send("Title", "Body", timeout=5)

        self.assertTrue(tray.event.is_set())
        self.assertEqual(tray.notified, [("Title", "Body")])


if __name__ == "__main__":
    unittest.main()
