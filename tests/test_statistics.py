import tempfile
import unittest

from statistics import StatisticsManager


class PostureStatisticsTests(unittest.TestCase):
    def test_posture_balance_tracks_aligned_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = StatisticsManager(path=f"{directory}/statistics.json")
            manager.register_posture_sample("ALIGNED")
            manager.register_posture_sample("ADJUST POSTURE")
            manager.register_posture_sample("ALIGNED")

            snapshot = manager.get_dashboard_snapshot()

            self.assertEqual(snapshot["posture_alignment_rate"], 66.7)

    def test_missing_upper_body_is_not_counted_as_an_adjustment(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = StatisticsManager(path=f"{directory}/statistics.json")
            manager.register_posture_sample("UPPER BODY NOT DETECTED")

            self.assertIsNone(manager.get_dashboard_snapshot()["posture_alignment_rate"])

    def test_timeline_includes_live_session_events(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = StatisticsManager(path=f"{directory}/statistics.json")
            manager.register_warning("posture_adjustment")

            events = manager.get_timeline_events()

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["topic"], "posture_adjustment")

    def test_touch_grass_unlocks_after_first_break(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = StatisticsManager(path=f"{directory}/statistics.json")
            manager.register_break_taken()
            manager.end_session_and_persist()

            achievements = manager.get_achievements()

            self.assertIn("Touch Grass", achievements)


if __name__ == "__main__":
    unittest.main()
