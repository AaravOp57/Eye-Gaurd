import unittest

from posture_detector import PostureDetector, PostureStatus


class Landmark:
    def __init__(self, x, y, visibility=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def pose_landmarks(left_ear, right_ear, left_shoulder, right_shoulder):
    landmarks = [Landmark(0.0, 0.0, 0.0) for _ in range(33)]
    landmarks[7] = Landmark(*left_ear)
    landmarks[8] = Landmark(*right_ear)
    landmarks[11] = Landmark(*left_shoulder)
    landmarks[12] = Landmark(*right_shoulder)
    return landmarks


class PostureDetectorTests(unittest.TestCase):
    def test_balanced_landmarks_are_aligned(self):
        result = PostureDetector().process(
            pose_landmarks((0.4, 0.3), (0.6, 0.3), (0.35, 0.6), (0.65, 0.6))
        )
        self.assertEqual(result.status, PostureStatus.ALIGNED)

    def test_tilted_shoulders_request_adjustment(self):
        result = PostureDetector().process(
            pose_landmarks((0.4, 0.3), (0.6, 0.3), (0.35, 0.52), (0.65, 0.63))
        )
        self.assertEqual(result.status, PostureStatus.ADJUST)
        self.assertEqual(result.guidance, "Level your shoulders")

    def test_small_shoulder_tilt_is_within_the_comfort_range(self):
        result = PostureDetector().process(
            pose_landmarks((0.4, 0.3), (0.6, 0.3), (0.35, 0.6), (0.65, 0.65))
        )
        self.assertEqual(result.status, PostureStatus.ALIGNED)

    def test_low_visibility_is_unknown(self):
        landmarks = pose_landmarks((0.4, 0.3), (0.6, 0.3), (0.35, 0.6), (0.65, 0.6))
        landmarks[11].visibility = 0.2
        result = PostureDetector().process(landmarks)
        self.assertEqual(result.status, PostureStatus.UNKNOWN)

    def test_calibration_uses_a_comfortable_reference(self):
        detector = PostureDetector()
        reference = pose_landmarks((0.4, 0.3), (0.6, 0.34), (0.35, 0.6), (0.65, 0.68))
        self.assertTrue(detector.calibrate(reference))
        self.assertEqual(detector.process(reference).status, PostureStatus.ALIGNED)


if __name__ == "__main__":
    unittest.main()
