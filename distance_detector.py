"""
distance_detector.py
---------------------
Approximates the user's distance from the screen using the interocular
distance (distance between the outer corners of the two eyes) measured
in pixels, expressed as a ratio of the frame width.

IMPORTANT: This is a *relative, uncalibrated* estimate. Without knowing
the physical focal length of the specific webcam, true distance in
centimeters cannot be measured reliably. EyeGuard AI therefore only
classifies distance into three qualitative buckets and never presents
the reading as a medically precise measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Tuple

import numpy as np

from eye_detector import LEFT_EYE_OUTER, RIGHT_EYE_OUTER


class DistanceStatus(str, Enum):
    TOO_CLOSE = "TOO CLOSE"
    GOOD = "GOOD DISTANCE"
    TOO_FAR = "TOO FAR"
    UNKNOWN = "UNKNOWN"


@dataclass
class DistanceResult:
    status: DistanceStatus
    interocular_ratio: float  # interocular pixel distance / frame width


class DistanceDetector:
    """
    Classifies approximate screen distance from the interocular distance
    ratio. Larger ratio = face takes up more of the frame = closer to
    the camera/screen. Thresholds are configurable via settings.py so
    they can be tuned per-webcam/per-user if needed.
    """

    def __init__(self, too_close_ratio: float = 0.16, too_far_ratio: float = 0.055) -> None:
        """
        Args:
            too_close_ratio: interocular/frame-width ratio above which the
                              user is classified as TOO CLOSE.
            too_far_ratio: ratio below which the user is classified as
                            TOO FAR. Between the two thresholds is GOOD.
        """
        self.too_close_ratio = too_close_ratio
        self.too_far_ratio = too_far_ratio

    def update_thresholds(self, too_close_ratio: float, too_far_ratio: float) -> None:
        self.too_close_ratio = too_close_ratio
        self.too_far_ratio = too_far_ratio

    def estimate(self, landmarks: Sequence, frame_width: int, frame_height: int) -> DistanceResult:
        """
        Args:
            landmarks: MediaPipe normalized landmark list for one detected face.
            frame_width / frame_height: pixel dimensions of the source frame.
        """
        left = landmarks[LEFT_EYE_OUTER]
        right = landmarks[RIGHT_EYE_OUTER]

        left_px = np.array([left.x * frame_width, left.y * frame_height])
        right_px = np.array([right.x * frame_width, right.y * frame_height])
        interocular_px = float(np.linalg.norm(left_px - right_px))

        ratio = interocular_px / frame_width if frame_width > 0 else 0.0

        if ratio >= self.too_close_ratio:
            status = DistanceStatus.TOO_CLOSE
        elif ratio <= self.too_far_ratio:
            status = DistanceStatus.TOO_FAR
        else:
            status = DistanceStatus.GOOD

        return DistanceResult(status=status, interocular_ratio=ratio)
