"""
eye_detector.py
----------------
Blink detection using the Eye Aspect Ratio (EAR) technique applied to
MediaPipe Face Mesh landmarks, adapted from the classic Soukupova &
Cech EAR method (originally designed for dlib's 6-point eye model).

EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

When the eye is open, EAR stays roughly constant. When the eye closes,
the vertical distances shrink rapidly, dropping EAR sharply. A blink is
counted once, on the transition from "closed" back to "open", so a
single blink is never counted more than once even if it spans several
frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# MediaPipe Face Mesh landmark indices approximating the classic 6-point
# eye model (outer corner, two upper lid points, inner corner, two lower
# lid points) for each eye.
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Outer eye corners, used elsewhere (distance_detector.py) for interocular
# distance estimation.
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33


@dataclass
class BlinkFrameResult:
    """Result of processing a single frame for blink detection."""

    ear: float
    is_eye_closed: bool
    blink_just_registered: bool


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _eye_aspect_ratio(points: Sequence[Tuple[float, float]]) -> float:
    """Compute EAR for a single eye given its 6 (x, y) landmark points."""
    # points order: [outer_corner, upper1, upper2, inner_corner, lower1, lower2]
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class EyeDetector:
    """
    Stateful blink detector. Feed it MediaPipe landmarks frame by frame
    via `process(landmarks, frame_width, frame_height)`.
    """

    def __init__(self, ear_threshold: float = 0.21, consec_frames: int = 2) -> None:
        """
        Args:
            ear_threshold: EAR value below which an eye is considered closed.
                           0.21 is a reasonable general-purpose default.
            consec_frames: Minimum consecutive closed-eye frames required
                           before the eye state is trusted as "closed".
                           Filters out single-frame landmark noise.
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self._closed_frame_streak = 0
        self._eye_state_closed = False
        self._total_blinks = 0

    def reset(self) -> None:
        self._closed_frame_streak = 0
        self._eye_state_closed = False
        self._total_blinks = 0

    @property
    def total_blinks(self) -> int:
        return self._total_blinks

    def process(
        self,
        landmarks: Sequence,
        frame_width: int,
        frame_height: int,
    ) -> BlinkFrameResult:
        """
        Args:
            landmarks: MediaPipe normalized landmark list (landmark.x/.y in 0..1)
            frame_width / frame_height: pixel dimensions of the source frame

        Returns:
            BlinkFrameResult with the current EAR, closed-state, and whether
            a new blink was just registered on this call.
        """
        left_points = [
            (landmarks[i].x * frame_width, landmarks[i].y * frame_height)
            for i in LEFT_EYE_IDX
        ]
        right_points = [
            (landmarks[i].x * frame_width, landmarks[i].y * frame_height)
            for i in RIGHT_EYE_IDX
        ]

        left_ear = _eye_aspect_ratio(left_points)
        right_ear = _eye_aspect_ratio(right_points)
        avg_ear = (left_ear + right_ear) / 2.0

        blink_just_registered = False
        currently_closed = avg_ear < self.ear_threshold

        if currently_closed:
            self._closed_frame_streak += 1
        else:
            # Eye just reopened - if it was closed long enough to count as a
            # real blink (not noise), register exactly one blink here.
            if self._eye_state_closed and self._closed_frame_streak >= self.consec_frames:
                self._total_blinks += 1
                blink_just_registered = True
            self._closed_frame_streak = 0

        self._eye_state_closed = currently_closed

        return BlinkFrameResult(
            ear=avg_ear,
            is_eye_closed=currently_closed,
            blink_just_registered=blink_just_registered,
        )
