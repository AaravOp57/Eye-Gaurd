"""Lightweight, non-medical posture guidance from MediaPipe Pose landmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class PostureStatus(str, Enum):
    ALIGNED = "ALIGNED"
    ADJUST = "ADJUST POSTURE"
    UNKNOWN = "UPPER BODY NOT DETECTED"


@dataclass
class PostureResult:
    status: PostureStatus
    guidance: str
    shoulder_tilt_degrees: float = 0.0
    head_tilt_degrees: float = 0.0
    warning_just_registered: bool = False


class PostureDetector:
    """Classify obvious shoulder and head alignment changes in a webcam view.

    The detector intentionally reports guidance rather than a medical judgement.
    It uses only the ears and shoulders, which are available from MediaPipe Pose.
    """

    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(self, tilt_threshold_degrees: float = 12.0, warning_frames: int = 60) -> None:
        self.tilt_threshold_degrees = tilt_threshold_degrees
        self.warning_frames = warning_frames
        self._adjustment_streak = 0
        self._warning_active = False
        self._baseline_shoulder_angle: float | None = None
        self._baseline_head_angle: float | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._baseline_shoulder_angle is not None and self._baseline_head_angle is not None

    def calibrate(self, landmarks: Sequence) -> bool:
        """Use the current visible pose as this session's comfortable reference."""
        angles = self._landmark_angles(landmarks)
        if angles is None:
            return False
        self._baseline_shoulder_angle, self._baseline_head_angle = angles
        self._reset_warning()
        return True

    def process(self, landmarks: Sequence) -> PostureResult:
        angles = self._landmark_angles(landmarks)
        if angles is None:
            self._reset_warning()
            return PostureResult(PostureStatus.UNKNOWN, "Keep your head and shoulders in view")

        shoulder_angle, head_angle = angles
        shoulder_tilt = abs(shoulder_angle - (self._baseline_shoulder_angle or 0.0))
        head_tilt = abs(head_angle - (self._baseline_head_angle or 0.0))

        if shoulder_tilt >= self.tilt_threshold_degrees:
            return self._adjustment_result("Level your shoulders", shoulder_tilt, head_tilt)
        if head_tilt >= self.tilt_threshold_degrees:
            return self._adjustment_result("Center your head over your shoulders", shoulder_tilt, head_tilt)

        self._reset_warning()
        return PostureResult(
            PostureStatus.ALIGNED,
            "Head and shoulders look balanced",
            shoulder_tilt,
            head_tilt,
        )

    def _landmark_angles(self, landmarks: Sequence) -> tuple[float, float] | None:
        required = (
            landmarks[self.LEFT_EAR],
            landmarks[self.RIGHT_EAR],
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.RIGHT_SHOULDER],
        )
        if any(getattr(point, "visibility", 1.0) < 0.45 for point in required):
            return None

        left_ear, right_ear, left_shoulder, right_shoulder = required
        shoulder_width = math.hypot(
            right_shoulder.x - left_shoulder.x,
            right_shoulder.y - left_shoulder.y,
        )
        if shoulder_width < 0.08:
            return None

        shoulder_angle = math.degrees(math.atan2(
            right_shoulder.y - left_shoulder.y,
            right_shoulder.x - left_shoulder.x,
        ))
        head_angle = math.degrees(math.atan2(
            right_ear.y - left_ear.y,
            right_ear.x - left_ear.x,
        ))
        return shoulder_angle, head_angle

    def _adjustment_result(self, guidance: str, shoulder_tilt: float, head_tilt: float) -> PostureResult:
        self._adjustment_streak += 1
        warning_just_registered = self._adjustment_streak >= self.warning_frames and not self._warning_active
        if warning_just_registered:
            self._warning_active = True
        return PostureResult(
            PostureStatus.ADJUST,
            guidance,
            shoulder_tilt,
            head_tilt,
            warning_just_registered,
        )

    def _reset_warning(self) -> None:
        self._adjustment_streak = 0
        self._warning_active = False
