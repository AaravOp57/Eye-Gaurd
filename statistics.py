"""
statistics.py
--------------
Tracks session and long-term statistics for EyeGuard AI and persists
them locally as JSON. Two layers are kept:

  * SessionStats  - resets every time the app starts a new monitoring session
  * StatisticsManager - loads/saves cumulative history to disk and exposes
                          convenience methods for updating it during a session
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STATS_PATH = os.path.join(STATS_DIR, "statistics.json")


@dataclass
class SessionStats:
    """Live statistics for the current monitoring session (in-memory only)."""

    start_time: float = field(default_factory=time.time)
    blink_count: int = 0
    blink_timestamps: List[float] = field(default_factory=list)  # for rolling BPM
    distance_samples: List[str] = field(default_factory=list)    # status strings
    posture_samples: List[str] = field(default_factory=list)
    low_blink_warnings: int = 0
    distance_warnings: int = 0
    posture_warnings: int = 0
    exercises_completed: int = 0
    breaks_taken: int = 0
    warning_events: List[Dict[str, Any]] = field(default_factory=list)
    health_score_history: List[int] = field(default_factory=list)
    metric_history: List[Dict[str, Any]] = field(default_factory=list)

    def session_duration_seconds(self) -> float:
        return time.time() - self.start_time

    def blinks_per_minute(self, window_seconds: float = 60.0) -> float:
        """Rolling blinks-per-minute based on blinks within the last window."""
        now = time.time()
        recent = [t for t in self.blink_timestamps if now - t <= window_seconds]
        # Trim the stored list occasionally so it doesn't grow unbounded
        if len(recent) != len(self.blink_timestamps):
            self.blink_timestamps = recent
        elapsed = min(now - self.start_time, window_seconds)
        # Guard against a near-zero elapsed time (e.g. right at session start),
        # which would otherwise blow the rate up to an absurd value for a
        # single early blink. Require at least a few seconds of data before
        # reporting a rate at all.
        if elapsed < 3.0:
            return 0.0
        rate = len(recent) * (60.0 / elapsed)
        # Blinking faster than ~60/min sustained is not realistic - clamp to
        # avoid noisy landmark jitter producing an implausible spike.
        return min(rate, 60.0)

    def dominant_distance_status(self) -> str:
        if not self.distance_samples:
            return "UNKNOWN"
        return max(set(self.distance_samples), key=self.distance_samples.count)

    def register_blink(self) -> None:
        self.blink_count += 1
        self.blink_timestamps.append(time.time())

    def register_distance_sample(self, status: str) -> None:
        self.distance_samples.append(status)
        # Keep memory bounded - only need a rolling window for "dominant" status
        if len(self.distance_samples) > 600:
            self.distance_samples = self.distance_samples[-600:]

    def register_posture_sample(self, status: str) -> None:
        if status == "UPPER BODY NOT DETECTED":
            return
        self.posture_samples.append(status)
        if len(self.posture_samples) > 600:
            self.posture_samples = self.posture_samples[-600:]

    def posture_alignment_rate(self) -> Optional[float]:
        if not self.posture_samples:
            return None
        aligned = sum(1 for status in self.posture_samples if status == "ALIGNED")
        return round(aligned / len(self.posture_samples) * 100, 1)

    def record_metric_sample(self, blink_rate: float, distance_status: str,
                             health_score: Optional[int] = None,
                             warning_count: int = 0) -> None:
        """Store sparse chart samples, keeping the live history bounded."""
        sample = {
            "timestamp": time.time(),
            "blink_rate": round(float(blink_rate), 1),
            "distance_status": distance_status,
            "health_score": int(health_score) if health_score is not None else None,
            "warning_count": int(warning_count),
        }
        self.metric_history.append(sample)
        self.metric_history = self.metric_history[-720:]

    def register_warning(self, topic: str) -> None:
        self.warning_events.append({"timestamp": time.time(), "topic": topic})
        self.warning_events = self.warning_events[-300:]


class StatisticsManager:
    """Persists cumulative statistics across sessions as JSON."""

    def __init__(self, path: str = STATS_PATH) -> None:
        self._path = path
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._data: Dict[str, Any] = self._load()
        self.session = SessionStats()

    def _default_data(self) -> Dict[str, Any]:
        return {
            "total_sessions": 0,
            "total_blinks": 0,
            "total_session_seconds": 0.0,
            "total_warnings": 0,
            "total_exercises_completed": 0,
            "total_breaks_taken": 0,
            "average_blink_rate_history": [],
            "health_score_history": [],
            "last_session_summary": {},
            "session_history": [],
        }

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self._path):
            return self._default_data()
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            defaults = self._default_data()
            return {**defaults, **data}
        except (json.JSONDecodeError, OSError, TypeError):
            return self._default_data()

    def save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except OSError:
            pass

    def start_new_session(self) -> None:
        self.session = SessionStats()

    # ---- Session-level convenience methods used by the detectors/UI ----

    def register_blink(self) -> None:
        self.session.register_blink()

    def register_distance_sample(self, status: str) -> None:
        self.session.register_distance_sample(status)

    def register_low_blink_warning(self) -> None:
        self.session.low_blink_warnings += 1

    def register_distance_warning(self) -> None:
        self.session.distance_warnings += 1

    def register_posture_sample(self, status: str) -> None:
        self.session.register_posture_sample(status)

    def register_posture_warning(self) -> None:
        self.session.posture_warnings += 1

    def register_exercise_completed(self) -> None:
        self.session.exercises_completed += 1

    def register_break_taken(self) -> None:
        self.session.breaks_taken += 1

    def register_warning(self, topic: str) -> None:
        self.session.register_warning(topic)

    def calculate_wellness_score(self, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a transparent, non-medical screen-habit score and its factors."""
        snapshot = snapshot or self.get_dashboard_snapshot()
        score = 100
        factors: List[Dict[str, Any]] = []
        blink_rate = float(snapshot.get("blinks_per_minute", 0))
        baseline_blink_rate = snapshot.get("baseline_blink_rate")
        if baseline_blink_rate and blink_rate:
            baseline_blink_rate = float(baseline_blink_rate)
            if blink_rate < baseline_blink_rate * 0.85:
                factors.append({"delta": -20, "reason": "Below your personal blink baseline"})
            elif blink_rate >= baseline_blink_rate * 1.05:
                factors.append({"delta": 5, "reason": "Above your personal blink baseline"})
        elif blink_rate and blink_rate < 12:
            factors.append({"delta": -20, "reason": "Lower blink rate"})
        elif blink_rate >= 15:
            factors.append({"delta": 5, "reason": "Better blink rate"})
        distance = snapshot.get("distance_status")
        if distance == "TOO CLOSE":
            factors.append({"delta": -12, "reason": "Frequent close-screen status"})
        elif distance == "GOOD DISTANCE":
            factors.append({"delta": 4, "reason": "Comfortable screen distance"})
        minutes = float(snapshot.get("session_duration_minutes", 0))
        if minutes >= 60:
            factors.append({"delta": -8, "reason": "Long continuous session"})
        elif minutes >= 20 and snapshot.get("breaks_taken", 0) == 0:
            factors.append({"delta": -4, "reason": "No break recorded yet"})
        posture_rate = snapshot.get("posture_alignment_rate")
        if posture_rate is not None and posture_rate < 65:
            factors.append({"delta": -8, "reason": "Frequent posture adjustments"})
        elif posture_rate is not None and posture_rate >= 85:
            factors.append({"delta": 4, "reason": "Balanced posture"})
        warnings = (
            int(snapshot.get("low_blink_warnings", 0))
            + int(snapshot.get("distance_warnings", 0))
            + int(snapshot.get("posture_warnings", 0))
        )
        if warnings >= 5:
            factors.append({"delta": -8, "reason": "Frequent wellness warnings"})
        if snapshot.get("breaks_taken", 0) > 0:
            factors.append({"delta": 5, "reason": "Breaks taken"})
        score = max(0, min(100, score + sum(item["delta"] for item in factors)))
        return {"score": score, "factors": factors}

    def record_metric_sample(self, blink_rate: float, distance_status: str,
                             health_score: Optional[int] = None,
                             warning_count: int = 0) -> None:
        """Record a sparse chart sample through the manager facade."""
        self.session.record_metric_sample(blink_rate, distance_status, health_score, warning_count)

    def register_health_score(self, score: int) -> None:
        self.session.health_score_history.append(score)
        self.record_metric_sample(
            self.session.blinks_per_minute(),
            self.session.dominant_distance_status(),
            score,
        )

    def get_metric_history(self) -> List[Dict[str, Any]]:
        return list(self.session.metric_history)

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        """A snapshot dict suitable for display in the UI or sending to Gemini."""
        baseline = self._get_baseline()
        return {
            "blink_count": self.session.blink_count,
            "blinks_per_minute": round(self.session.blinks_per_minute(), 1),
            "distance_status": self.session.dominant_distance_status(),
            "session_duration_minutes": round(self.session.session_duration_seconds() / 60, 1),
            "low_blink_warnings": self.session.low_blink_warnings,
            "distance_warnings": self.session.distance_warnings,
            "posture_warnings": self.session.posture_warnings,
            "posture_alignment_rate": self.session.posture_alignment_rate(),
            "exercises_completed": self.session.exercises_completed,
            "breaks_taken": self.session.breaks_taken,
            "warning_count": self.session.low_blink_warnings + self.session.distance_warnings + self.session.posture_warnings,
            "baseline_blink_rate": baseline.get("blink_rate"),
            "baseline_distance_status": baseline.get("typical_distance", "UNKNOWN"),
        }

    def _get_baseline(self) -> Dict[str, Any]:
        sessions = self._data.get("session_history", [])[-7:]
        rates = [float(item.get("blinks_per_minute", 0)) for item in sessions if item.get("blinks_per_minute") is not None]
        distances = [item.get("distance_status", "UNKNOWN") for item in sessions]
        return {
            "blink_rate": round(sum(rates) / len(rates), 1) if rates else None,
            "typical_distance": max(set(distances), key=distances.count) if distances else "UNKNOWN",
        }

    def end_session_and_persist(self) -> None:
        """Fold the current session into cumulative history and save to disk."""
        snapshot = self.get_dashboard_snapshot()
        self._data["total_sessions"] += 1
        self._data["total_blinks"] += self.session.blink_count
        self._data["total_session_seconds"] += self.session.session_duration_seconds()
        self._data["total_warnings"] += (
            self.session.low_blink_warnings + self.session.distance_warnings + self.session.posture_warnings
        )
        self._data["total_exercises_completed"] += self.session.exercises_completed
        self._data["total_breaks_taken"] += self.session.breaks_taken
        self._data["average_blink_rate_history"].append(snapshot["blinks_per_minute"])
        self._data["health_score_history"].extend(self.session.health_score_history)
        self._data["last_session_summary"] = snapshot
        self._data["session_history"].append({
            "ended_at": time.time(),
            **snapshot,
            "health_score": (
                round(sum(self.session.health_score_history) / len(self.session.health_score_history), 1)
                if self.session.health_score_history else None
            ),
            "breaks_taken": self.session.breaks_taken,
            "warning_events": list(self.session.warning_events),
        })
        # Keep history lists from growing without bound
        self._data["average_blink_rate_history"] = self._data["average_blink_rate_history"][-200:]
        self._data["health_score_history"] = self._data["health_score_history"][-200:]
        self._data["session_history"] = self._data["session_history"][-200:]
        self.save()

    def get_cumulative_data(self) -> Dict[str, Any]:
        return dict(self._data)

    def get_period_summary(self, days: int) -> Dict[str, Any]:
        """Return persisted session totals for a selectable calendar window."""
        if days <= 1:
            return self.get_dashboard_snapshot()
        cutoff = time.time() - days * 86400
        sessions = [s for s in self._data.get("session_history", []) if s.get("ended_at", 0) >= cutoff]
        if not sessions:
                return {"blink_count": 0, "blinks_per_minute": 0.0, "session_duration_minutes": 0.0,
                    "low_blink_warnings": 0, "distance_warnings": 0, "exercises_completed": 0,
                    "posture_warnings": 0, "posture_alignment_rate": None, "breaks_taken": 0,
                    "distance_status": "UNKNOWN", "health_score": None, "warning_count": 0}
        total_minutes = sum(float(s.get("session_duration_minutes", 0)) for s in sessions)
        total_blinks = sum(int(s.get("blink_count", 0)) for s in sessions)
        return {
            "blink_count": total_blinks,
            "blinks_per_minute": round(total_blinks / (total_minutes or 1) , 1),
            "session_duration_minutes": round(total_minutes, 1),
            "low_blink_warnings": sum(int(s.get("low_blink_warnings", 0)) for s in sessions),
            "distance_warnings": sum(int(s.get("distance_warnings", 0)) for s in sessions),
            "posture_warnings": sum(int(s.get("posture_warnings", 0)) for s in sessions),
            "posture_alignment_rate": (
                round(sum(float(s["posture_alignment_rate"]) for s in sessions if s.get("posture_alignment_rate") is not None)
                      / len([s for s in sessions if s.get("posture_alignment_rate") is not None]), 1)
                if any(s.get("posture_alignment_rate") is not None for s in sessions) else None
            ),
            "exercises_completed": sum(int(s.get("exercises_completed", 0)) for s in sessions),
            "breaks_taken": sum(int(s.get("breaks_taken", 0)) for s in sessions),
            "distance_status": sessions[-1].get("distance_status", "UNKNOWN"),
            "health_score": sessions[-1].get("health_score"),
            "warning_count": sum(
                int(s.get("low_blink_warnings", 0)) + int(s.get("distance_warnings", 0))
                + int(s.get("posture_warnings", 0)) for s in sessions
            ),
        }

    def get_historical_context(self) -> Dict[str, Any]:
        """Provide compact prior-session context for adaptive coaching."""
        sessions = self._data.get("session_history", [])
        recent = sessions[-7:]
        rates = [float(item.get("blinks_per_minute", 0)) for item in recent]
        distances = [item.get("distance_status", "UNKNOWN") for item in recent]
        scores = [item.get("health_score") for item in recent if item.get("health_score") is not None]
        warning_counts = [
            int(item.get("low_blink_warnings", 0)) + int(item.get("distance_warnings", 0))
            for item in recent
        ]
        durations = [float(item.get("session_duration_minutes", 0)) for item in recent]
        good_distance_count = sum(1 for item in distances if item == "GOOD DISTANCE")
        return {
            "previous_sessions": len(sessions),
            "baseline": {
                "blink_rate": round(sum(rates) / len(rates), 1) if rates else None,
                "typical_distance": (
                    max(set(distances), key=distances.count) if distances else "UNKNOWN"
                ),
                "average_session_minutes": round(sum(durations) / len(durations), 1) if durations else None,
                "average_warnings": round(sum(warning_counts) / len(warning_counts), 1) if warning_counts else None,
                "average_breaks": round(
                    sum(int(item.get("breaks_taken", 0)) for item in recent) / len(recent), 1
                ) if recent else None,
            },
            "recent_average_blink_rate": round(sum(rates) / len(rates), 1) if rates else None,
            "recent_average_session_minutes": round(sum(float(item.get("session_duration_minutes", 0)) for item in recent) / len(recent), 1) if recent else None,
            "recent_average_warnings": round(sum(int(item.get("low_blink_warnings", 0)) + int(item.get("distance_warnings", 0)) for item in recent) / len(recent), 1) if recent else None,
            "recent_average_breaks": round(sum(int(item.get("breaks_taken", 0)) for item in recent) / len(recent), 1) if recent else None,
            "recent_average_score": round(sum(scores) / len(scores), 1) if scores else None,
            "recent_distance_statuses": distances,
            "distance_consistency": round(good_distance_count / len(distances), 2) if distances else None,
            "previous_recommendations": self._data.get("previous_recommendations", [])[-5:],
        }

    def remember_recommendations(self, recommendations: List[str]) -> None:
        self._data["previous_recommendations"] = (self._data.get("previous_recommendations", []) + recommendations)[-20:]
        self.save()

    def get_warning_heatmap(self) -> Dict[int, int]:
        heatmap: Dict[int, int] = {}
        for session in self._data.get("session_history", []):
            for event in session.get("warning_events", []):
                hour = time.localtime(float(event.get("timestamp", 0))).tm_hour
                heatmap[hour] = heatmap.get(hour, 0) + 1
        return heatmap

    def get_timeline_events(self) -> List[Dict[str, Any]]:
        """Return saved and current-session warning events in chronological order."""
        events: List[Dict[str, Any]] = []
        for session in self._data.get("session_history", []):
            for event in session.get("warning_events", []):
                events.append({
                    "timestamp": float(event.get("timestamp", 0)),
                    "topic": str(event.get("topic", "Screen-habit event")),
                })
        for event in self.session.warning_events:
            events.append({
                "timestamp": float(event.get("timestamp", 0)),
                "topic": str(event.get("topic", "Screen-habit event")),
            })
        return sorted(events, key=lambda item: item["timestamp"])

    def get_trends(self) -> Dict[str, Optional[float]]:
        sessions = self._data.get("session_history", [])
        if len(sessions) < 2:
            return {"blink_rate": None, "warnings": None, "breaks": None}
        current = sessions[-7:]
        previous = sessions[-14:-7]
        if not previous:
            return {"blink_rate": None, "warnings": None, "breaks": None}

        def average(items: List[Dict[str, Any]], key) -> float:
            return sum(float(key(item)) for item in items) / len(items)

        current_warnings = lambda item: int(item.get("low_blink_warnings", 0)) + int(item.get("distance_warnings", 0))
        return {
            "blink_rate": round((average(current, lambda item: item.get("blinks_per_minute", 0)) / (average(previous, lambda item: item.get("blinks_per_minute", 0)) or 1) - 1) * 100, 1),
            "warnings": round((average(current, current_warnings) / (average(previous, current_warnings) or 1) - 1) * 100, 1),
            "breaks": round((average(current, lambda item: item.get("breaks_taken", 0)) / (average(previous, lambda item: item.get("breaks_taken", 0)) or 1) - 1) * 100, 1),
        }

    def get_achievements(self) -> List[str]:
        sessions = self._data.get("session_history", [])
        achievements: List[str] = []
        if sessions:
            achievements.append("First Session")
        if self._data.get("total_breaks_taken", 0) >= 1:
            achievements.append("Touch Grass")
        if self._data.get("total_breaks_taken", 0) >= 10:
            achievements.append("10 Breaks Taken")
        if self._data.get("total_exercises_completed", 0) >= 10:
            achievements.append("Completed 10 Exercises")
        if len(sessions) >= 7:
            achievements.append("7-Session Tracking Streak")
        if len(sessions) >= 2 and sessions[-1].get("blinks_per_minute", 0) > sessions[-2].get("blinks_per_minute", 0):
            achievements.append("Improved Blink Habits")
        return achievements
