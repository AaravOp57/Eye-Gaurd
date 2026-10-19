"""
eyeguard_ai_coach.py
---------------------
AIHealthCoach wraps the Google Gemini API to turn raw session statistics
into a friendly, structured health report for the dashboard.

Design notes:
  * Gemini is asked to return ONLY a JSON object matching a fixed schema.
  * The response is parsed defensively - if Gemini is unreachable, the
    API key is missing/invalid, or the response isn't valid JSON, a local
    rule-based fallback report is returned instead so the app never crashes
    and the dashboard is never left empty.
  * This class performs blocking network I/O internally. Callers (the UI)
    MUST invoke `get_health_report()` on a background thread - see
    eyeguard_ui.py, which wraps every call in a threading.Thread and
    delivers the result back to the main thread via a queue.
  * The AI Coach only ever offers general wellness suggestions. It must
    never claim to diagnose or treat a medical condition - this is
    enforced in the prompt itself and in the fallback text.
  * Multilingual: Gemini is told which language to write in, and every
    local fallback sentence comes from the locales/*.json files via t().
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from i18n import t


def _language_name() -> str:
    return t.language_name()


SYSTEM_INSTRUCTIONS = """You are EyeGuard AI's Health Coach, a friendly digital eye-wellness
assistant embedded in a desktop app. You receive current and historical
screen-time and eye-behavior statistics from a webcam-based monitor and must
respond with ONLY a single
JSON object (no markdown fences, no commentary, no extra text) matching
exactly this schema:

{
  "health_score": <integer 0-100>,
  "analysis": "<one or two sentence plain-language analysis>",
    "top_priority": "<most important screen-habit adjustment>",
    "secondary_priority": "<secondary screen-habit adjustment>",
    "going_well": "<one screen-habit that is going well>",
  "recommendations": ["<short actionable tip>", "..."],
  "exercise": {
    "name": "<short exercise name>",
    "instructions": "<one sentence, easy to follow>",
    "duration_seconds": <integer>,
    "remind_after_minutes": <integer>
  },
  "motivation": "<one short encouraging sentence>"
}

Rules:
- You are a wellness assistant, not a medical professional. Never diagnose,
  never claim to treat any medical or eye condition, and never replace
  professional medical advice. If the stats suggest a persistent problem,
  gently suggest the user consider consulting an eye-care professional
  rather than making any clinical claim yourself.
- Base the health_score and analysis on the statistics provided.
- Compare current values with historical values when available and describe
    changes as screen-habit trends, never as medical improvement.
- Avoid repeating a previous recommendation unless the current data clearly
    calls for it.
- Keep language simple, encouraging, and concise.
- Output ONLY the JSON object, nothing else.
"""


@dataclass
class ExerciseRecommendation:
    name: str = "Blinking Relaxation"
    instructions: str = "Gently close and open your eyes several times."
    duration_seconds: int = 30
    remind_after_minutes: int = 20


@dataclass
class HealthReport:
    health_score: int = 70
    analysis: str = "Keep monitoring your eye habits to build a clearer picture over time."
    top_priority: str = "Build a consistent blink habit during focused work."
    secondary_priority: str = "Take regular screen breaks."
    going_well: str = "You are collecting useful screen-habit data."
    recommendations: List[str] = field(default_factory=lambda: [
        "Blink more frequently, especially during focused screen work.",
        "Maintain a comfortable, arm's-length viewing distance.",
        "Take regular screen breaks using the 20-20-20 rule.",
    ])
    exercise: ExerciseRecommendation = field(default_factory=ExerciseRecommendation)
    motivation: str = "Small habits, practiced consistently, protect your eyes over time!"
    source: str = "fallback"  # "gemini" or "fallback", useful for the UI/debugging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_score": self.health_score,
            "analysis": self.analysis,
            "top_priority": self.top_priority,
            "secondary_priority": self.secondary_priority,
            "going_well": self.going_well,
            "recommendations": self.recommendations,
            "exercise": {
                "name": self.exercise.name,
                "instructions": self.exercise.instructions,
                "duration_seconds": self.exercise.duration_seconds,
                "remind_after_minutes": self.exercise.remind_after_minutes,
            },
            "motivation": self.motivation,
            "source": self.source,
        }


def _localized_defaults() -> HealthReport:
    """Default report text in the CURRENT language (the dataclass defaults above
    are fixed English, so we build a fresh one at call time instead)."""
    return HealthReport(
        analysis=t("coach_default_analysis"),
        top_priority=t("coach_default_top"),
        secondary_priority=t("coach_default_secondary"),
        going_well=t("coach_default_going_well"),
        recommendations=[
            t("coach_default_rec_blink"),
            t("coach_default_rec_distance"),
            t("coach_default_rec_break"),
        ],
        exercise=ExerciseRecommendation(
            name=t("coach_exercise_name"),
            instructions=t("coach_exercise_instructions"),
        ),
        motivation=t("coach_default_motivation"),
    )


class AIHealthCoach:
    """Generates structured wellness reports from session statistics via Gemini."""

    MODEL_NAME = "gemini-2.5-flash"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def set_api_key(self, api_key: str) -> None:
        self.api_key = api_key

    def _build_prompt(self, stats: Dict[str, Any]) -> str:
        return (
            "Here are the current and historical screen-habit statistics:\n"
            f"{json.dumps(stats, indent=2)}\n\n"
            "Respond with the JSON object described in your instructions, "
            "based on these statistics.\n"
            f"Write every text value in {_language_name()}. Keep the JSON keys "
            "exactly as in the schema (in English) and write numbers as digits."
        )

    def get_health_report(self, stats: Dict[str, Any]) -> HealthReport:
        """
        Blocking call - request a health report from Gemini using the given
        statistics dict. MUST be called from a background thread, never the
        GUI thread. Always returns a usable HealthReport, even on failure.
        """
        if not self.api_key or not self.api_key.strip():
            return self._fallback_report(stats, reason="missing_api_key")

        try:
            raw_text = self._call_gemini(stats)
        except Exception:
            return self._fallback_report(stats, reason="api_error")

        parsed = self._safe_parse_json(raw_text)
        if parsed is None:
            return self._fallback_report(stats, reason="invalid_json")

        return self._report_from_dict(parsed)

    def answer_question(self, question: str, stats: Dict[str, Any]) -> str:
        """Answer one user question without exposing a blocking call to the UI."""
        if self.api_key and self.api_key.strip():
            try:
                import google as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(model_name=self.MODEL_NAME)
                response = model.generate_content(
                    "You are a concise eye-wellness coach. Give general, non-diagnostic "
                    f"guidance. Reply in {_language_name()}. Use these current metrics:\n"
                    f"{json.dumps(stats)}\nUser question: {question}"
                )
                if response.text:
                    return response.text.strip()
            except Exception:
                pass

        bpm = stats.get("blinks_per_minute", 0)
        distance = stats.get("distance_status", "UNKNOWN")
        # NOTE: these keyword checks only match English questions. The Gemini
        # path above handles any language; this is just the offline fallback.
        if "blink" in question.lower():
            return t("coach_answer_blink", bpm=bpm)
        if "distance" in question.lower() or "screen" in question.lower():
            return t("coach_answer_distance", distance=distance)
        return t("coach_answer_default")

    def _call_gemini(self, stats: Dict[str, Any]) -> str:
        """Performs the actual (blocking) Gemini API call. Raises on failure."""
        import google as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTIONS,
        )
        response = model.generate_content(self._build_prompt(stats))
        return response.text or ""

    @staticmethod
    def _safe_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse a JSON object from Gemini's raw text response."""
        if not raw_text:
            return None
        text = raw_text.strip()
        # Strip markdown code fences if the model added them despite instructions
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Last resort: find the first {...} block in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> HealthReport:
        """Safely build a HealthReport from a parsed dict, tolerating missing keys."""
        default = _localized_defaults()
        try:
            score = int(data.get("health_score", default.health_score))
            score = max(0, min(100, score))
        except (TypeError, ValueError):
            score = default.health_score

        analysis = str(data.get("analysis", default.analysis)) or default.analysis
        top_priority = str(data.get("top_priority", default.top_priority)) or default.top_priority
        secondary_priority = str(data.get("secondary_priority", default.secondary_priority)) or default.secondary_priority
        going_well = str(data.get("going_well", default.going_well)) or default.going_well

        recs = data.get("recommendations", default.recommendations)
        if not isinstance(recs, list) or not recs:
            recs = default.recommendations
        recs = [str(r) for r in recs][:6]

        ex_data = data.get("exercise", {}) if isinstance(data.get("exercise"), dict) else {}
        try:
            exercise = ExerciseRecommendation(
                name=str(ex_data.get("name", default.exercise.name)),
                instructions=str(ex_data.get("instructions", default.exercise.instructions)),
                duration_seconds=int(ex_data.get("duration_seconds", default.exercise.duration_seconds)),
                remind_after_minutes=int(ex_data.get("remind_after_minutes", default.exercise.remind_after_minutes)),
            )
        except (TypeError, ValueError):
            exercise = default.exercise

        motivation = str(data.get("motivation", default.motivation)) or default.motivation

        return HealthReport(
            health_score=score,
            analysis=analysis,
            top_priority=top_priority,
            secondary_priority=secondary_priority,
            going_well=going_well,
            recommendations=recs,
            exercise=exercise,
            motivation=motivation,
            source="gemini",
        )

    @staticmethod
    def _fallback_report(stats: Dict[str, Any], reason: str) -> HealthReport:
        """
        Local, rule-based report used whenever Gemini can't be reached or
        returns something unusable. Keeps the AI Coach panel populated with
        sensible, non-diagnostic guidance at all times.
        """
        bpm = stats.get("blinks_per_minute", 15)
        distance_status = stats.get("distance_status", "UNKNOWN")
        warnings = stats.get("low_blink_warnings", 0) + stats.get("distance_warnings", 0)

        score = 90
        recs: List[str] = []

        if isinstance(bpm, (int, float)) and bpm < 12:
            score -= 20
            recs.append(t("coach_rec_blink_more"))
        if distance_status == "TOO CLOSE":
            score -= 15
            recs.append(t("coach_rec_move_back"))
        elif distance_status == "TOO FAR":
            recs.append(t("coach_rec_too_far"))
        if isinstance(warnings, (int, float)) and warnings > 5:
            score -= 10
            recs.append(t("coach_rec_longer_break"))

        previous = {
            str(item).strip().lower()
            for item in stats.get("historical", {}).get("previous_recommendations", [])
        }
        recs = [item for item in recs if item.strip().lower() not in previous]
        if not recs:
            recs = [
                t("coach_rec_generic_blink"),
                t("coach_default_rec_distance"),
                t("coach_rec_generic_break"),
            ]

        if isinstance(stats.get("wellness_score"), (int, float)):
            score = int(stats["wellness_score"])
        score = max(0, min(100, score))

        analysis_by_reason = {
            "missing_api_key": t("coach_analysis_no_key"),
            "api_error": t("coach_analysis_api_error"),
            "invalid_json": t("coach_analysis_bad_json"),
        }
        analysis = analysis_by_reason.get(reason, t("coach_analysis_default"))

        top_priority = t("coach_default_top")
        secondary_priority = t("coach_default_secondary")
        going_well = t("coach_default_going_well")
        if isinstance(bpm, (int, float)) and bpm < 12:
            top_priority = t("coach_top_blink")
        elif distance_status == "TOO CLOSE":
            top_priority = t("coach_top_closer")
        if warnings > 0:
            secondary_priority = t("coach_secondary_warnings")
        if distance_status == "GOOD DISTANCE":
            going_well = t("coach_going_well_distance")

        return HealthReport(
            health_score=score,
            analysis=analysis,
            top_priority=top_priority,
            secondary_priority=secondary_priority,
            going_well=going_well,
            recommendations=recs,
            exercise=_localized_defaults().exercise,
            motivation=t("coach_motivation_fallback"),
            source="fallback",
        )
    
