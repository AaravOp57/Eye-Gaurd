import unittest
from unittest.mock import patch

from eyeguard_ai_coach import AIHealthCoach
from i18n import t


class AIHealthCoachQuestionTests(unittest.TestCase):
    def setUp(self):
        self.original_language = t.current_lang
        t.set_language("en")

    def tearDown(self):
        t.set_language(self.original_language)

    def test_question_prompt_includes_concern_metrics_and_safety_guidance(self):
        coach = AIHealthCoach("test-key")
        prompt = coach._build_question_prompt(
            "My eyes feel tired after reading. What can I do?",
            {"blinks_per_minute": 10},
            [{"role": "user", "content": "My screen feels too bright."},
             {"role": "assistant", "content": "Try lowering brightness."}],
        )

        self.assertIn("My eyes feel tired after reading", prompt)
        self.assertIn('"blinks_per_minute": 10', prompt)
        self.assertIn("User: My screen feels too bright.", prompt)
        self.assertIn("Coach: Try lowering brightness.", prompt)
        self.assertIn("Do not diagnose", prompt)
        self.assertIn("urgent medical care", prompt)

    def test_question_is_sent_to_the_ai_and_response_returned(self):
        coach = AIHealthCoach("test-key")
        with patch.object(coach, "_call_gemini_question", return_value="Try resting your eyes.") as call:
            answer = coach.answer_question("My eyes feel tired", {"blinks_per_minute": 10})

        self.assertEqual(answer, "Try resting your eyes.")
        call.assert_called_once_with("My eyes feel tired", {"blinks_per_minute": 10}, [])

    def test_missing_key_returns_actionable_message(self):
        coach = AIHealthCoach()

        answer = coach.answer_question("My eyes feel tired", {})

        self.assertIn("API key", answer)

    def test_api_failure_is_visible_instead_of_returning_generic_advice(self):
        coach = AIHealthCoach("test-key")
        with patch.object(coach, "_call_gemini_question", side_effect=TimeoutError):
            answer = coach.answer_question("My eyes feel tired", {})

        self.assertIn("couldn't reach the AI service", answer)


if __name__ == "__main__":
    unittest.main()
