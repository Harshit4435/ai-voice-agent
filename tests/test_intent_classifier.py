"""Unit Tests for Intent Classifier Inference and Calibration."""
import unittest
from app.ml.intent_classifier import InboundIntentClassifier
from app.models.schemas import InboundIntentEnum


class TestIntentClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = InboundIntentClassifier()

    def test_predict_name_and_reason(self):
        text = "Hello, my name is John Miller and I'm calling regarding the enterprise software contract."
        result = self.classifier.predict(text)
        self.assertEqual(result.intent, InboundIntentEnum.STATEMENT_NAME_AND_REASON)
        self.assertGreaterEqual(result.confidence, 0.4)
        self.assertIn("name is", result.matched_keywords)

    def test_predict_urgent_inquiry(self):
        text = "This is an emergency, our main production server cluster has crashed!"
        result = self.classifier.predict(text)
        self.assertEqual(result.intent, InboundIntentEnum.URGENT_INQUIRY)
        self.assertGreaterEqual(result.confidence, 0.5)

    def test_predict_leave_callback_info(self):
        text = "You can reach me at 415-555-0199, and the best time to call is tomorrow at 2 PM."
        result = self.classifier.predict(text)
        self.assertEqual(result.intent, InboundIntentEnum.LEAVE_CALLBACK_INFO)
        self.assertGreaterEqual(result.confidence, 0.4)

    def test_confidence_calibration_probabilities(self):
        text = "I would like to schedule a 30-minute callback meeting with Alex."
        result = self.classifier.predict(text)
        self.assertIsInstance(result.all_probabilities, dict)
        self.assertGreater(len(result.all_probabilities), 0)
        # Sum of probabilities across classes should approximate 1.0
        prob_sum = sum(result.all_probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_empty_string_handling(self):
        result = self.classifier.predict("")
        self.assertIsNotNone(result.intent)
        self.assertGreaterEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
