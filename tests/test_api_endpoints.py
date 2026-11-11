"""Integration Tests for FastAPI REST Endpoints."""
import unittest
from fastapi.testclient import TestClient
from app.main import app


class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_start_call_endpoint(self):
        response = self.client.post("/api/call/start")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
        self.assertEqual(data["current_state"], "AWAITING_NAME_AND_REASON")
        self.assertIn("Please state your name and the reason for your call", data["bot_speech_response"])

    def test_process_turn_endpoint(self):
        # Start call first
        start_res = self.client.post("/api/call/start")
        session_id = start_res.json()["session_id"]

        # Send caller statement
        payload = {
            "session_id": session_id,
            "speech_text": "Hi, my name is Carlos Ramirez. I'm calling about the enterprise software demo we discussed."
        }
        res = self.client.post("/api/call/turn", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("bot_speech_response", data)
        self.assertEqual(data["extracted_entities"]["caller_name"], "Carlos Ramirez")
        self.assertIsNotNone(data["intent_analysis"])

    def test_recipient_status_get_and_post(self):
        # Get status
        get_res = self.client.get("/api/recipient/status")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.json()
        self.assertIn("name", data)
        self.assertIn("status", data)

        # Update status to IN_MEETING
        post_res = self.client.post("/api/recipient/status", json={"status": "IN_MEETING"})
        self.assertEqual(post_res.status_code, 200)
        self.assertEqual(post_res.json()["status"], "IN_MEETING")

        # Revert to BUSY
        self.client.post("/api/recipient/status", json={"status": "BUSY"})

    def test_model_metrics_endpoint(self):
        res = self.client.get("/api/model/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("model_name", data)
        self.assertIn("holdout_validation_metrics", data)
        self.assertIn("per_class_metrics", data)

    def test_predict_intent_endpoint(self):
        payload = {"text": "Emergency outage detected on the primary payment cluster!"}
        res = self.client.post("/api/model/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["intent"], "URGENT_INQUIRY")
        self.assertGreaterEqual(data["confidence"], 0.5)


if __name__ == "__main__":
    unittest.main()

