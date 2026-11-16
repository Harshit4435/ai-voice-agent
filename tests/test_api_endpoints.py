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

    def test_pipecat_pipeline_endpoints(self):
        # 1. Pipecat Status
        status_res = self.client.get("/api/pipecat/status")
        self.assertEqual(status_res.status_code, 200)
        self.assertEqual(status_res.json()["status"], "active")
        self.assertEqual(status_res.json()["engine"], "Pipecat Voice AI Framework")

        # 2. Pipecat Start
        start_res = self.client.post("/api/pipecat/start")
        self.assertEqual(start_res.status_code, 200)
        data = start_res.json()
        self.assertIn("session_id", data)
        self.assertEqual(data["current_state"], "AWAITING_NAME_AND_REASON")
        session_id = data["session_id"]

        # 3. Pipecat Turn 1
        t1_res = self.client.post("/api/pipecat/turn", json={
            "session_id": session_id,
            "speech_text": "Hi this is Harshad calling for a contract"
        })
        self.assertEqual(t1_res.status_code, 200)
        t1_data = t1_res.json()
        self.assertEqual(t1_data["extracted_entities"]["caller_name"], "Harshad")
        self.assertEqual(t1_data["extracted_entities"]["reason_for_calling"], "a contract")
        self.assertEqual(t1_data["current_state"], "AWAITING_CALLBACK_DETAILS")

        # 4. Pipecat Turn 2
        t2_res = self.client.post("/api/pipecat/turn", json={
            "session_id": session_id,
            "speech_text": "My phone is 555-019-2834 and call me tomorrow at 2"
        })
        self.assertEqual(t2_res.status_code, 200)
        t2_data = t2_res.json()
        self.assertEqual(t2_data["current_state"], "CALL_COMPLETED")
        self.assertIsNotNone(t2_data["callback_record"])

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

