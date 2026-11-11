"""Unit Tests for Recipient Service and Notification Logging."""
import unittest
from app.models.schemas import RecipientStatusEnum
from app.services.recipient_service import RecipientService


class TestRecipientService(unittest.TestCase):
    def setUp(self):
        self.service = RecipientService()

    def test_default_recipient_state(self):
        profile = self.service.get_recipient_profile()
        self.assertEqual(profile.name, "Vittoris Team")
        self.assertEqual(profile.status, RecipientStatusEnum.BUSY)
        self.assertFalse(self.service.is_recipient_available())

    def test_set_recipient_status_available(self):
        self.service.set_recipient_status(RecipientStatusEnum.AVAILABLE)
        self.assertTrue(self.service.is_recipient_available())

    def test_dispatch_transfer_notification(self):
        notif = self.service.dispatch_transfer_notification(
            caller_name="John Miller", reason="commercial contract proposal"
        )
        self.assertIn("LIVE CALL TRANSFER", notif)
        self.assertIn("John Miller", notif)
        self.assertIn("commercial contract proposal", notif)

    def test_record_callback_request(self):
        record = self.service.record_callback_request(
            call_id="call_test_123",
            caller_name="Sarah Connor",
            phone_number="415-555-0199",
            best_time="tomorrow at 2 PM",
            reason="production database recovery",
            urgency="URGENT",
        )
        self.assertEqual(record.caller_name, "Sarah Connor")
        self.assertEqual(record.phone_number, "415-555-0199")
        self.assertEqual(record.urgency, "URGENT")
        self.assertTrue(record.notification_dispatched)

        inbox = self.service.get_all_callbacks()
        self.assertGreaterEqual(len(inbox), 1)
        self.assertEqual(inbox[0].call_id, "call_test_123")


if __name__ == "__main__":
    unittest.main()

