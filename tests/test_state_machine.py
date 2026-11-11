"""Unit Tests for Inbound Call State Machine (Available vs Busy Paths)."""
import unittest
from app.models.schemas import CallStateEnum, RecipientStatusEnum
from app.services.recipient_service import recipient_service
from app.state_machine.call_session import InboundCallSession


class TestCallStateMachine(unittest.TestCase):
    def test_busy_recipient_callback_flow(self):
        """Tests the full dialogue flow when the recipient is busy."""
        # 1. Ensure recipient is BUSY
        recipient_service.set_recipient_status(RecipientStatusEnum.BUSY)

        session = InboundCallSession()
        turn1 = session.start_call()

        # Check initial greeting
        self.assertEqual(turn1.current_state, CallStateEnum.AWAITING_NAME_AND_REASON)
        self.assertIn("Please state your name and the reason for your call", turn1.bot_speech_response)

        # 2. Caller states name and reason
        turn2 = session.process_turn(
            "Hi, my name is John Miller and I'm calling regarding the commercial contract proposal."
        )

        # Recipient is busy, so AI should ask for phone number and best time
        self.assertEqual(turn2.current_state, CallStateEnum.AWAITING_CALLBACK_DETAILS)
        self.assertIn("John Miller", turn2.bot_speech_response)
        self.assertIn("currently busy", turn2.bot_speech_response)
        self.assertIn("leave your phone number and the best time", turn2.bot_speech_response)

        # 3. Caller leaves callback information
        turn3 = session.process_turn(
            "My phone number is 415-555-0199 and the best time to reach me is tomorrow at 2 PM."
        )

        # Call should now be completed and callback recorded
        self.assertEqual(turn3.current_state, CallStateEnum.CALL_COMPLETED)
        self.assertIn("4 1 5, 5 5 5, 0 1 9 9", turn3.bot_speech_response)
        self.assertIn("tomorrow at 2 PM", turn3.bot_speech_response)
        self.assertIsNotNone(turn3.callback_record)
        self.assertEqual(turn3.callback_record.caller_name, "John Miller")
        self.assertEqual(turn3.callback_record.phone_number, "415-555-0199")

    def test_available_recipient_transfer_flow(self):
        """Tests the full dialogue flow when the recipient is available."""
        # 1. Set recipient to AVAILABLE
        recipient_service.set_recipient_status(RecipientStatusEnum.AVAILABLE)

        session = InboundCallSession()
        turn1 = session.start_call()
        self.assertEqual(turn1.current_state, CallStateEnum.AWAITING_NAME_AND_REASON)

        # 2. Caller states name and reason
        turn2 = session.process_turn(
            "Hello, this is Dr. Sarah Jenkins calling to discuss the clinical trial partnership."
        )

        # Recipient is available, so AI should initiate live transfer
        self.assertEqual(turn2.current_state, CallStateEnum.TRANSFERRING_CALL)
        self.assertIn("Sarah Jenkins", turn2.bot_speech_response)
        self.assertIn("available right now", turn2.bot_speech_response)
        self.assertIn("connect your call directly", turn2.bot_speech_response)
        self.assertIsNotNone(turn2.notification_preview)
        self.assertIn("LIVE CALL TRANSFER", turn2.notification_preview)

        # Reset status back to BUSY for subsequent tests
        recipient_service.set_recipient_status(RecipientStatusEnum.BUSY)


if __name__ == "__main__":
    unittest.main()

