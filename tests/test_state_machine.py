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

    def test_repeat_caller_treated_as_fresh_new_user(self):
        """Tests that when the same person calls/logins again, they are treated as a fresh new user."""
        recipient_service.set_recipient_status(RecipientStatusEnum.BUSY)

        session = InboundCallSession(session_id="repeat_user_session")
        session.start_call()

        # Caller provides details in call 1
        session.process_turn("Hi, my name is Alice Smith and I'm calling about billing services.")
        session.process_turn("My phone is 650-555-0144 and best time is Monday morning.")
        self.assertEqual(session.state, CallStateEnum.CALL_COMPLETED)
        self.assertEqual(session.caller_entities.caller_name, "Alice Smith")

        # Now, the same person logs in / starts call again
        turn_new = session.start_call()

        # Must be treated as a brand new user: all entities cleared, asking for all details again
        self.assertEqual(turn_new.current_state, CallStateEnum.AWAITING_NAME_AND_REASON)
        self.assertIsNone(session.caller_entities.caller_name)
        self.assertIsNone(session.caller_entities.reason_for_calling)
        self.assertIsNone(session.caller_entities.phone_number)
        self.assertIsNone(session.caller_entities.best_time_to_reach)
        self.assertIn("Please state your name and the reason for your call", turn_new.bot_speech_response)

    def test_requires_all_details_step_by_step(self):
        """Tests that the agent asks for every missing detail rather than assuming or skipping."""
        recipient_service.set_recipient_status(RecipientStatusEnum.BUSY)
        session = InboundCallSession()
        session.start_call()

        # Step 1: Caller only gives their name
        turn1 = session.process_turn("Hello, this is Michael Chang.")
        self.assertEqual(turn1.current_state, CallStateEnum.AWAITING_NAME_AND_REASON)
        self.assertEqual(session.caller_entities.caller_name, "Michael Chang")
        self.assertIn("Michael Chang", turn1.bot_speech_response)
        self.assertIn("reason for your call", turn1.bot_speech_response)

        # Step 2: Caller now provides the reason
        turn2 = session.process_turn("I am calling regarding an enterprise API license.")
        self.assertEqual(turn2.current_state, CallStateEnum.AWAITING_CALLBACK_DETAILS)
        self.assertEqual(session.caller_entities.reason_for_calling, "an enterprise API license")
        self.assertIn("leave your phone number and the best time", turn2.bot_speech_response)

        # Step 3: Caller gives phone only, missing best time
        turn3 = session.process_turn("My phone number is 415-555-9988.")
        self.assertEqual(turn3.current_state, CallStateEnum.AWAITING_CALLBACK_DETAILS)
        self.assertEqual(session.caller_entities.phone_number, "415-555-9988")
        self.assertIn("best time", turn3.bot_speech_response)

        # Step 4: Caller gives best time
        turn4 = session.process_turn("The best time is tomorrow afternoon at 3 PM.")
        self.assertEqual(turn4.current_state, CallStateEnum.CALL_COMPLETED)
        self.assertIsNotNone(turn4.callback_record)
        self.assertEqual(turn4.callback_record.caller_name, "Michael Chang")
        self.assertEqual(turn4.callback_record.phone_number, "415-555-9988")


if __name__ == "__main__":
    unittest.main()

