"""CLI Demonstration Runner for Inbound Call AI Agent.

Simulates both conversational paths:
1. Recipient is BUSY -> Solicits callback number and preferred time window.
2. Recipient is AVAILABLE -> Direct live transfer.
"""
import time
from app.models.schemas import RecipientStatusEnum
from app.services.recipient_service import recipient_service
from app.state_machine.call_session import InboundCallSession


def run_cli_demo():
    print("=" * 75)
    print("📞  INBOUND CALL AI AGENT — VOICE SCREENING & ROUTING DEMO")
    print("=" * 75)

    # -------------------------------------------------------------
    # SCENARIO 1: Recipient is BUSY (The primary user scenario)
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("SCENARIO 1: Recipient Alex Mercer is BUSY (In Meeting)")
    print("#" * 60)
    recipient_service.set_recipient_status(RecipientStatusEnum.BUSY)

    session = InboundCallSession()
    turn1 = session.start_call()
    print(f"\n[AI Receptionist]: {turn1.bot_speech_response}")

    caller_statement = "Hi, my name is John Miller and I'm calling regarding the commercial contract proposal."
    print(f"\n[Caller]: {caller_statement}")
    time.sleep(0.5)

    turn2 = session.process_turn(caller_statement)
    print(f"\n[AI Receptionist]: {turn2.bot_speech_response}")
    print(f"   ↳ Extracted Caller Name: {turn2.extracted_entities.caller_name}")
    print(f"   ↳ Extracted Reason: {turn2.extracted_entities.reason_for_calling}")
    print(f"   ↳ Predicted Intent: {turn2.intent_analysis.intent.value} ({turn2.intent_analysis.confidence * 100:.1f}%)")

    caller_callback_info = "My phone number is 415-555-0199 and the best time to reach me is tomorrow at 2 PM."
    print(f"\n[Caller]: {caller_callback_info}")
    time.sleep(0.5)

    turn3 = session.process_turn(caller_callback_info)
    print(f"\n[AI Receptionist]: {turn3.bot_speech_response}")
    print(f"   ↳ Extracted Phone: {turn3.extracted_entities.phone_number}")
    print(f"   ↳ Extracted Best Time: {turn3.extracted_entities.best_time_to_reach}")
    print(f"\n[Recipient Notification Dispatched]:\n{turn3.notification_preview}")

    # -------------------------------------------------------------
    # SCENARIO 2: Recipient is AVAILABLE
    # -------------------------------------------------------------
    print("\n" + "#" * 60)
    print("SCENARIO 2: Recipient Alex Mercer is AVAILABLE")
    print("#" * 60)
    recipient_service.set_recipient_status(RecipientStatusEnum.AVAILABLE)

    session2 = InboundCallSession()
    turn1_avail = session2.start_call()
    print(f"\n[AI Receptionist]: {turn1_avail.bot_speech_response}")

    caller_avail = "Hello, this is Dr. Sarah Jenkins calling to discuss the Q3 clinical research partnership."
    print(f"\n[Caller]: {caller_avail}")
    time.sleep(0.5)

    turn2_avail = session2.process_turn(caller_avail)
    print(f"\n[AI Receptionist]: {turn2_avail.bot_speech_response}")
    print(f"   ↳ State: {turn2_avail.current_state.value}")
    print(f"\n[Recipient Live Transfer Notification]:\n{turn2_avail.notification_preview}")

    print("\n" + "=" * 75)
    print("🎉 Demo complete! Both scenarios passed cleanly.")
    print("=" * 75)


if __name__ == "__main__":
    run_cli_demo()

