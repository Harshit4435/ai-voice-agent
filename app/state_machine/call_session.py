"""Inbound Call State Machine.

Orchestrates telephony dialogue turns:
1. Greets caller: "Please state your name and reason for calling."
2. Extracts caller name and purpose; classifies intent.
3. Checks recipient availability (AVAILABLE vs BUSY).
4. If AVAILABLE -> Announces live call transfer to recipient.
5. If BUSY -> Solicits callback phone number and preferred time window.
6. Records callback message and dispatches notification to recipient.
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, Optional

from app.ml.entity_extractor import entity_extractor
from app.ml.intent_classifier import intent_classifier
from app.models.schemas import (
    CallbackRecord,
    CallStateEnum,
    CallTurnResponse,
    ExtractedEntities,
    IntentClassificationResult,
    RecipientStatusEnum,
)
from app.services.recipient_service import recipient_service

logger = logging.getLogger("inbound_ai.state_machine")


class InboundCallSession:
    """Manages an individual inbound telephone call lifecycle."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.state = CallStateEnum.GREETING
        self.caller_entities = ExtractedEntities()
        self.history: list[dict[str, str]] = []
        self.latest_intent: Optional[IntentClassificationResult] = None
        self.callback_record: Optional[CallbackRecord] = None

    def start_call(self) -> CallTurnResponse:
        """Initial inbound call trigger when telephone rings and connects."""
        self.state = CallStateEnum.AWAITING_NAME_AND_REASON
        greeting = (
            "Thank you for calling. Please state your name and the reason for your call."
        )
        self.history.append({"speaker": "AI", "text": greeting})
        recipient = recipient_service.get_recipient_profile()

        return CallTurnResponse(
            session_id=self.session_id,
            current_state=self.state,
            bot_speech_response=greeting,
            recipient_status=recipient.status,
            extracted_entities=self.caller_entities,
            intent_analysis=None,
            callback_record=None,
            notification_preview=None,
        )

    def process_turn(self, caller_text: str) -> CallTurnResponse:
        """Processes each conversational turn spoken by the caller."""
        self.history.append({"speaker": "CALLER", "text": caller_text})
        recipient = recipient_service.get_recipient_profile()
        notification_preview: Optional[str] = None

        # 1. Run Intent Classification and Entity Extraction
        intent_res = intent_classifier.predict(caller_text)
        self.latest_intent = intent_res
        new_entities = entity_extractor.extract_all(caller_text)

        # Merge extracted entities into session state
        if new_entities.caller_name:
            self.caller_entities.caller_name = new_entities.caller_name
        if new_entities.reason_for_calling:
            self.caller_entities.reason_for_calling = new_entities.reason_for_calling
        if new_entities.phone_number:
            self.caller_entities.phone_number = new_entities.phone_number
        if new_entities.best_time_to_reach:
            self.caller_entities.best_time_to_reach = new_entities.best_time_to_reach
        if new_entities.urgency_level == "URGENT":
            self.caller_entities.urgency_level = "URGENT"

        name = self.caller_entities.caller_name or "there"
        reason = self.caller_entities.reason_for_calling or "your inquiry"

        # 2. State Machine Routing
        if self.state in (CallStateEnum.GREETING, CallStateEnum.AWAITING_NAME_AND_REASON):
            # Caller just provided name and reason
            is_available = recipient_service.is_recipient_available()

            if is_available:
                # Recipient is available -> Transfer Call
                self.state = CallStateEnum.TRANSFERRING_CALL
                response_text = (
                    f"Thank you, {name}. {recipient.name} is available right now. "
                    f"Please hold on while I connect you directly to discuss {reason}."
                )
                notification_preview = recipient_service.dispatch_transfer_notification(
                    caller_name=name, reason=reason
                )
            else:
                # Recipient is BUSY / IN_MEETING -> Ask for Phone Number & Best Time
                self.state = CallStateEnum.AWAITING_CALLBACK_DETAILS
                response_text = (
                    f"Thank you, {name}. {recipient.name} is currently busy and unavailable to take your call right now. "
                    f"Could you please leave your phone number and the best time for him to reach back to you?"
                )

        elif self.state == CallStateEnum.AWAITING_CALLBACK_DETAILS:
            # Caller provided phone number and preferred time window
            phone = self.caller_entities.phone_number or "your provided number"
            best_time = self.caller_entities.best_time_to_reach or "as soon as possible"

            # Create and dispatch callback record
            record = recipient_service.record_callback_request(
                call_id=self.session_id,
                caller_name=self.caller_entities.caller_name or "Unknown Caller",
                phone_number=phone,
                best_time=best_time,
                reason=reason,
                urgency=self.caller_entities.urgency_level,
            )
            self.callback_record = record
            self.state = CallStateEnum.CALL_COMPLETED

            response_text = (
                f"Thank you, {name}. I have noted your number as {phone} and requested a callback for {best_time} "
                f"regarding {reason}. {recipient.name} has been notified and will reach back to you then. Have a wonderful day!"
            )
            notification_preview = (
                f"📱 [DISPATCHED TO {recipient.name} via {recipient.notification_channel}]\n"
                f"Callback Request from {name} ({phone}) for {best_time}.\n"
                f"Reason: {reason}"
            )

        elif self.state == CallStateEnum.TRANSFERRING_CALL:
            response_text = f"Connecting your call to {recipient.name}... Please remain on the line."
            self.state = CallStateEnum.CALL_COMPLETED

        else:
            response_text = (
                f"Your request has already been logged, {name}. {recipient.name} will reach back to you. Goodbye!"
            )
            self.state = CallStateEnum.CALL_COMPLETED

        self.history.append({"speaker": "AI", "text": response_text})

        return CallTurnResponse(
            session_id=self.session_id,
            current_state=self.state,
            bot_speech_response=response_text,
            recipient_status=recipient.status,
            extracted_entities=self.caller_entities,
            intent_analysis=self.latest_intent,
            callback_record=self.callback_record,
            notification_preview=notification_preview,
        )


class SessionManager:
    """In-memory active session registry."""

    def __init__(self) -> None:
        self.sessions: Dict[str, InboundCallSession] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> InboundCallSession:
        if not session_id:
            session = InboundCallSession()
            self.sessions[session.session_id] = session
            return session

        if session_id not in self.sessions:
            session = InboundCallSession(session_id=session_id)
            self.sessions[session_id] = session
            return session

        return self.sessions[session_id]


session_manager = SessionManager()
