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


def format_digits_for_speech(phone_str: str) -> str:
    """Formats phone number digit-by-digit with natural breathing commas for speech synthesis."""
    digits = [c for c in phone_str if c.isdigit()]
    if len(digits) == 10:
        # e.g. "4 1 5, 5 5 5, 0 1 9 9"
        area = " ".join(digits[0:3])
        prefix = " ".join(digits[3:6])
        line = " ".join(digits[6:10])
        return f"{area}, {prefix}, {line}"
    return " ".join(digits)


class InboundCallSession:
    """Manages an individual inbound telephone call lifecycle.
    
    Treats every call/login as a brand-new user session without inheriting
    or assuming prior caller identity, and systematically ensures all required details
    (Name, Reason, Phone Number, and Callback Time) are collected.
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.state = CallStateEnum.GREETING
        self.caller_entities = ExtractedEntities()
        self.history: list[dict[str, str]] = []
        self.latest_intent: Optional[IntentClassificationResult] = None
        self.callback_record: Optional[CallbackRecord] = None

    def reset_session(self) -> None:
        """Completely resets session to new user state, wiping all caller data."""
        self.state = CallStateEnum.GREETING
        self.caller_entities = ExtractedEntities()
        self.history.clear()
        self.latest_intent = None
        self.callback_record = None

    def start_call(self, caller_phone: Optional[str] = None) -> CallTurnResponse:
        """Initial inbound call trigger when telephone rings and connects."""
        # Always treat as a brand-new user: reset all caller entities
        self.reset_session()
        self.state = CallStateEnum.AWAITING_NAME_AND_REASON
        if caller_phone:
            self.caller_entities.phone_number = caller_phone

        greeting = (
            "Hello! Thank you for calling. Please state your name and the reason for your call."
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
        """Processes each conversational turn spoken by the caller, requiring all details."""
        self.history.append({"speaker": "CALLER", "text": caller_text})
        recipient = recipient_service.get_recipient_profile()
        notification_preview: Optional[str] = None

        # 1. Run Intent Classification and Entity Extraction
        intent_res = intent_classifier.predict(caller_text)
        self.latest_intent = intent_res
        new_entities = entity_extractor.extract_all(caller_text)

        # Merge newly extracted entities into current session state
        if new_entities.caller_name:
            # Preserve existing valid name unless caller explicitly provides/corrects their name
            if not self.caller_entities.caller_name or any(k in caller_text.lower() for k in ("name is", "name's", "this is", "i am", "i'm")):
                self.caller_entities.caller_name = new_entities.caller_name
        if new_entities.reason_for_calling:
            self.caller_entities.reason_for_calling = new_entities.reason_for_calling
        if new_entities.phone_number:
            self.caller_entities.phone_number = new_entities.phone_number
        if new_entities.best_time_to_reach:
            self.caller_entities.best_time_to_reach = new_entities.best_time_to_reach
        if new_entities.urgency_level == "URGENT":
            self.caller_entities.urgency_level = "URGENT"

        has_name = bool(self.caller_entities.caller_name)
        has_reason = bool(self.caller_entities.reason_for_calling)
        name = self.caller_entities.caller_name or "there"
        reason = self.caller_entities.reason_for_calling or "your inquiry"

        # 2. State Machine Routing
        if self.state in (CallStateEnum.GREETING, CallStateEnum.AWAITING_NAME_AND_REASON):
            # Check if all initial required details (name and reason) are collected
            if has_name and has_reason:
                is_available = recipient_service.is_recipient_available()
                if is_available:
                    self.state = CallStateEnum.TRANSFERRING_CALL
                    response_text = (
                        f"Thank you, {name}. The Vittoris team is available right now. "
                        f"Please hold on while I connect your call directly to discuss {reason}."
                    )
                    notification_preview = recipient_service.dispatch_transfer_notification(
                        caller_name=name, reason=reason
                    )
                else:
                    self.state = CallStateEnum.AWAITING_CALLBACK_DETAILS
                    if self.caller_entities.phone_number:
                        spoken_phone = format_digits_for_speech(self.caller_entities.phone_number)
                        response_text = (
                            f"Thank you, {name}. The Vittoris team is currently busy and unavailable to take your call right now. "
                            f"I have your phone number from this call as {spoken_phone}. "
                            f"What would be the best time or window for our team to reach back to you?"
                        )
                    else:
                        response_text = (
                            f"Thank you, {name}. The Vittoris team is currently busy and unavailable to take your call right now. "
                            f"Could you please leave your phone number and the best time for our team to reach back to you?"
                        )
            elif has_name and not has_reason:
                # Name provided, but reason is missing: prompt for reason
                self.state = CallStateEnum.AWAITING_NAME_AND_REASON
                response_text = f"Thank you, {name}. Could you please state the reason for your call?"
            elif has_reason and not has_name:
                # Reason provided, but name is missing: prompt for name
                self.state = CallStateEnum.AWAITING_NAME_AND_REASON
                response_text = f"Thank you for reaching out regarding {reason}. May I please have your name before we continue?"
            else:
                # Neither provided: prompt for both details
                self.state = CallStateEnum.AWAITING_NAME_AND_REASON
                response_text = "Hello! Please state your name and the reason for your call so I can assist you."

        elif self.state == CallStateEnum.AWAITING_CALLBACK_DETAILS:
            has_phone = bool(self.caller_entities.phone_number)
            has_time = bool(self.caller_entities.best_time_to_reach)

            if has_phone and has_time:
                phone = self.caller_entities.phone_number
                spoken_phone = format_digits_for_speech(phone) if any(c.isdigit() for c in phone) else phone
                best_time = self.caller_entities.best_time_to_reach

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
                    f"Thank you, {name}. I have noted your number as {spoken_phone}, and requested a callback for {best_time} "
                    f"regarding {reason}. The Vittoris team has been notified and will reach back to you then. Have a wonderful day!"
                )
                notification_preview = (
                    f"📱 [DISPATCHED TO Vittoris Team via {recipient.notification_channel}]\n"
                    f"Callback Request from {name} ({phone}) for {best_time}.\n"
                    f"Reason: {reason}"
                )
            elif has_phone and not has_time:
                # Phone provided, prompt for best callback time
                phone = self.caller_entities.phone_number
                spoken_phone = format_digits_for_speech(phone) if any(c.isdigit() for c in phone) else phone
                self.state = CallStateEnum.AWAITING_CALLBACK_DETAILS
                response_text = (
                    f"Thank you, {name}. I've got your number as {spoken_phone}. "
                    f"What would be the best time or window for our team to reach back to you?"
                )
            elif has_time and not has_phone:
                # Time provided, prompt for callback phone number
                best_time = self.caller_entities.best_time_to_reach
                self.state = CallStateEnum.AWAITING_CALLBACK_DETAILS
                response_text = (
                    f"Thank you, {name}. We will reach back {best_time}. "
                    f"Could you please provide the phone number our team should call you on?"
                )
            else:
                # Neither provided
                self.state = CallStateEnum.AWAITING_CALLBACK_DETAILS
                response_text = (
                    f"Could you please leave your phone number and the best time for our team to reach back to you?"
                )

        elif self.state == CallStateEnum.TRANSFERRING_CALL:
            response_text = "Connecting your call to the Vittoris team... Please remain on the line."
            self.state = CallStateEnum.CALL_COMPLETED

        else:
            response_text = (
                f"Your request has already been logged, {name}. The Vittoris team will reach back to you. Goodbye!"
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
    """In-memory active session registry with strict per-call isolation."""

    def __init__(self) -> None:
        self.sessions: Dict[str, InboundCallSession] = {}

    def create_fresh_session(self, session_id: Optional[str] = None) -> InboundCallSession:
        """Always creates a brand new, isolated session treating the user as completely new."""
        sid = session_id or str(uuid.uuid4())
        session = InboundCallSession(session_id=sid)
        self.sessions[sid] = session
        return session

    def get_or_create(self, session_id: Optional[str] = None) -> InboundCallSession:
        if not session_id:
            return self.create_fresh_session()
        if session_id not in self.sessions:
            return self.create_fresh_session(session_id)
        return self.sessions[session_id]


session_manager = SessionManager()

