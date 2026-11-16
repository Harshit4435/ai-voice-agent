"""Pipecat AI Voice Agent Pipeline.

Integrates Pipecat (pipecat-ai v1.16+) for conversational voice agent processing:
- Frame-based pipeline with TranscriptionFrame, TextFrame, InterruptionFrame
- Inbound voice receptionist conversational state machine
- Contextual entity resolution: Name, Reason, Phone Number, Callback Time
- Zero repetition loops
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any, Dict, Optional

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.ml.entity_extractor import entity_extractor
from app.models.schemas import (
    CallbackRecord,
    CallStateEnum,
    CallTurnResponse,
    ExtractedEntities,
    RecipientStatusEnum,
)
from app.services.recipient_service import recipient_service

logger = logging.getLogger("pipecat_agent")


def format_digits_for_tts(phone_raw: str) -> str:
    """Formats phone numbers for smooth digit-by-digit spoken delivery."""
    digits = re.sub(r"\D", "", phone_raw)
    if len(digits) >= 10:
        parts = []
        for i in range(0, len(digits), 3):
            chunk = digits[i:i + 3]
            parts.append(" ".join(list(chunk)))
        return ", ".join(parts)
    return phone_raw


class PipecatInboundSession:
    """Manages an active Pipecat inbound voice call session."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.state: CallStateEnum = CallStateEnum.GREETING
        self.caller_entities: ExtractedEntities = ExtractedEntities()
        self.history: list[Dict[str, str]] = []
        self.callback_record: Optional[CallbackRecord] = None

    def start(self) -> str:
        self.state = CallStateEnum.AWAITING_NAME_AND_REASON
        greeting = "Hello! Thank you for calling. Please state your name and the reason for your call."
        self.history.append({"speaker": "AI", "text": greeting})
        return greeting

    def handle_speech(self, text: str) -> Dict[str, Any]:
        self.history.append({"speaker": "CALLER", "text": text})

        # 1. Entity Extraction
        extracted = entity_extractor.extract_all(text)
        if extracted.caller_name:
            if not self.caller_entities.caller_name or any(k in text.lower() for k in ("name is", "this is", "i am", "i'm")):
                self.caller_entities.caller_name = extracted.caller_name

        if extracted.reason_for_calling:
            self.caller_entities.reason_for_calling = extracted.reason_for_calling

        if extracted.phone_number:
            self.caller_entities.phone_number = extracted.phone_number

        if extracted.best_time_to_reach:
            self.caller_entities.best_time_to_reach = extracted.best_time_to_reach

        if extracted.urgency_level == "URGENT":
            self.caller_entities.urgency_level = "URGENT"

        has_name = bool(self.caller_entities.caller_name)
        has_reason = bool(self.caller_entities.reason_for_calling)

        # Contextual reason capture: if caller was already asked for reason, accept substantive text
        if has_name and not has_reason and not extracted.caller_name:
            clean = text.strip().rstrip(".!?")
            if clean.lower() not in ("hello", "hi", "hey", "yes", "no"):
                stripped = re.sub(r"^(?:the\s+)?reason\s+(?:for\s+(?:the\s+)?call\s+)?is\s+", "", clean, flags=re.I)
                stripped = re.sub(r"^i\s+(?:just\s+)?(?:want|need)\s+(?:to\s+)?", "", stripped, flags=re.I).strip()
                if len(stripped) > 2:
                    self.caller_entities.reason_for_calling = stripped
                    has_reason = True

        # Contextual phone capture: if waiting for phone number and caller supplies digits
        if not self.caller_entities.phone_number:
            raw_digits = re.sub(r"\D", "", text)
            if 7 <= len(raw_digits) <= 16:
                clean_phone = entity_extractor.extract_phone(text) or text.strip().rstrip(".!?")
                self.caller_entities.phone_number = clean_phone

        # Check for caller hangup request / goodbye
        if re.search(r"\b(bye|goodbye|hang up|hung up|that's all|that is all|end call)\b", text, re.I):
            if has_name and self.caller_entities.phone_number and self.caller_entities.reason_for_calling:
                return self._finalize_callback(name, reason)
            else:
                self.state = CallStateEnum.CALL_COMPLETED
                bot_text = f"Thank you for calling{', ' + name if has_name else ''}. Have a wonderful day! Goodbye."
                self.history.append({"speaker": "AI", "text": bot_text})
                return {
                    "session_id": self.session_id,
                    "current_state": "CALL_COMPLETED",
                    "bot_speech_response": bot_text,
                    "extracted_entities": self.caller_entities.model_dump(),
                    "notification_preview": None,
                    "callback_record": None,
                    "hang_up": True,
                }

        name = self.caller_entities.caller_name or "there"
        reason = self.caller_entities.reason_for_calling or "your inquiry"
        notification_preview: Optional[str] = None

        # 2. Conversational State Machine
        if self.state in (CallStateEnum.GREETING, CallStateEnum.AWAITING_NAME_AND_REASON):
            if has_name and has_reason:
                self.state = CallStateEnum.AWAITING_CALLBACK_DETAILS
                has_phone = bool(self.caller_entities.phone_number)
                has_time = bool(self.caller_entities.best_time_to_reach)

                if has_phone and has_time:
                    return self._finalize_callback(name, reason)
                elif has_phone and not has_time:
                    spoken_phone = format_digits_for_tts(self.caller_entities.phone_number)
                    bot_text = f"Thank you, {name}. I have your phone number as {spoken_phone}. What would be the best time for our team to reach back out to you?"
                elif not has_phone and has_time:
                    bot_text = f"Thank you, {name}. We will reach back {self.caller_entities.best_time_to_reach}. Could you please provide the best phone number for our team to reach out to you on?"
                else:
                    bot_text = f"Thank you, {name}. What is the best phone number and time for our team to reach back out to you?"
            elif has_name and not has_reason:
                if re.match(r"^(hello|hi|hey)$", text.strip(), re.I):
                    bot_text = f"Hello, {name}! I'm listening. Could you please share the reason for your call so our team can assist you?"
                else:
                    bot_text = f"Thank you, {name}. Could you please state the reason for your call?"
            elif has_reason and not has_name:
                bot_text = f"Thank you for reaching out regarding {reason}. May I please have your name before we continue?"
            else:
                bot_text = "Hello! Thank you for calling. Please state your name and the reason for your call."

        elif self.state == CallStateEnum.AWAITING_CALLBACK_DETAILS:
            has_phone = bool(self.caller_entities.phone_number)
            has_time = bool(self.caller_entities.best_time_to_reach)

            if has_phone and has_time:
                return self._finalize_callback(name, reason)
            elif has_phone and not has_time:
                spoken_phone = format_digits_for_tts(self.caller_entities.phone_number)
                bot_text = f"Thank you, {name}. I've got your number as {spoken_phone}. What would be the best time for our team to reach back out to you?"
            elif not has_phone and has_time:
                bot_text = f"Thank you, {name}. We will reach back {self.caller_entities.best_time_to_reach}. Could you please provide the phone number our team should call you on?"
            else:
                bot_text = f"Thank you, {name}. What is the best phone number and time for our team to reach back out to you?"

        else:
            bot_text = f"Your request has already been logged, {name}. Our team will reach back out to you. Have a wonderful day!"

        self.history.append({"speaker": "AI", "text": bot_text})
        return {
            "session_id": self.session_id,
            "current_state": self.state.value if hasattr(self.state, "value") else str(self.state),
            "bot_speech_response": bot_text,
            "extracted_entities": self.caller_entities.model_dump(),
            "notification_preview": notification_preview,
            "callback_record": self.callback_record.model_dump() if self.callback_record else None,
        }

    def _finalize_callback(self, name: str, reason: str) -> Dict[str, Any]:
        self.state = CallStateEnum.CALL_COMPLETED
        phone = self.caller_entities.phone_number or "Unknown"
        best_time = self.caller_entities.best_time_to_reach or "as soon as possible"
        spoken_phone = format_digits_for_tts(phone)

        record = recipient_service.record_callback_request(
            call_id=self.session_id,
            caller_name=name,
            phone_number=phone,
            best_time=best_time,
            reason=reason,
            urgency=self.caller_entities.urgency_level,
        )
        self.callback_record = record

        bot_text = (
            f"Thank you, {name}. I have noted your number as {spoken_phone}, and requested a callback for "
            f"{best_time} regarding {reason}. The team has been notified and will reach back out to you then. Have a wonderful day! Goodbye."
        )
        notif = (
            f"📱 [DISPATCHED TO Team via Slack & SMS]\n"
            f"Callback Request from {name} ({phone}) for {best_time}.\n"
            f"Reason: {reason}"
        )
        self.history.append({"speaker": "AI", "text": bot_text})
        return {
            "session_id": self.session_id,
            "current_state": "CALL_COMPLETED",
            "bot_speech_response": bot_text,
            "extracted_entities": self.caller_entities.model_dump(),
            "notification_preview": notif,
            "callback_record": record.model_dump(),
            "hang_up": True,
        }


class PipecatInboundProcessor(FrameProcessor):
    """Pipecat FrameProcessor that hooks into Pipecat pipelines."""

    def __init__(self, session: Optional[PipecatInboundSession] = None) -> None:
        super().__init__()
        self.session = session or PipecatInboundSession()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text
            result = self.session.handle_speech(text)
            reply = result["bot_speech_response"]
            await self.push_frame(TTSTextFrame(reply, text_type="sentence"))
            await self.push_frame(TextFrame(reply))
            if result.get("hang_up") or result.get("current_state") == "CALL_COMPLETED":
                await self.push_frame(EndFrame())
        elif isinstance(frame, InterruptionFrame):
            # Barge-in event from Pipecat VAD
            logger.info("Pipecat barge-in: caller interrupted AI speech")
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class PipecatSessionManager:
    """Session registry for active Pipecat voice agent sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, PipecatInboundSession] = {}

    @property
    def sessions(self) -> Dict[str, PipecatInboundSession]:
        return self._sessions

    def get_or_create(self, session_id: Optional[str] = None) -> PipecatInboundSession:
        if not session_id or session_id not in self._sessions:
            sess = PipecatInboundSession(session_id=session_id)
            self._sessions[sess.session_id] = sess
            return sess
        return self._sessions[session_id]

    def create_session(self) -> tuple[PipecatInboundSession, str]:
        sess = self.get_or_create()
        greeting = sess.start()
        return sess, greeting

    def process_speech(self, session_id: str, text: str) -> Dict[str, Any]:
        sess = self.get_or_create(session_id)
        return sess.handle_speech(text)

    def reset_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def reset_session(self, session_id: Optional[str] = None) -> PipecatInboundSession:
        sess = PipecatInboundSession(session_id=session_id)
        self._sessions[sess.session_id] = sess
        return sess


pipecat_session_manager = PipecatSessionManager()
