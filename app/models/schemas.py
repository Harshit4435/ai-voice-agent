"""Pydantic Schemas for Inbound Call AI Agent.

Defines schemas for call sessions, intent analysis, entity extraction,
recipient availability, and callback message records.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CallStateEnum(str, Enum):
    IDLE = "IDLE"
    RINGING = "RINGING"
    GREETING = "GREETING"                          # "Please state your name and reason for calling."
    AWAITING_NAME_AND_REASON = "AWAITING_NAME_AND_REASON"
    RECIPIENT_CHECKING = "RECIPIENT_CHECKING"      # Checking if recipient is available vs busy
    TRANSFERRING_CALL = "TRANSFERRING_CALL"        # Recipient is available -> connecting caller
    AWAITING_CALLBACK_DETAILS = "AWAITING_CALLBACK_DETAILS" # Recipient is busy -> asking phone & best time
    CALL_COMPLETED = "CALL_COMPLETED"
    CALL_TERMINATED = "CALL_TERMINATED"


class RecipientStatusEnum(str, Enum):
    AVAILABLE = "AVAILABLE"
    BUSY = "BUSY"
    IN_MEETING = "IN_MEETING"
    DO_NOT_DISTURB = "DO_NOT_DISTURB"


class InboundIntentEnum(str, Enum):
    STATEMENT_NAME_AND_REASON = "STATEMENT_NAME_AND_REASON"
    URGENT_INQUIRY = "URGENT_INQUIRY"
    GENERAL_INQUIRY = "GENERAL_INQUIRY"
    SCHEDULE_CALLBACK = "SCHEDULE_CALLBACK"
    LEAVE_CALLBACK_INFO = "LEAVE_CALLBACK_INFO"
    CALL_STATUS_CHECK = "CALL_STATUS_CHECK"


class ExtractedEntities(BaseModel):
    caller_name: Optional[str] = Field(default=None, description="Extracted full name of caller")
    reason_for_calling: Optional[str] = Field(default=None, description="Extracted purpose/topic of call")
    phone_number: Optional[str] = Field(default=None, description="Extracted phone number for callback")
    best_time_to_reach: Optional[str] = Field(default=None, description="Preferred callback time window")
    urgency_level: str = Field(default="NORMAL", description="NORMAL or URGENT")


class IntentClassificationResult(BaseModel):
    intent: InboundIntentEnum
    confidence: float = Field(ge=0.0, le=1.0)
    all_probabilities: Dict[str, float] = Field(default_factory=dict)
    matched_keywords: List[str] = Field(default_factory=list)


class CallerTurnInput(BaseModel):
    session_id: str
    speech_text: str = Field(..., min_length=1, description="Transcript of what caller spoke")
    is_voice_input: bool = False


class RecipientProfile(BaseModel):
    name: str = "Alex Mercer"
    role: str = "Lead Solutions Architect"
    status: RecipientStatusEnum = RecipientStatusEnum.BUSY
    direct_phone: str = "+1 (555) 019-2834"
    notification_channel: str = "Slack & SMS"


class CallbackRecord(BaseModel):
    call_id: str
    caller_name: str
    phone_number: str
    best_time_to_reach: str
    reason: str
    created_at_iso: str
    recipient_name: str
    urgency: str = "NORMAL"
    notification_dispatched: bool = True


class CallTurnResponse(BaseModel):
    session_id: str
    current_state: CallStateEnum
    bot_speech_response: str
    recipient_status: RecipientStatusEnum
    extracted_entities: ExtractedEntities
    intent_analysis: Optional[IntentClassificationResult] = None
    callback_record: Optional[CallbackRecord] = None
    notification_preview: Optional[str] = None

