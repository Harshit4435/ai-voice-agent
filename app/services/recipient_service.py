"""Recipient Management and Notification Dispatcher.

Maintains recipient availability status (AVAILABLE, BUSY, IN_MEETING) and dispatches
simulated Slack/SMS notifications for incoming transfers or callback requests.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from app.models.schemas import CallbackRecord, RecipientProfile, RecipientStatusEnum

logger = logging.getLogger("inbound_ai.recipient")


class RecipientService:
    """Manages recipient status, presence, and dispatching callback notifications."""

    def __init__(self) -> None:
        self.recipient = RecipientProfile(
            name="Alex Mercer",
            role="Lead Solutions Architect",
            status=RecipientStatusEnum.BUSY,  # Default to BUSY as requested
            direct_phone="+1 (555) 019-2834",
            notification_channel="Slack #inbound-alerts & SMS",
        )
        self.callback_inbox: List[CallbackRecord] = []

    def get_recipient_profile(self) -> RecipientProfile:
        return self.recipient

    def set_recipient_status(self, new_status: RecipientStatusEnum) -> RecipientProfile:
        self.recipient.status = new_status
        logger.info(f"Recipient {self.recipient.name} status updated to {new_status.value}")
        return self.recipient

    def is_recipient_available(self) -> bool:
        return self.recipient.status == RecipientStatusEnum.AVAILABLE

    def dispatch_transfer_notification(self, caller_name: str, reason: str) -> str:
        """Dispatches notification when caller is being transferred live."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        notification = (
            f"🟢 [LIVE CALL TRANSFER - {timestamp}]\n"
            f"Recipient: {self.recipient.name} ({self.recipient.role})\n"
            f"Caller: {caller_name}\n"
            f"Reason: {reason}\n"
            f"Action: Routing audio call to direct line {self.recipient.direct_phone}"
        )
        logger.info(notification)
        return notification

    def record_callback_request(
        self,
        call_id: str,
        caller_name: str,
        phone_number: str,
        best_time: str,
        reason: str,
        urgency: str = "NORMAL",
    ) -> CallbackRecord:
        """Stores callback record and dispatches recipient alert."""
        record = CallbackRecord(
            call_id=call_id,
            caller_name=caller_name,
            phone_number=phone_number,
            best_time_to_reach=best_time,
            reason=reason,
            created_at_iso=datetime.now(timezone.utc).isoformat(),
            recipient_name=self.recipient.name,
            urgency=urgency,
            notification_dispatched=True,
        )
        self.callback_inbox.append(record)

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        alert_badge = "🔴 [URGENT CALLBACK]" if urgency == "URGENT" else "🟡 [VOICEMAIL / CALLBACK REQUEST]"
        notification = (
            f"{alert_badge} - {timestamp}\n"
            f"To: {self.recipient.name} (Status: {self.recipient.status.value})\n"
            f"From: {caller_name}\n"
            f"Callback Phone: {phone_number}\n"
            f"Best Time to Reach: {best_time}\n"
            f"Reason: {reason}\n"
            f"Channel: {self.recipient.notification_channel}"
        )
        logger.info(notification)
        return record

    def get_all_callbacks(self) -> List[CallbackRecord]:
        return list(reversed(self.callback_inbox))


recipient_service = RecipientService()
