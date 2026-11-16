"""Entity Extraction Engine for Inbound Calls.

Extracts caller names, reasons for calling, phone numbers, preferred callback times,
and urgency signals from conversational speech transcripts.
"""
from __future__ import annotations

import re
from typing import Optional

from app.models.schemas import ExtractedEntities


class InboundEntityExtractor:
    """Robust pattern & linguistic entity extractor for telephony screening."""

    def __init__(self) -> None:
        # 1. Name patterns
        self._name_patterns = [
            re.compile(r"(?:my name is|name's|this is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", re.IGNORECASE),
            re.compile(r"(?:this is)\s+(Dr\.|Mr\.|Mrs\.|Ms\.)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.IGNORECASE),
            re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:here|speaking)\b", re.IGNORECASE),
        ]

        # 2. Reason patterns
        self._reason_patterns = [
            re.compile(r"(?:calling regarding|calling about|calling to discuss|calling to)\s+([^.,;]+)", re.IGNORECASE),
            re.compile(r"(?:inquire about|follow up on|discuss|checking on|inquiry about)\s+([^.,;]+)", re.IGNORECASE),
            re.compile(r"(?:reason for calling is|regarding|about)\s+([^.,;]+)", re.IGNORECASE),
        ]

        # 3. Phone patterns (E.164, North American NANP, standard formats)
        self._phone_pattern = re.compile(
            r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?"
        )
        self._simple_phone_pattern = re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        )

        # 4. Best callback time patterns
        self._time_patterns = [
            re.compile(r"(?:best time (?:is|to (?:call|reach|speak)(?: me)?(?: is)?)|call me back|available)\s+([^.,;]+)", re.IGNORECASE),
            re.compile(r"(?:between\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s+and\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?(?:\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday))?)", re.IGNORECASE),
            re.compile(r"(?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|morning|afternoon|evening|after lunch|before noon)", re.IGNORECASE),
            re.compile(r"(?:after\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*(?:today|tomorrow)?)", re.IGNORECASE),
            re.compile(r"(?:before\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*(?:today|tomorrow)?)", re.IGNORECASE),
        ]

        # 5. Urgency tokens
        self._urgent_keywords = {
            "emergency", "urgent", "critical", "outage", "crashed", "sev-1", "severity 1",
            "security breach", "immediately", "code red", "down", "time-sensitive"
        }

    def extract_name(self, text: str) -> Optional[str]:
        """Extracts caller full name cleanly without capturing subsequent clauses."""
        stop_words = {
            "a", "an", "the", "just", "calling", "interested", "trying", "reaching",
            "wondering", "looking", "here", "speaking", "leaving", "regarding", "about"
        }
        # Stop-boundary matching
        boundary_pattern = re.compile(
            r"(?:my name is|name's|this is|i am|i'm)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?=\s+(?:calling|about|regarding|from|with|and|,|\.|$))",
            re.IGNORECASE
        )
        match = boundary_pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            # Exclude titles if solitary or false triggers
            if candidate.lower() not in stop_words:
                words = candidate.split()
                if not any(w.lower() in stop_words for w in words):
                    return candidate.title()

        for pattern in self._name_patterns:
            match = pattern.search(text)
            if match:
                groups = [g for g in match.groups() if g]
                if groups:
                    name_candidate = groups[-1].strip()
                    # Strip any trailing transition keywords
                    name_candidate = re.split(r"\s+(?:and|calling|from|regarding|about|to|with)\b", name_candidate, flags=re.I)[0].strip()
                    if len(name_candidate.split()) >= 1 and len(name_candidate) > 2:
                        if name_candidate.lower() not in stop_words and not any(w.lower() in stop_words for w in name_candidate.split()):
                            return name_candidate.title()
        return None

    def extract_reason(self, text: str) -> Optional[str]:
        """Extracts the reason or purpose of the call."""
        for pattern in self._reason_patterns:
            match = pattern.search(text)
            if match:
                reason = match.group(1).strip()
                # Clean trailing periods or conjunctions
                reason = re.sub(r"[\.!\?]+$", "", reason).strip()
                if len(reason) > 3:
                    return reason
        return None

    def extract_phone(self, text: str) -> Optional[str]:
        """Extracts standard or spoken telephone numbers across international and space-separated formats."""
        word_to_digit = {
            "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
        }
        # 1. Normalize spoken word digits like 'nine zero one five'
        tokens = text.lower().split()
        normalized_words = []
        for t in tokens:
            clean = re.sub(r"[^\w]", "", t)
            if clean in word_to_digit:
                normalized_words.append(word_to_digit[clean])
            else:
                normalized_words.append(t)
        normalized_text = " ".join(normalized_words)

        # 2. Match after explicit keywords like 'my number is', 'phone is', etc.
        kw_pattern = re.compile(
            r"(?:phone(?:\s+number)?|number(?:\s+is)?|call(?:\s+me)?(?:\s+at|\s+on)?|reach(?:\s+me)?(?:\s+at|\s+on)?|contact)\s*(?:is|:)?\s*([+\d\s\-\(\)\.]{7,32})",
            re.IGNORECASE
        )
        match = kw_pattern.search(normalized_text)
        if match:
            candidate = match.group(1).strip().rstrip(".,;:!?")
            digits = re.sub(r"\D", "", candidate)
            if len(digits) >= 10:
                return candidate[:24].rstrip(".,;:!?")

        # 3. Match any sequence with digits and space/hyphen/dot separators
        digit_seqs = re.findall(r"(?:\+?\d[\d\s\-\(\)\.]{8,}\d)", normalized_text)
        for seq in digit_seqs:
            clean_seq = seq.strip().rstrip(".,;:!?")
            digits = re.sub(r"\D", "", clean_seq)
            if 10 <= len(digits) <= 15:
                return clean_seq

        # 4. Fallback to simple phone pattern
        match = self._simple_phone_pattern.search(text)
        if match:
            raw_phone = match.group(0).strip().rstrip(".,;:!?")
            digits = re.sub(r"\D", "", raw_phone)
            if len(digits) >= 10:
                return raw_phone
        return None

    def extract_best_time(self, text: str) -> Optional[str]:
        """Extracts preferred callback time or window."""
        for pattern in self._time_patterns:
            match = pattern.search(text)
            if match:
                raw_time = match.group(0 if not match.groups() or not match.group(1) else 1).strip()
                raw_time = re.sub(r"[\.!\?]+$", "", raw_time).strip()
                return raw_time
        return None

    def detect_urgency(self, text: str) -> str:
        """Determines if call requires urgent immediate escalation."""
        lower = text.lower()
        for kw in self._urgent_keywords:
            if kw in lower:
                return "URGENT"
        return "NORMAL"

    def extract_all(self, text: str) -> ExtractedEntities:
        """Runs the complete extraction pipeline over a transcript."""
        return ExtractedEntities(
            caller_name=self.extract_name(text),
            reason_for_calling=self.extract_reason(text),
            phone_number=self.extract_phone(text),
            best_time_to_reach=self.extract_best_time(text),
            urgency_level=self.detect_urgency(text),
        )


entity_extractor = InboundEntityExtractor()

