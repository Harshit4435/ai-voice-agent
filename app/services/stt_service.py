"""Speech-To-Text (STT) Processing Service.

Normalizes speech transcriptions, handles audio byte payloads, and cleans filler noise.
"""
from __future__ import annotations

import re


class SpeechToTextService:
    """Processes speech input from telephone WebSockets or browser microphone."""

    def normalize_transcript(self, raw_text: str) -> str:
        """Cleans transcription artifacts, normalizes punctuation and whitespace."""
        if not raw_text:
            return ""

        cleaned = re.sub(r"\s+", " ", raw_text).strip()
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned


stt_service = SpeechToTextService()
