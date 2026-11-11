"""Intent Classifier Inference Engine.

Loads the calibrated TF-IDF + LinearSVC pipeline, produces probability-calibrated
predictions for inbound caller utterances, and provides confidence telemetry.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib

from app.models.schemas import InboundIntentEnum, IntentClassificationResult

logger = logging.getLogger("inbound_ai.classifier")


class InboundIntentClassifier:
    """Production inference engine for intent categorization with confidence calibration."""

    def __init__(self, model_path: str | Path = "artifacts/trained_intent_model.joblib") -> None:
        self.model_path = Path(model_path)
        self.pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            logger.warning(
                f"Model file {self.model_path} not found. Triggering automated on-demand training..."
            )
            from app.ml.train_model import train_and_validate_intent_model
            train_and_validate_intent_model(artifacts_dir=self.model_path.parent)

        self.pipeline = joblib.load(self.model_path)
        logger.info(f"Loaded calibrated intent classification pipeline from {self.model_path}")

    def predict(self, text: str) -> IntentClassificationResult:
        """Classifies an incoming caller speech utterance into an intent with confidence scores."""
        clean_text = text.strip()
        if not clean_text:
            return IntentClassificationResult(
                intent=InboundIntentEnum.STATEMENT_NAME_AND_REASON,
                confidence=0.5,
                all_probabilities={},
                matched_keywords=[],
            )

        # Predict intent
        predicted_label: str = self.pipeline.predict([clean_text])[0]

        # Extract calibrated probabilities across all classes
        probabilities: Dict[str, float] = {}
        confidence = 0.85

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([clean_text])[0]
            classes = self.pipeline.classes_
            for cls_name, p in zip(classes, probs):
                probabilities[str(cls_name)] = round(float(p), 4)
            confidence = probabilities.get(predicted_label, 0.85)

        # Keyword alignment
        matched_keywords: List[str] = []
        lower_text = clean_text.lower()
        key_indicators = [
            "name is", "this is", "calling regarding", "calling to", "about",
            "emergency", "urgent", "crashed", "outage", "down",
            "schedule", "book", "meeting", "call back", "appointment",
            "number is", "reach me", "best time", "tomorrow",
            "ticket", "status", "invoice", "application", "pricing", "features"
        ]
        for kw in key_indicators:
            if kw in lower_text:
                matched_keywords.append(kw)

        return IntentClassificationResult(
            intent=InboundIntentEnum(predicted_label),
            confidence=round(float(confidence), 4),
            all_probabilities=probabilities,
            matched_keywords=matched_keywords,
        )


intent_classifier = InboundIntentClassifier()
