"""FastAPI Application for Inbound Call AI Agent.

Serves the interactive frontend, provides REST endpoints for call session turns,
recipient status toggles, callback message inbox, and model accuracy metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ml.intent_classifier import intent_classifier
from app.models.schemas import (
    CallbackRecord,
    CallerTurnInput,
    CallTurnResponse,
    RecipientProfile,
    RecipientStatusEnum,
)
from app.services.recipient_service import recipient_service
from app.state_machine.call_session import session_manager

app = FastAPI(
    title="Inbound Call AI Agent API",
    version="1.0.0",
    description="Automated inbound call screener: intent classification, recipient availability routing, and callback logging.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# --- Frontend Static Mount ---
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Inbound Call AI Agent API is running. Frontend located at /frontend/index.html"}


# --- Call Session Endpoints ---
@app.post("/api/call/start", response_model=CallTurnResponse, tags=["call_flow"])
async def start_inbound_call(session_id: str | None = None) -> CallTurnResponse:
    """Initiates an inbound call session and returns the initial screening prompt."""
    session = session_manager.get_or_create(session_id)
    return session.start_call()


@app.post("/api/call/turn", response_model=CallTurnResponse, tags=["call_flow"])
async def process_call_turn(turn_input: CallerTurnInput) -> CallTurnResponse:
    """Processes caller utterance, extracts entities, and routes according to recipient status."""
    session = session_manager.get_or_create(turn_input.session_id)
    return session.process_turn(turn_input.speech_text)


# --- Recipient Management Endpoints ---
@app.get("/api/recipient/status", response_model=RecipientProfile, tags=["recipient"])
async def get_recipient_status() -> RecipientProfile:
    """Retrieves current recipient availability profile."""
    return recipient_service.get_recipient_profile()


class SetStatusRequest(BaseModel):
    status: RecipientStatusEnum


@app.post("/api/recipient/status", response_model=RecipientProfile, tags=["recipient"])
async def update_recipient_status(payload: SetStatusRequest) -> RecipientProfile:
    """Toggles recipient status between AVAILABLE, BUSY, IN_MEETING, DO_NOT_DISTURB."""
    return recipient_service.set_recipient_status(payload.status)


@app.get("/api/recipient/callbacks", response_model=List[CallbackRecord], tags=["recipient"])
async def list_callback_requests() -> List[CallbackRecord]:
    """Retrieves all logged callback requests and voicemails."""
    return recipient_service.get_all_callbacks()


# --- Model Accuracy & Metrics Endpoints ---
@app.get("/api/model/metrics", tags=["ml_model"])
async def get_model_accuracy_metrics() -> Dict[str, Any]:
    """Returns the serialized training validation report, accuracy table, and confusion matrix."""
    report_path = ARTIFACTS_DIR / "accuracy_report.json"
    if not report_path.exists():
        from app.ml.train_model import train_and_validate_intent_model
        return train_and_validate_intent_model(artifacts_dir=ARTIFACTS_DIR)

    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


class PredictIntentRequest(BaseModel):
    text: str


@app.post("/api/model/predict", tags=["ml_model"])
async def predict_intent(payload: PredictIntentRequest) -> Dict[str, Any]:
    """Ad-hoc endpoint to test intent classification on arbitrary text."""
    res = intent_classifier.predict(payload.text)
    return res.model_dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

