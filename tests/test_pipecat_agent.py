"""Automated Unit and Integration Tests for Pipecat Voice AI Agent.

Tests Pipecat session state transitions, frames, contextual slot filling,
and zero-repetition loop guarantees.
"""
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.models.schemas import CallStateEnum
from app.pipecat_agent import (
    PipecatInboundProcessor,
    PipecatInboundSession,
    PipecatSessionManager,
    format_digits_for_tts,
)
from pipecat.frames.frames import TextFrame, TranscriptionFrame, TTSTextFrame


@pytest.fixture
def test_client():
    return TestClient(app)


def test_pipecat_session_initialization():
    session = PipecatInboundSession()
    greeting = session.start()
    assert session.state == CallStateEnum.AWAITING_NAME_AND_REASON
    assert "Please state your name and the reason for your call" in greeting
    assert session.caller_entities.caller_name is None
    assert session.caller_entities.reason_for_calling is None


def test_pipecat_single_turn_name_and_reason():
    """Tests intake when caller provides name and reason together."""
    session = PipecatInboundSession()
    session.start()

    res = session.handle_speech("hi this is Harshad calling for a contract")
    assert res["current_state"] == CallStateEnum.AWAITING_CALLBACK_DETAILS.value
    assert res["extracted_entities"]["caller_name"] == "Harshad"
    assert "contract" in res["extracted_entities"]["reason_for_calling"].lower()
    assert "phone number" in res["bot_speech_response"].lower()
    assert "Could you please state the reason" not in res["bot_speech_response"]


def test_pipecat_repetition_recovery_contextual_reason():
    """Tests that caller stating 'the reason for the call is...' is immediately accepted without loop."""
    session = PipecatInboundSession()
    session.start()

    # Turn 1: Only name
    res1 = session.handle_speech("hi my name is Harshad")
    assert res1["extracted_entities"]["caller_name"] == "Harshad"
    assert res1["extracted_entities"]["reason_for_calling"] is None

    # Turn 2: Stating reason
    res2 = session.handle_speech("the reason for the call is I want to have a contract with")
    assert res2["current_state"] == CallStateEnum.AWAITING_CALLBACK_DETAILS.value
    assert res2["extracted_entities"]["reason_for_calling"] is not None
    assert "contract" in res2["extracted_entities"]["reason_for_calling"].lower()
    # Ensure it did NOT ask for reason again!
    assert "Could you please state the reason" not in res2["bot_speech_response"]


def test_pipecat_phone_number_spaced_digits_and_completion():
    """Tests phone number extraction with spaced digits and successful completion."""
    session = PipecatInboundSession()
    session.start()

    session.handle_speech("hi this is Harshad calling for a contract")
    res2 = session.handle_speech("the best time would be tomorrow at 2 and my number is 555 019 2834")

    assert res2["current_state"] == CallStateEnum.CALL_COMPLETED.value
    assert res2["extracted_entities"]["phone_number"] == "555 019 2834"
    assert res2["extracted_entities"]["best_time_to_reach"] == "tomorrow at 2"
    assert res2["callback_record"] is not None
    assert "Have a wonderful day" in res2["bot_speech_response"]


def test_pipecat_format_digits_for_tts():
    formatted = format_digits_for_tts("5550192834")
    assert "5 5 5" in formatted
    assert "0 1 9" in formatted


def test_pipecat_session_manager():
    manager = PipecatSessionManager()
    session, greeting = manager.create_session()
    assert session.session_id in manager.sessions
    assert "Hello!" in greeting

    res = manager.process_speech(session.session_id, "this is Alice calling about pricing")
    assert res["extracted_entities"]["caller_name"] == "Alice"
    assert "pricing" in res["extracted_entities"]["reason_for_calling"].lower()


def test_pipecat_frame_processor():
    import asyncio
    session = PipecatInboundSession()
    session.start()
    processor = PipecatInboundProcessor(session)

    received_frames = []

    async def mock_push_frame(frame, direction=None):
        received_frames.append(frame)

    processor.push_frame = mock_push_frame

    frame = TranscriptionFrame(
        text="hi this is Harshad calling for a contract",
        user_id="test_caller",
        timestamp=1000.0,
    )
    asyncio.run(processor.process_frame(frame, None))

    tts_frames = [f for f in received_frames if type(f) is TTSTextFrame]
    text_frames = [f for f in received_frames if type(f) is TextFrame]

    assert len(tts_frames) == 1
    assert len(text_frames) == 1
    assert "Harshad" in tts_frames[0].text


def test_pipecat_api_endpoints(test_client):
    status_res = test_client.get("/api/pipecat/status")
    assert status_res.status_code == 200
    data = status_res.json()
    assert data["engine"] == "Pipecat Voice AI Framework"

    start_res = test_client.post("/api/pipecat/start", json={})
    assert start_res.status_code == 200
    start_data = start_res.json()
    session_id = start_data["session_id"]
    assert "Hello!" in start_data["bot_speech_response"]

    turn_res = test_client.post("/api/pipecat/turn", json={
        "session_id": session_id,
        "speech_text": "hi this is Harshad calling for a contract"
    })
    assert turn_res.status_code == 200
    turn_data = turn_res.json()
    assert turn_data["extracted_entities"]["caller_name"] == "Harshad"
    assert "contract" in turn_data["extracted_entities"]["reason_for_calling"]
