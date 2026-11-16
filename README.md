# Pipecat AI Voice Agent 📞🤖

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-10b981?style=for-the-badge&logo=github)](https://harshit4435.github.io/ai-voice-agent/)
[![Pipecat AI](https://img.shields.io/badge/Framework-Pipecat%20AI%20v1.16-9b72cb?style=for-the-badge)](https://github.com/pipecat-ai/pipecat)
[![Tests](https://img.shields.io/badge/Tests-25%20Passing-blue?style=for-the-badge&logo=pytest)](https://github.com/Harshit4435/ai-voice-agent)

An intelligent, hands-free inbound voice receptionist powered by **Pipecat AI Framework**, featuring voice pipeline orchestration, auto-start microphone, real-time barge-in interruption, contextual intent extraction, and automated lead dispatch.

🌐 **Live Server / Demo**: **[https://harshit4435.github.io/ai-voice-agent/](https://harshit4435.github.io/ai-voice-agent/)**

---

## 🌟 Key Capabilities

1. **Pipecat AI Pipeline Framework**:
   - Built on `pipecat-ai` v1.16 with frame-based voice orchestration (`TranscriptionFrame`, `TTSTextFrame`, `InterruptionFrame`).
   - Contextual conversation engine that smoothly handles caller intake without repetitive robotic loops.
2. **Auto-Start Microphone**:
   - The microphone automatically activates when the AI finishes speaking (`utterance.onend`), enabling smooth conversational turns without touching any buttons.
3. **Barge-in / Instant Voice Interruption**:
   - If the caller begins speaking while the AI is talking, speech playback is cancelled instantly (`synth.cancel()`), allowing natural conversational interruptions.
4. **Google Voice Line & Caller ID Integration**:
   - **When Caller ID is detected**: The AI recognizes the caller's number, acknowledges it, and avoids asking for the phone number again.
   - **When Caller ID is unknown/private**: Prompts for caller number with support for international, space-separated, and spoken-word formats (`"555 019 2834"`).
5. **Fresh Caller Sessions**:
   - Each inbound call is strictly isolated and starts with a fresh session, ensuring no state leakage between calls.
6. **Dynamic Recipient Status Routing**:
   - **When Recipient is BUSY / IN A MEETING**: Collects caller name, reason, phone number, and preferred callback window, then dispatches an automated Slack/SMS alert to the team.
   - **When Recipient is AVAILABLE**: Informs caller and initiates a live call transfer to the recipient's direct line.
7. **Calibrated Machine Learning Model**:
   - Sublinear TF-IDF + Calibrated Linear Support Vector Classifier (`LinearSVC`).
   - Stratified 5-Fold Cross-Validation (**88.57% mean CV accuracy**) and holdout validation (**88.46% accuracy, 88.54% Macro F1**).
8. **Automated Test Suite**:
   - 24 comprehensive unit & integration tests passing with 100% success rate.

---

## 🏗️ Architecture & Directory Structure

```text
inbound-call-ai-agent/
├── app/
│   ├── main.py                     # FastAPI server with REST & Static mount
│   ├── config.py                   # App configuration
│   ├── models/
│   │   └── schemas.py              # Pydantic schemas (Session, Turns, Intents)
│   ├── ml/
│   │   ├── dataset.py              # 130 curated balanced conversational utterances
│   │   ├── train_model.py          # Model training with Stratified K-Fold validation
│   │   ├── intent_classifier.py    # Probability-calibrated inference engine
│   │   └── entity_extractor.py     # Regex & linguistic parser (Name, Reason, Phone, Time)
│   ├── state_machine/
│   │   └── call_session.py         # Conversational state machine & turn routing
│   └── services/
│       ├── recipient_service.py    # Recipient status manager & notification dispatcher
│       └── stt_service.py          # Speech-to-text transcript normalizer
├── frontend/
│   ├── index.html                  # Phone simulator & accuracy dashboard
│   ├── style.css                   # Glassmorphism cyber-aesthetic styles
│   └── app.js                      # Speech recognition, audio TTS, API client
├── tests/
│   ├── test_intent_classifier.py   # Intent prediction & probability tests
│   ├── test_entity_extractor.py    # Entity parsing tests
│   ├── test_state_machine.py       # Full dialogue flow tests (Busy vs Available)
│   ├── test_recipient_service.py   # Notification & status tests
│   └── test_api_endpoints.py       # FastAPI integration tests
├── artifacts/
│   ├── accuracy_report.json        # Detailed model accuracy table & confusion matrix
│   └── trained_intent_model.joblib # Serialized model pipeline
├── demo_call_simulation.py         # Standalone CLI demonstration script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Run on Localhost)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Localhost Web App
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser to: **`http://localhost:8000`**

### 3. Run Standalone CLI Demo
```bash
python3 demo_call_simulation.py
```

### 4. Run Mandatory Tests
```bash
PYTHONPATH="." pytest -v tests/
```

---

## 📊 Model Evaluation & Accuracy Table

### Accuracy Summary
- **Model Architecture**: TF-IDF N-grams `(1, 2)` + Calibrated `LinearSVC(C=1.2)`
- **Cross-Validation**: Stratified 5-Fold Cross Validation (**88.57% ± 6.46%**)
- **Holdout Validation Accuracy**: **88.46%**
- **Macro F1-Score**: **88.54%**
- **Inference Latency**: **< 2.5 ms**

### Per-Class Accuracy Metrics

| Intent Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **STATEMENT_NAME_AND_REASON** | 80.0% | 80.0% | 80.0% | 5 |
| **URGENT_INQUIRY** | 100.0% | 100.0% | 100.0% | 4 |
| **GENERAL_INQUIRY** | 80.0% | 100.0% | 88.9% | 4 |
| **SCHEDULE_CALLBACK** | 100.0% | 75.0% | 85.7% | 4 |
| **LEAVE_CALLBACK_INFO** | 100.0% | 100.0% | 100.0% | 5 |
| **CALL_STATUS_CHECK** | 75.0% | 75.0% | 75.0% | 4 |
| **Overall Macro Average** | **89.2%** | **88.3%** | **88.5%** | **26 (Holdout)** |

