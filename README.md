# Inbound Call AI Agent 📞🤖

An intelligent automated voice screening, intent classification, and recipient availability router.

---

## 🌟 Key Capabilities

1. **Inbound Call Screening**: Greets callers with: *"Thank you for calling. Please state your name and the reason for your call."*
2. **Intent & Entity Extraction**:
   - Accurately classifies caller intent into 6 distinct operational categories.
   - Extracts caller name, purpose/reason, callback phone number, and preferred reach-back time window.
3. **Dynamic Recipient Status Routing**:
   - **When Recipient is BUSY / IN A MEETING**: Notifies caller that the recipient is unavailable and captures their phone number & best callback window, logging the message and dispatching a Slack/SMS alert to the recipient.
   - **When Recipient is AVAILABLE**: Informs caller and initiates a live call transfer to the recipient's direct line.
4. **Calibrated Machine Learning Model**:
   - Sublinear TF-IDF + Calibrated Linear Support Vector Classifier (`LinearSVC`).
   - Trained with Stratified 5-Fold Cross-Validation (**88.57% mean CV accuracy**) and holdout validation (**88.46% accuracy, 88.54% Macro F1**).
5. **Interactive Localhost Web Dashboard**:
   - Modern glassmorphism UI running at `http://localhost:8000`.
   - Real-time speech input (Web Speech API microphone), browser TTS voice playback, live state stepper, and model telemetry.
6. **Mandatory Test Suite**:
   - 22 comprehensive unit & integration tests passing with 100% success rate.

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
