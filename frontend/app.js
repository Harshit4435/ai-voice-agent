/**
 * Frontend Telephony Simulator & Model Metrics Client.
 * Handles microphone recording, Web Speech TTS, state machine synchronization, and metrics display.
 */

let currentSessionId = null;
let isCallActive = false;
let callSeconds = 0;
let timerInterval = null;
let recognition = null;
let isRecording = false;

// Audio Context & TTS
const synth = window.speechSynthesis;

// On Page Load
document.addEventListener('DOMContentLoaded', () => {
  initSpeechRecognition();
  fetchRecipientStatus();
  fetchModelMetrics();
});

function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      isRecording = true;
      document.getElementById('btn-mic').classList.add('recording');
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      document.getElementById('caller-input-text').value = transcript;
      sendCallerTurn();
    };

    recognition.onerror = (e) => {
      console.warn('Speech recognition error:', e.error);
      stopMicRecording();
    };

    recognition.onend = () => {
      stopMicRecording();
    };
  } else {
    document.getElementById('btn-mic').title = 'Speech recognition not supported in this browser';
  }
}

function toggleMicRecording() {
  if (!recognition) {
    alert('Web Speech API is not supported in this browser. Please type your message.');
    return;
  }
  if (!isRecording) {
    recognition.start();
  } else {
    recognition.stop();
  }
}

function stopMicRecording() {
  isRecording = false;
  document.getElementById('btn-mic').classList.remove('recording');
}

// Handle Call Button Toggle
async function handleCallButton() {
  const btn = document.getElementById('btn-call-action');
  if (!isCallActive) {
    await startInboundCall();
  } else {
    endCallSession();
  }
}

// Start Call
async function startInboundCall() {
  currentSessionId = 'call_' + Math.random().toString(36).substring(2, 9);
  isCallActive = true;

  // UI Updates
  const btn = document.getElementById('btn-call-action');
  btn.textContent = '🛑 End Call (Hang Up)';
  btn.classList.add('hangup');

  document.getElementById('phone-avatar').classList.add('ringing');
  document.getElementById('call-status-heading').textContent = 'Call Connected';
  document.getElementById('call-subtitle').textContent = 'AI Receptionist on line';

  startTimer();
  clearDialogue();
  updateStepper(1);

  try {
    const res = await fetch(`/api/call/start?session_id=${currentSessionId}`, { method: 'POST' });
    const data = await res.json();
    handleBotTurnResponse(data);
  } catch (err) {
    console.error('Failed to start call:', err);
    // Offline / direct fallback
    const fallbackGreeting = "Thank you for calling. Please state your name and the reason for your call.";
    handleBotTurnResponse({
      current_state: "AWAITING_NAME_AND_REASON",
      bot_speech_response: fallbackGreeting,
      recipient_status: document.getElementById('recipient-status-select').value,
      extracted_entities: {}
    });
  }
}

// End Call
function endCallSession() {
  isCallActive = false;
  clearInterval(timerInterval);

  const btn = document.getElementById('btn-call-action');
  btn.textContent = '📞 Simulate Inbound Call';
  btn.classList.remove('hangup');

  document.getElementById('phone-avatar').classList.remove('ringing', 'speaking');
  document.getElementById('call-status-heading').textContent = 'Call Disconnected';
  document.getElementById('call-subtitle').textContent = 'Click to simulate another inbound call';
  document.getElementById('badge-call-state').textContent = 'TERMINATED';

  appendDialogue('bot', 'Call ended. Have a great day!');
  if (synth) synth.cancel();
}

function resetCallSession() {
  endCallSession();
  callSeconds = 0;
  document.getElementById('call-timer').textContent = '00:00';
  document.getElementById('dialogue-container').innerHTML = `
    <div class="bubble bot">
      <div class="bubble-meta">AI RECEPTIONIST</div>
      <em>Line connected. Waiting for call initiation...</em>
    </div>
  `;
  resetEntitiesDisplay();
  updateStepper(0);
}

// Timer
function startTimer() {
  callSeconds = 0;
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    callSeconds++;
    const mins = String(Math.floor(callSeconds / 60)).padStart(2, '0');
    const secs = String(callSeconds % 60).padStart(2, '0');
    document.getElementById('call-timer').textContent = `${mins}:${secs}`;
  }, 1000);
}

// Send Turn
async function sendCallerTurn() {
  const input = document.getElementById('caller-input-text');
  const text = input.value.trim();
  if (!text) return;

  if (!isCallActive) {
    await startInboundCall();
  }

  appendDialogue('caller', text);
  input.value = '';

  try {
    const res = await fetch('/api/call/turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSessionId,
        speech_text: text,
      }),
    });
    const data = await res.json();
    handleBotTurnResponse(data);
  } catch (err) {
    console.error('Error processing turn:', err);
  }
}

function handleInputKeydown(e) {
  if (e.key === 'Enter') {
    sendCallerTurn();
  }
}

function fillPrompt(text) {
  document.getElementById('caller-input-text').value = text;
  sendCallerTurn();
}

// Handle Bot Response
function handleBotTurnResponse(data) {
  const avatar = document.getElementById('phone-avatar');
  avatar.classList.remove('ringing');
  avatar.classList.add('speaking');

  // Update State Badge
  document.getElementById('badge-call-state').textContent = data.current_state;

  // Append dialogue
  appendDialogue('bot', data.bot_speech_response);

  // Play TTS
  speakText(data.bot_speech_response, () => {
    avatar.classList.remove('speaking');
  });

  // Update Entities Display
  if (data.extracted_entities) {
    const e = data.extracted_entities;
    if (e.caller_name) document.getElementById('val-caller-name').textContent = e.caller_name;
    if (e.reason_for_calling) document.getElementById('val-reason').textContent = e.reason_for_calling;
    if (e.phone_number) document.getElementById('val-phone').textContent = e.phone_number;
    if (e.best_time_to_reach) document.getElementById('val-time').textContent = e.best_time_to_reach;
  }

  // Update Intent & Confidence
  if (data.intent_analysis) {
    document.getElementById('val-intent').textContent = data.intent_analysis.intent;
    document.getElementById('val-confidence').textContent = `${(data.intent_analysis.confidence * 100).toFixed(1)}%`;
  }

  // Update State Stepper
  if (data.current_state === 'GREETING' || data.current_state === 'AWAITING_NAME_AND_REASON') {
    updateStepper(2);
  } else if (data.current_state === 'AWAITING_CALLBACK_DETAILS' || data.current_state === 'TRANSFERRING_CALL') {
    updateStepper(3);
  } else if (data.current_state === 'CALL_COMPLETED') {
    updateStepper(4);
  }

  // Notification Feed
  if (data.notification_preview) {
    appendNotification(data.notification_preview);
  }
}

function speakText(text, onEnd) {
  const ttsEnabled = document.getElementById('tts-toggle').checked;
  if (!ttsEnabled || !synth) {
    setTimeout(onEnd, 1500);
    return;
  }

  synth.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.onend = onEnd;
  utterance.onerror = onEnd;
  synth.speak(utterance);
}

function appendDialogue(sender, text) {
  const container = document.getElementById('dialogue-container');
  const bubble = document.createElement('div');
  bubble.className = `bubble ${sender}`;
  const meta = sender === 'bot' ? 'AI RECEPTIONIST' : 'CALLER';
  bubble.innerHTML = `<div class="bubble-meta">${meta}</div>${escapeHtml(text)}`;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
}

function clearDialogue() {
  document.getElementById('dialogue-container').innerHTML = '';
}

function appendNotification(notifText) {
  const stream = document.getElementById('notification-stream');
  const placeholder = stream.querySelector('.notif-placeholder');
  if (placeholder) placeholder.remove();

  const box = document.createElement('div');
  box.className = 'notif-card-box';
  box.textContent = notifText;
  stream.prepend(box);
}

// Stepper
function updateStepper(step) {
  for (let i = 1; i <= 4; i++) {
    const node = document.getElementById(`step-${i}`);
    const line = document.getElementById(`line-${i}`);
    if (i < step) {
      node.className = 'step-node done';
      if (line) line.className = 'step-line active';
    } else if (i === step) {
      node.className = 'step-node active';
      if (line) line.className = 'step-line';
    } else {
      node.className = 'step-node';
      if (line) line.className = 'step-line';
    }
  }
}

function resetEntitiesDisplay() {
  document.getElementById('val-caller-name').textContent = '—';
  document.getElementById('val-reason').textContent = '—';
  document.getElementById('val-phone').textContent = '—';
  document.getElementById('val-time').textContent = '—';
  document.getElementById('val-intent').textContent = '—';
  document.getElementById('val-confidence').textContent = '—';
}

// Recipient Status
async function fetchRecipientStatus() {
  try {
    const res = await fetch('/api/recipient/status');
    const data = await res.json();
    document.getElementById('recipient-status-select').value = data.status;
    document.getElementById('badge-recipient-status').textContent = `RECIPIENT ${data.status}`;
  } catch (err) {
    console.warn('Could not fetch recipient status:', err);
  }
}

async function changeRecipientStatus(newStatus) {
  try {
    const res = await fetch('/api/recipient/status', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: newStatus }),
    });
    const data = await res.json();
    document.getElementById('badge-recipient-status').textContent = `RECIPIENT ${data.status}`;
  } catch (err) {
    console.error('Failed to change status:', err);
  }
}

// Accuracy & Model Metrics
async function fetchModelMetrics() {
  try {
    const res = await fetch('/api/model/metrics');
    const data = await res.json();
    renderAccuracyTable(data);
  } catch (err) {
    console.warn('Could not load accuracy metrics:', err);
  }
}

function renderAccuracyTable(report) {
  const tbody = document.getElementById('accuracy-table-body');
  tbody.innerHTML = '';

  const metaBar = document.getElementById('model-meta-bar');
  if (report.cv_mean_accuracy) {
    metaBar.innerHTML = `
      <strong>Model:</strong> ${report.model_name} | 
      <strong>5-Fold CV Mean:</strong> <span style="color: var(--green); font-weight:700;">${(report.cv_mean_accuracy * 100).toFixed(2)}%</span> | 
      <strong>Holdout Validation:</strong> <span style="color: var(--cyan); font-weight:700;">${(report.holdout_validation_metrics.accuracy * 100).toFixed(2)}%</span>
    `;
  }

  const perClass = report.per_class_metrics || {};
  for (const [cls, m] of Object.entries(perClass)) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color: #fff; font-weight: 600;">${cls}</td>
      <td style="color: var(--cyan);">${(m.precision * 100).toFixed(1)}%</td>
      <td style="color: var(--purple);">${(m.recall * 100).toFixed(1)}%</td>
      <td style="color: var(--green); font-weight: 700;">${(m.f1_score * 100).toFixed(1)}%</td>
      <td style="color: var(--text-muted);">${m.support}</td>
    `;
    tbody.appendChild(tr);
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

