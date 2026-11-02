# Local Answer Assistant — Desktop Prototype

A small Windows desktop app inspired by Parakeet AI. It records microphone audio,
transcribes it with **faster-whisper running locally**, and streams a suggested
answer from an **Ollama model running locally**.

## Featuresa

- Start / Stop microphone recording
- Local speech-to-text (Whisper `small`, CPU + int8)
- Editable transcription text
- Suggest Answer button
- Progressive token-by-token answer display
- Select among locally installed Ollama models
- No cloud AI API keys; audio and prompts are sent only to local processes

## Requirements

- Windows 10/11 (64-bit)
- Python 3.10–3.12 recommended
- Microphone
- Ollama for Windows: https://ollama.com/download

## Setup (one-time)

1. Install Python and tick **Add Python to PATH**.
2. Install Ollama from https://ollama.com/download.
3. Open PowerShell and download a model:

   ```powershell
   ollama pull qwen2.5:3b
   ```

   The app also offers `llama3.2:3b` and `gemma2:2b`; pull one of those
   if you want to select it in the app.
4. In the project folder, open PowerShell and run:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, run:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   and activate again.

## Run

Make sure Ollama is running (the Windows app normally starts its local service),
then run:

```powershell
python app.py
```

Or double-click `run.bat`.

On first transcription, faster-whisper downloads the Whisper `small` model.
This is a one-time download; subsequent transcription uses the local model.
The first answer model download is also one-time.

## How to use

1. Click **Start recording** and speak a question.
2. Click **Stop**. The app transcribes the captured audio.
3. Review or edit the text in the Transcription box.
4. Choose an installed Ollama model.
5. Click **Suggest Answer**. The answer appears progressively as Ollama streams tokens.

## Architecture

- **UI:** PySide6 desktop window
- **Audio capture:** sounddevice, 16 kHz mono PCM
- **Speech-to-text:** faster-whisper / Whisper, CPU int8
- **LLM:** Ollama local HTTP API (`localhost:11434`)
- **Streaming:** Ollama newline-delimited JSON stream; each response token is
  appended to the answer panel as it arrives
- **Responsiveness:** transcription and generation run in QThreads so the UI
  remains responsive

## Troubleshooting

- **No microphone / PortAudio error:** check Windows microphone privacy settings
  and allow desktop apps to access the microphone. Close apps that may exclusively
  occupy the microphone.
- **Ollama connection error:** open Ollama, then run `ollama list` in PowerShell.
  Confirm `qwen2.5:3b` is installed with `ollama pull qwen2.5:3b`.
- **Slow first transcription:** model files are downloaded on first use. CPU
  transcription speed depends on your computer.
- **Model not found:** the model selected in the app must already be installed
  in Ollama. Pull it from PowerShell before selecting it.

## Assignment checklist

- [x] Windows desktop application
- [x] Start and Stop buttons
- [x] Microphone audio capture
- [x] Speech transcription displayed in the app
- [x] Suggest Answer action
- [x] AI answer displayed progressively while generated
- [x] Speech-to-text and LLM run locally
