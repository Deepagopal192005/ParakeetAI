# Internship Assignment — Submission Notes

**Project:** Local Answer Assistant (Windows desktop prototype)

**Short description:** A desktop prototype that records microphone input,
transcribes speech using a local Whisper model, and generates a progressively
streamed answer using a local Ollama LLM.

**How to run:** Follow the Setup and Run sections in `README.md`.

**Demo flow:**
1. Launch the app with `python app.py`.
2. Click Start recording and ask a question, for example:
   "What is a REST API?"
3. Click Stop and wait for the transcription.
4. Click Suggest Answer.
5. Show the answer appearing progressively in the answer panel.

**Privacy:** No external AI API is used. Whisper runs locally and the Ollama
request is sent to the local Ollama service at `localhost:11434`. Initial model
downloads require internet access.
