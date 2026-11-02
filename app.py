import os
import sys
import tempfile
import wave
import threading
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
from faster_whisper import WhisperModel
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QMessageBox, QFrame
)

APP_TITLE = "Local Answer Assistant"
SAMPLE_RATE = 16000
OLLAMA_URL = "http://localhost:11434/api/generate"


class TranscriptionWorker(QThread):
    finished_text = Signal(str)
    failed = Signal(str)
    status = Signal(str)

    def __init__(self, audio_path):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            self.status.emit("Loading local speech model…")
            # Model files are downloaded once on first run and then reused locally.
            model = WhisperModel("small", device="cpu", compute_type="int8")
            self.status.emit("Transcribing audio locally…")
            segments, _ = model.transcribe(
                self.audio_path, beam_size=5, vad_filter=True
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            if not transcript:
                transcript = "No speech detected. Please try recording again."
            self.finished_text.emit(transcript)
        except Exception as exc:
            self.failed.emit(
                "Transcription failed. Check your microphone and model download.\n\n"
                + str(exc)
            )


class AnswerWorker(QThread):
    chunk = Signal(str)
    finished_answer = Signal()
    failed = Signal(str)

    def __init__(self, question, model_name):
        super().__init__()
        self.question = question
        self.model_name = model_name

    def run(self):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model_name,
                    "prompt": (
                        "You are a helpful interview and everyday assistant. "
                        "Answer the user's question clearly and accurately. "
                        "Use concise paragraphs or bullets when useful.\n\n"
                        f"Question: {self.question}\n\nAnswer:"
                    ),
                    "stream": True,
                },
                stream=True,
                timeout=(10, 600),
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                item = __import__("json").loads(line.decode("utf-8"))
                token = item.get("response", "")
                if token:
                    self.chunk.emit(token)
                if item.get("done"):
                    break
            self.finished_answer.emit()
        except requests.exceptions.ConnectionError:
            self.failed.emit(
                "Cannot connect to Ollama. Open Ollama and run:\n"
                f"ollama pull {self.model_name}"
            )
        except Exception as exc:
            self.failed.emit("Answer generation failed:\n\n" + str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(780, 650)
        self.audio_chunks = []
        self.stream = None
        self.recording = False
        self.transcription_worker = None
        self.answer_worker = None
        self.build_ui()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        title = QLabel("Local Answer Assistant")
        title.setObjectName("title")
        subtitle = QLabel("Private, on-device speech transcription and AI answers")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.status = QLabel("Ready • Ollama must be running locally")
        self.status.setObjectName("status")
        layout.addWidget(self.status)

        controls = QHBoxLayout()
        self.start_btn = QPushButton("●  Start recording")
        self.start_btn.setObjectName("primary")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch()
        layout.addLayout(controls)

        layout.addWidget(self.section_label("TRANSCRIPTION"))
        self.transcript = QTextEdit()
        self.transcript.setPlaceholderText(
            "Your microphone transcription will appear here…"
        )
        self.transcript.setMinimumHeight(150)
        layout.addWidget(self.transcript)

        answer_controls = QHBoxLayout()
        answer_controls.addWidget(self.section_label("SUGGESTED ANSWER"))
        answer_controls.addStretch()
        self.model_select = QComboBox()
        self.model_select.addItems(["qwen2.5:3b", "llama3.2:3b", "gemma2:2b"])
        self.model_select.setToolTip("Ollama model already installed on this computer")
        answer_controls.addWidget(QLabel("Local model:"))
        answer_controls.addWidget(self.model_select)
        layout.addLayout(answer_controls)

        self.answer = QTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setPlaceholderText(
            "Click Suggest Answer to generate a local AI response…"
        )
        self.answer.setMinimumHeight(190)
        layout.addWidget(self.answer)

        bottom = QHBoxLayout()
        self.suggest_btn = QPushButton("✦  Suggest Answer")
        self.suggest_btn.setObjectName("primary")
        self.suggest_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear")
        bottom.addWidget(self.suggest_btn)
        bottom.addWidget(self.clear_btn)
        bottom.addStretch()
        layout.addLayout(bottom)

        note = QLabel(
            "Audio and AI processing run on this computer. First-time model setup "
            "requires an internet connection to download model files."
        )
        note.setObjectName("note")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.suggest_btn.clicked.connect(self.suggest_answer)
        self.clear_btn.clicked.connect(self.clear_all)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #101522; color: #edf2ff;
                font-family: Segoe UI, Arial; font-size: 13px; }
            QLabel#title { font-size: 27px; font-weight: 700; color: #ffffff; }
            QLabel#subtitle, QLabel#note { color: #9aa8c4; }
            QLabel#status { color: #82e6c1; padding: 9px 12px;
                background: #172b2b; border-radius: 8px; }
            QTextEdit { background: #171f30; border: 1px solid #2b3850;
                border-radius: 10px; padding: 12px; color: #edf2ff;
                selection-background-color: #526bdb; }
            QPushButton { background: #27334a; color: #f3f6ff;
                border: 1px solid #3a4965; border-radius: 9px;
                padding: 11px 17px; font-weight: 600; }
            QPushButton:hover { background: #354562; }
            QPushButton:disabled { color: #6d7890; background: #20293a; }
            QPushButton#primary { background: #5969e8; border: none; }
            QPushButton#primary:hover { background: #6d7bf4; }
            QComboBox { background: #171f30; border: 1px solid #35435c;
                border-radius: 7px; padding: 7px; min-width: 130px; }
        """)

    def section_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("color:#9eacd0; font-size:11px; font-weight:700; letter-spacing:1px;")
        return label

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self.audio_chunks.append(indata.copy())

    def start_recording(self):
        try:
            self.audio_chunks = []
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=self.audio_callback
            )
            self.stream.start()
            self.recording = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.suggest_btn.setEnabled(False)
            self.status.setText("Recording… speak clearly, then press Stop")
        except Exception as exc:
            QMessageBox.critical(self, "Microphone error", str(exc))

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        try:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            if not self.audio_chunks:
                raise RuntimeError("No audio was captured. Check microphone permissions.")
            audio = np.concatenate(self.audio_chunks, axis=0).flatten()
            temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_int16 = np.clip(audio, -1, 1)
            audio_int16 = (audio_int16 * 32767).astype(np.int16)
            with wave.open(temp.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(audio_int16.tobytes())
            self.audio_path = temp.name
            self.stop_btn.setEnabled(False)
            self.start_btn.setEnabled(True)
            self.status.setText("Audio captured • preparing transcription…")
            self.transcription_worker = TranscriptionWorker(self.audio_path)
            self.transcription_worker.status.connect(self.status.setText)
            self.transcription_worker.finished_text.connect(self.on_transcription)
            self.transcription_worker.failed.connect(self.on_error)
            self.transcription_worker.start()
        except Exception as exc:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            QMessageBox.critical(self, "Recording error", str(exc))

    def on_transcription(self, text):
        self.transcript.setPlainText(text)
        self.suggest_btn.setEnabled(bool(text.strip()) and not text.startswith("No speech"))
        self.status.setText("Transcription complete • ready to suggest an answer")
        try:
            os.unlink(self.audio_path)
        except Exception:
            pass

    def suggest_answer(self):
        question = self.transcript.toPlainText().strip()
        if not question:
            QMessageBox.information(self, "No question", "Record or enter a question first.")
            return
        self.answer.clear()
        self.suggest_btn.setEnabled(False)
        self.status.setText("Generating answer locally…")
        self.answer_worker = AnswerWorker(question, self.model_select.currentText())
        self.answer_worker.chunk.connect(self.append_answer)
        self.answer_worker.finished_answer.connect(self.answer_finished)
        self.answer_worker.failed.connect(self.on_error)
        self.answer_worker.start()

    def append_answer(self, token):
        self.answer.moveCursor(QTextCursor.End)
        self.answer.insertPlainText(token)
        self.answer.ensureCursorVisible()

    def answer_finished(self):
        self.suggest_btn.setEnabled(True)
        self.status.setText("Answer generated • processed by your local Ollama model")

    def on_error(self, message):
        self.status.setText("Something needs attention")
        self.suggest_btn.setEnabled(bool(self.transcript.toPlainText().strip()))
        QMessageBox.warning(self, "Assistant", message)

    def clear_all(self):
        if self.recording:
            self.stop_recording()
        self.transcript.clear()
        self.answer.clear()
        self.suggest_btn.setEnabled(False)
        self.status.setText("Ready • Ollama must be running locally")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
