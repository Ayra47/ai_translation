"""
stt.py — Speech-to-Text + Language Detection using faster-whisper

Uses OpenAI's Whisper model locally to:
  1. Detect which language was spoken
  2. Transcribe the audio to text
"""

import numpy as np
import tempfile
import os
from typing import Optional

try:
    from faster_whisper import WhisperModel
    BACKEND = "faster_whisper"
except ImportError:
    try:
        import whisper as openai_whisper
        BACKEND = "openai_whisper"
    except ImportError:
        raise ImportError(
            "No Whisper backend found. Install one of:\n"
            "  pip install faster-whisper\n"
            "  pip install openai-whisper"
        )


# Human-readable language names
LANG_NAMES = {
    "ru": "Russian 🇷🇺",
    "en": "English 🇬🇧",
    "vi": "Vietnamese 🇻🇳",
    "zh": "Chinese 🇨🇳",
    "de": "German 🇩🇪",
    "fr": "French 🇫🇷",
    "es": "Spanish 🇪🇸",
    "ja": "Japanese 🇯🇵",
    "ko": "Korean 🇰🇷",
    "ar": "Arabic 🇸🇦",
    "it": "Italian 🇮🇹",
    "pt": "Portuguese 🇵🇹",
    "tr": "Turkish 🇹🇷",
    "uk": "Ukrainian 🇺🇦",
}


class SpeechToText:
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_name = model_name
        self.backend = BACKEND
        self._model = None
        self.device = device
        self.compute_type = compute_type
        print(f"  STT backend: {BACKEND}")

    def _load_model(self):
        if self._model is not None:
            return
        print(f"  Loading Whisper '{self.model_name}' model (first run may download)...")
        if self.backend == "faster_whisper":
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        else:
            self._model = openai_whisper.load_model(self.model_name)
        print(f"  Whisper model loaded ✓")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        allowed_languages: Optional[list[str]] = None,
    ) -> tuple[str, str, float]:
        """
        Transcribe audio and detect language.
        
        Returns:
            (detected_language_code, transcribed_text, confidence)
            e.g. ("ru", "привет как дела", 0.98)
        """
        self._load_model()

        if len(audio) < sample_rate * 0.3:  # Less than 0.3s of audio
            return "", "", 0.0

        # Normalize audio
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95

        if self.backend == "faster_whisper":
            return self._transcribe_faster_whisper(audio, sample_rate, allowed_languages)
        else:
            return self._transcribe_openai_whisper(audio, sample_rate, allowed_languages)

    def _transcribe_faster_whisper(self, audio, sample_rate, allowed_languages):
        lang_filter = allowed_languages if allowed_languages else None
        
        segments, info = self._model.transcribe(
            audio,
            language=None,          # auto-detect
            beam_size=5,
            best_of=5,
            vad_filter=True,        # filter out non-speech
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        
        text = " ".join(seg.text for seg in segments).strip()
        detected_lang = info.language
        confidence = info.language_probability

        # If detected language isn't in our expected list, note it
        if allowed_languages and detected_lang not in allowed_languages:
            # Still return it — caller decides what to do
            pass

        return detected_lang, text, confidence

    def _transcribe_openai_whisper(self, audio, sample_rate, allowed_languages):
        import scipy.io.wavfile as wav_io

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            wav_io.write(tmp_path, sample_rate, (audio * 32767).astype(np.int16))
            result = self._model.transcribe(tmp_path, language=None)
            text = result["text"].strip()
            detected_lang = result["language"]
            # openai-whisper doesn't expose confidence easily
            confidence = 1.0
        finally:
            os.unlink(tmp_path)

        return detected_lang, text, confidence

    def get_lang_name(self, code: str) -> str:
        return LANG_NAMES.get(code, code.upper())
