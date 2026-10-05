from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import requests
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LIBRETRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "http://localhost:5001")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
logger.info("Whisper model loaded.")

# Map Whisper language codes to LibreTranslate codes (most are the same)
LANG_MAP = {
    "zh": "zh",
    "jw": "jv",  # Javanese fallback
}

LANGUAGE_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "et": "Estonian", "fa": "Persian",
    "fi": "Finnish", "fr": "French", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "jw": "Javanese", "kn": "Kannada",
    "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian", "mk": "Macedonian",
    "ml": "Malayalam", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian",
    "sq": "Albanian", "sr": "Serbian", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Filipino",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "zh": "Chinese",
}


@app.get("/health")
def health():
    return {"status": "ok", "model": WHISPER_MODEL}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Save uploaded audio to temp file
    suffix = ".webm"
    if audio.filename:
        ext = os.path.splitext(audio.filename)[-1]
        if ext:
            suffix = ext

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"Transcribing file: {tmp_path} ({len(content)} bytes)")

        segments, info = model.transcribe(tmp_path, beam_size=5)
        text = " ".join([seg.text for seg in segments]).strip()
        detected_lang = info.language
        confidence = round(info.language_probability * 100)

        logger.info(f"Detected: {detected_lang} ({confidence}%), text: {text[:80]}")

        if not text:
            return {
                "original": "",
                "translation": "",
                "detectedLanguage": detected_lang,
                "detectedLanguageName": LANGUAGE_NAMES.get(detected_lang, detected_lang.upper()),
                "confidence": confidence,
                "error": "No speech detected"
            }

        # Translate if not already English
        translation = text
        translation_error = None

        if detected_lang != "en":
            lt_lang = LANG_MAP.get(detected_lang, detected_lang)
            try:
                resp = requests.post(
                    f"{LIBRETRANSLATE_URL}/translate",
                    json={"q": text, "source": lt_lang, "target": "en", "format": "text"},
                    timeout=30
                )
                if resp.ok:
                    translation = resp.json().get("translatedText", text)
                else:
                    translation_error = f"LibreTranslate error: {resp.status_code}"
                    logger.warning(translation_error)
            except Exception as e:
                translation_error = str(e)
                logger.warning(f"Translation failed: {e}")

        return {
            "original": text,
            "translation": translation,
            "detectedLanguage": detected_lang,
            "detectedLanguageName": LANGUAGE_NAMES.get(detected_lang, detected_lang.upper()),
            "confidence": confidence,
            "translationError": translation_error,
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
