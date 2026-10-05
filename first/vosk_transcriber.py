#!/usr/bin/env python3
"""
Vosk Speech-to-Text Transcriber
Supports: English, Russian (Vosk), Vietnamese (PhoWhisper)
"""

import sys
import os
import wave
import json
import argparse
import signal
import queue
import re
import subprocess
import numpy as np
import warnings
import logging
from threading import Thread
try:
    import tty
    import termios
    import select
    _HAS_KEYMON = True
except ImportError:
    _HAS_KEYMON = False
from vosk import Model, KaldiRecognizer

# Suppress transformers/huggingface warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers.generation").setLevel(logging.ERROR)

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperFeatureExtractor
    import torch
    try:
        import transformers.safetensors_conversion as _sc
        _sc.auto_conversion = lambda *a, **kw: None
    except Exception:
        pass
except ImportError:
    AutoProcessor = None
    AutoModelForSpeechSeq2Seq = None
    torch = None
    WhisperFeatureExtractor = None

try:
    from translator import EraXTranslator
except ImportError:
    EraXTranslator = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import langdetect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0
except ImportError:
    langdetect = None

# Supported languages and their model folders
LANGUAGES = {
    "en": {
        "name": "English",
        "model_folder": "vosk-model-small-en-us-0.15",
        "langdetect_code": "en"
    },
    "ru": {
        "name": "Russian", 
        "model_folder": "vosk-model-small-ru-0.22",
        "langdetect_code": "ru"
    },
    "vi": {
        "name": "Vietnamese",
        "model_folder": "vosk-model-small-vn-0.4",
        "langdetect_code": "vi"
    }
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

LANG_CODE_TO_TRANSLATE_NAME = {
    "en": "English",
    "ru": "Russian",
    "vi": "Vietnamese",
}
TRANSLATE_NAME_TO_CODE = {
    "english": "en", "en": "en",
    "russian": "ru", "ru": "ru",
    "vietnamese": "vi", "vi": "vi",
}
TRANSLATE_NAME_TO_VOICE = {
    "english": "Samantha",
    "russian": "Milena",
    "french": "Thomas",
    "german": "Anna",
    "spanish": "Monica",
    "chinese": "Ting-Ting",
    "japanese": "Kyoko",
}

# Audio settings
SAMPLE_RATE = 16000
BUFFER_SIZE = 4000

running = True

# Language detection mapping
def detect_text_language(text):
    """Detect language of text and return matching vosk language code."""
    if not langdetect:
        return None
    
    try:
        # Clean text
        text = text.strip().lower()
        if len(text) < 3:
            return None
        
        detected = langdetect.detect(text)
        
        # Map detected language to vosk code
        for lang_code, lang_info in LANGUAGES.items():
            if lang_info["langdetect_code"] == detected:
                return lang_code
        
        return None
    except:
        return None


def text_matches_language(text, lang_code):
    """Check if text likely matches the expected language."""
    if not langdetect:
        return True  # Skip check if langdetect not installed
    
    detected = detect_text_language(text)
    if detected is None:
        return True  # Can't detect, assume match
    
    return detected == lang_code


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\n\nStopping...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


class KeyboardMonitor:
    """Non-blocking keyboard listener using termios cbreak mode."""
    def __init__(self):
        self._key_queue = queue.Queue()
        self._thread = Thread(target=self._listen, daemon=True)
        self._active = False
        self._fd = None
        self._old_settings = None

    def start(self):
        if not sys.stdin.isatty() or not _HAS_KEYMON:
            return
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._active = True
        self._thread.start()

    def stop(self):
        self._active = False
        if self._old_settings is not None and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def _listen(self):
        try:
            while self._active:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1)
                    self._key_queue.put(key)
        except Exception:
            pass

    def get_key(self):
        try:
            return self._key_queue.get_nowait()
        except queue.Empty:
            return None


try:
    import piper as _piper
    _HAS_PIPER = True
except ImportError:
    _HAS_PIPER = False

PIPER_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "piper")
PIPER_VI_MODEL = os.path.join(PIPER_MODELS_DIR, "vi_VN-vais1000-medium.onnx")

_piper_vi_voice = None

def _get_piper_vi_voice():
    global _piper_vi_voice
    if _piper_vi_voice is None and _HAS_PIPER and os.path.exists(PIPER_VI_MODEL):
        _piper_vi_voice = _piper.PiperVoice.load(PIPER_VI_MODEL)
    return _piper_vi_voice


class TTSSpeaker:
    """Speaks text asynchronously.

    Vietnamese: Piper TTS (vi_VN-vais1000-medium, offline, high quality).
    Other languages: macOS `say`.
    """
    def __init__(self, voice=None, lang_name=None, speed=1.0):
        self.voice = voice
        self.lang_name = lang_name
        self.speed = speed
        self._queue = queue.Queue()
        self._playing = False
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def is_playing(self):
        return self._playing

    def speak(self, text):
        self._queue.put((text, self.voice, self.lang_name, self.speed))

    @staticmethod
    def _is_vietnamese(lang_name):
        return (lang_name or "").lower() in ("vietnamese", "vi")

    def _speak_piper_vi(self, text, speed=1.0):
        vi_voice = _get_piper_vi_voice()
        if vi_voice is None:
            print("[TTS] Piper model not found, skipping speech")
            return
        import wave, tempfile
        from piper.voice import SynthesisConfig
        syn_cfg = SynthesisConfig(length_scale=speed)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            with wave.open(tmp, "wb") as wf:
                vi_voice.synthesize_wav(text, wf, syn_config=syn_cfg)
            subprocess.run(["afplay", tmp], capture_output=True)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            text, voice, lang_name, speed = item
            self._playing = True
            try:
                if self._is_vietnamese(lang_name):
                    self._speak_piper_vi(text, speed=speed)
                else:
                    cmd = ["say"]
                    if voice:
                        cmd += ["-v", voice]
                    # map length_scale to say rate: 1.0 → 175 wpm, 1.4 → ~125 wpm
                    say_rate = int(175 / speed)
                    cmd += ["-r", str(say_rate)]
                    cmd.append(text)
                    subprocess.run(cmd, capture_output=True)
            except Exception as e:
                print(f"[TTS error] {e}")
            finally:
                self._playing = False

    def stop(self):
        self._queue.put(None)


def get_model_path(lang_code):
    """Get the model path for the specified language."""
    lang_info = LANGUAGES[lang_code]
    model_path = os.path.join(MODELS_DIR, lang_info["model_folder"])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model for {lang_info['name']} not found at {model_path}.\n"
            f"Please download the model to the models folder."
        )
    
    return model_path


def parse_languages(lang_string):
    """Parse language string (e.g., 'en,ru' or 'en') into list of language codes."""
    return [lang.strip().lower() for lang in lang_string.split(',')]


def transcribe_audio(audio_file, lang_codes):
    """Transcribe audio file using Vosk (en, ru) and PhoWhisper (vi)."""

    # Check if Vietnamese is in the list
    use_phowhisper = "vi" in lang_codes
    vosk_lang_codes = [code for code in lang_codes if code != "vi"]

    models = []
    recognizers = []

    # Load Vosk models for non-Vietnamese languages
    for lang_code in vosk_lang_codes:
        model_path = get_model_path(lang_code)
        print(f"Loading Vosk model: {LANGUAGES[lang_code]['name']}...")
        model = Model(model_path)
        models.append(model)
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        recognizer.SetWords(True)
        recognizers.append(recognizer)

    # Load PhoWhisper for Vietnamese
    pho_model = None
    pho_processor = None
    pho_device = None
    if use_phowhisper:
        if AutoProcessor is None or torch is None:
            print("Warning: transformers/torch not installed. Vietnamese recognition disabled.")
            use_phowhisper = False
        else:
            print("Loading PhoWhisper-medium model for Vietnamese...")
            model_name = "vinai/PhoWhisper-medium"
            try:
                pho_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True,
                )
                pho_processor = AutoProcessor.from_pretrained(model_name)
                pho_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                pho_model.to(pho_device)
                pho_model.eval()
            except Exception as e:
                print(f"Error loading PhoWhisper: {e}")
                use_phowhisper = False

    with wave.open(audio_file, "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

        if channels != 1 or sample_width != 2:
            raise ValueError("Audio file must be mono 16-bit PCM. Use ffmpeg to convert.")

        results = []

        # Process with Vosk recognizers
        while True:
            data = wf.readframes(BUFFER_SIZE)
            if len(data) == 0:
                break

            for i, recognizer in enumerate(recognizers):
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if "text" in result and result["text"]:
                        results.append(f"[{LANGUAGES[vosk_lang_codes[i]]['name']}] {result['text']}")

        # Get final results from Vosk recognizers
        for i, recognizer in enumerate(recognizers):
            final_result = json.loads(recognizer.FinalResult())
            if "text" in final_result and final_result["text"]:
                results.append(f"[{LANGUAGES[vosk_lang_codes[i]]['name']}] {final_result['text']}")

        # Process with PhoWhisper for Vietnamese
        if use_phowhisper and pho_model:
            # Reset file position
            wf.setpos(0)
            
            import tempfile
            
            # Copy audio to temp file for PhoWhisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                with wave.open(tmp_path, "wb") as out_wf:
                    out_wf.setnchannels(channels)
                    out_wf.setsampwidth(sample_width)
                    out_wf.setframerate(sample_rate)
                    wf.setpos(0)
                    out_wf.writeframes(wf.readframes(wf.getnframes()))
            
            try:
                print("Transcribing Vietnamese with PhoWhisper...")
                
                # Load audio using soundfile
                if sf:
                    audio_input, _ = sf.read(tmp_path)
                else:
                    # Fallback: read using wave
                    with wave.open(tmp_path, "rb") as tmp_wf:
                        tmp_wf.setpos(0)
                        frames = tmp_wf.readframes(tmp_wf.getnframes())
                        audio_input = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Process audio
                inputs = pho_processor(audio_input, sampling_rate=16000, return_tensors="pt")
                if pho_device:
                    inputs = {k: v.to(pho_device) for k, v in inputs.items()}

                # Generate transcription with constraints
                with torch.no_grad():
                    predicted_ids = pho_model.generate(
                        **inputs,
                        language="vi",
                        task="transcribe",
                        temperature=0.0,
                    )
                
                # Decode transcription
                transcription = pho_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                
                if transcription:
                    results.append(f"[Vietnamese] {transcription}")
            except Exception as e:
                print(f"PhoWhisper error: {e}")
            finally:
                os.unlink(tmp_path)

        return results


def transcribe_microphone(lang_codes, confidence_threshold=0.5, raw_output=False, output_file=None, translate_to=None, speak=False, tts_speed=1.4, gain=1.0):
    """Transcribe audio from microphone in real-time with Vosk (en, ru) and PhoWhisper (vi)."""

    if pyaudio is None:
        print("Error: PyAudio not installed. Install with: pip install pyaudio")
        sys.exit(1)

    if langdetect is None:
        print("Warning: langdetect not installed. Install with: pip install langdetect")
        print("Language detection will be disabled.\n")

    # Check if Vietnamese is in the list
    use_phowhisper = "vi" in lang_codes
    vosk_lang_codes = [code for code in lang_codes if code != "vi"]

    vosk_models = []
    recognizers = []

    # Load Vosk models for non-Vietnamese languages
    for lang_code in vosk_lang_codes:
        model_path = get_model_path(lang_code)
        print(f"Loading Vosk model: {LANGUAGES[lang_code]['name']}...")
        model = Model(model_path)
        vosk_models.append(model)
        recognizer = KaldiRecognizer(model, SAMPLE_RATE)
        recognizer.SetWords(True)
        recognizers.append(recognizer)

    # Load PhoWhisper for Vietnamese
    pho_model = None
    pho_processor = None
    pho_device = None
    if use_phowhisper:
        if AutoProcessor is None or torch is None:
            print("Warning: transformers/torch not installed. Vietnamese recognition disabled.")
            use_phowhisper = False
        else:
            print("Loading PhoWhisper-medium model for Vietnamese...")
            model_name = "vinai/PhoWhisper-medium"
            try:
                pho_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True,
                )
                pho_processor = AutoProcessor.from_pretrained(model_name)
                pho_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                pho_model.to(pho_device)
                pho_model.eval()
            except Exception as e:
                print(f"Error loading PhoWhisper: {e}")
                use_phowhisper = False

    # --- Live language switching setup ---
    swap_vosk_model = None
    swap_lang_code = None
    swap_mode = False
    keyboard = None

    # Vosk ↔ Vosk swap (e.g. -l ru --translate en)
    if translate_to and len(vosk_lang_codes) == 1 and _HAS_KEYMON:
        _swap_code = TRANSLATE_NAME_TO_CODE.get(translate_to.lower())
        if _swap_code and _swap_code != vosk_lang_codes[0] and _swap_code in LANGUAGES and _swap_code != "vi":
            try:
                print(f"Loading {LANGUAGES[_swap_code]['name']} model for live switching...")
                swap_vosk_model = Model(get_model_path(_swap_code))
                swap_lang_code = _swap_code
                swap_mode = True
            except FileNotFoundError:
                print(f"Note: {LANGUAGES[_swap_code]['name']} model not found. Live switching disabled.\n")

    # PhoWhisper (VI) ↔ Vosk swap (e.g. -l vi --translate ru)
    vi_swap_model = None
    vi_swap_lang_code = None
    vi_swap_mode = False

    if use_phowhisper and not vosk_lang_codes and translate_to and _HAS_KEYMON:
        _swap_code = TRANSLATE_NAME_TO_CODE.get(translate_to.lower())
        if _swap_code and _swap_code in LANGUAGES and _swap_code != "vi":
            try:
                print(f"Loading {LANGUAGES[_swap_code]['name']} model for live switching...")
                vi_swap_model = Model(get_model_path(_swap_code))
                vi_swap_lang_code = _swap_code
                vi_swap_mode = True
            except FileNotFoundError:
                print(f"Note: {LANGUAGES[_swap_code]['name']} model not found. Live switching disabled.\n")

    pa = pyaudio.PyAudio()

    # List available devices
    print("\nAvailable audio devices:")
    for i in range(pa.get_device_count()):
        device = pa.get_device_info_by_index(i)
        if device['maxInputChannels'] > 0:
            print(f"  {i}: {device['name']}")
    print()

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=BUFFER_SIZE
        )

        # Build language names string
        lang_names = []
        for code in vosk_lang_codes:
            lang_names.append(LANGUAGES[code]['name'])
        if use_phowhisper:
            lang_names.append("Vietnamese (PhoWhisper)")
        lang_names_str = " + ".join(lang_names)

        print(f"Listening... ({lang_names_str})")
        print(f"Confidence threshold: {confidence_threshold}")
        print("Press Ctrl+C to stop\n")
        print("-" * 50)

        stream.start_stream()

        # Initialize translator
        translator = None
        if translate_to and EraXTranslator:
            try:
                translator = EraXTranslator(target_lang=translate_to)
            except Exception as e:
                print(f"Error loading translator: {e}\n")
                translator = None

        # Initialize TTS speaker
        tts_speaker = None
        if speak and translate_to:
            _voice = TRANSLATE_NAME_TO_VOICE.get(translate_to.lower())
            tts_speaker = TTSSpeaker(voice=_voice, lang_name=translate_to, speed=tts_speed)
            if TTSSpeaker._is_vietnamese(translate_to):
                if _HAS_PIPER and os.path.exists(PIPER_VI_MODEL):
                    print("Loading Piper Vietnamese TTS model...")
                    _get_piper_vi_voice()
                    _label = "Piper vi_VN-vais1000-medium"
                else:
                    _label = "say (fallback, Piper model not found)"
            elif _voice:
                _label = f"say -v {_voice}"
            else:
                _label = "say (default)"
            print(f"TTS enabled ({_label})\n")

        vosk_partial_texts = [""] * len(vosk_lang_codes)
        all_results = []
        last_vosk_full_text = [""] * len(vosk_lang_codes)

        # PhoWhisper state
        pho_audio_buffer = bytearray()
        pho_last_transcription = ""
        pho_transcription_interval = 3.0  # Transcribe every 3 seconds
        last_pho_transcription_time = 0
        pho_min_audio_duration = 2.0  # Minimum 2 seconds of audio
        pho_silence_threshold = 0.005  # Lower threshold for silence detection

        original_lang_code = vosk_lang_codes[0] if vosk_lang_codes else None
        original_vosk_model = vosk_models[0] if vosk_models else None
        is_swapped = False
        _pho_active = use_phowhisper  # gates PhoWhisper processing; toggled in vi_swap_mode

        if swap_mode or vi_swap_mode:
            keyboard = KeyboardMonitor()
            keyboard.start()
            if swap_mode:
                _a = LANGUAGES[vosk_lang_codes[0]]['name']
                _b = LANGUAGES[swap_lang_code]['name']
                _tb = LANG_CODE_TO_TRANSLATE_NAME.get(vosk_lang_codes[0], vosk_lang_codes[0])
                print(f"[1] {_a} → {translate_to}  |  [2] {_b} → {_tb}")
            else:
                _tb = LANG_CODE_TO_TRANSLATE_NAME.get("vi", "Vietnamese")
                print(f"[1] Vietnamese → {translate_to}  |  [2] {LANGUAGES[vi_swap_lang_code]['name']} → {_tb}")
            print("Press 1 or 2 to swap languages\n")

        import time
        import tempfile

        _tts_was_playing = False
        _tts_cooldown = 0
        # Each BUFFER_SIZE/SAMPLE_RATE ≈ 0.25s; 6 frames ≈ 1.5s drain after TTS
        _TTS_DRAIN_FRAMES = 6

        while running:
            data = stream.read(BUFFER_SIZE, exception_on_overflow=False)

            # Microphone gain amplification
            if gain != 1.0:
                _samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                _samples *= gain
                np.clip(_samples, -32768, 32767, out=_samples)
                data = _samples.astype(np.int16).tobytes()

            # Suppress recognition while TTS is speaking and briefly after
            if tts_speaker and tts_speaker.is_playing:
                _tts_was_playing = True
                continue
            if _tts_was_playing:
                _tts_was_playing = False
                _tts_cooldown = _TTS_DRAIN_FRAMES
            if _tts_cooldown > 0:
                _tts_cooldown -= 1
                continue

            # Live language switching via keypresses
            if swap_mode and keyboard:
                _key = keyboard.get_key()
                if _key == '1' and is_swapped:
                    is_swapped = False
                    recognizers[0] = KaldiRecognizer(original_vosk_model, SAMPLE_RATE)
                    recognizers[0].SetWords(True)
                    vosk_lang_codes[0] = original_lang_code
                    vosk_partial_texts[0] = ""
                    last_vosk_full_text[0] = ""
                    if translator:
                        translator.target_lang = translate_to
                    if tts_speaker:
                        tts_speaker.voice = TRANSLATE_NAME_TO_VOICE.get(translate_to.lower())
                        tts_speaker.lang_name = translate_to
                    print(f"\n[►] {LANGUAGES[original_lang_code]['name']} → {translate_to}  ")
                elif _key == '2' and not is_swapped and swap_vosk_model:
                    is_swapped = True
                    recognizers[0] = KaldiRecognizer(swap_vosk_model, SAMPLE_RATE)
                    recognizers[0].SetWords(True)
                    vosk_lang_codes[0] = swap_lang_code
                    vosk_partial_texts[0] = ""
                    last_vosk_full_text[0] = ""
                    _tr = LANG_CODE_TO_TRANSLATE_NAME.get(original_lang_code, original_lang_code)
                    if translator:
                        translator.target_lang = _tr
                    if tts_speaker:
                        tts_speaker.voice = TRANSLATE_NAME_TO_VOICE.get(_tr.lower())
                        tts_speaker.lang_name = _tr
                    print(f"\n[►] {LANGUAGES[swap_lang_code]['name']} → {_tr}  ")

            # VI ↔ Vosk live switching (e.g. -l vi --translate ru)
            if vi_swap_mode and keyboard:
                _key = keyboard.get_key()
                if _key == '1' and not _pho_active:
                    # switch back to Vietnamese (PhoWhisper)
                    _pho_active = True
                    recognizers[:] = []
                    vosk_lang_codes[:] = []
                    vosk_partial_texts[:] = []
                    last_vosk_full_text[:] = []
                    pho_audio_buffer = bytearray()
                    if translator:
                        translator.target_lang = translate_to
                    if tts_speaker:
                        tts_speaker.voice = TRANSLATE_NAME_TO_VOICE.get((translate_to or "").lower())
                        tts_speaker.lang_name = translate_to
                    print(f"\n[►] Vietnamese → {translate_to}  ")
                elif _key == '2' and _pho_active:
                    # switch to Vosk language
                    _pho_active = False
                    _new_rec = KaldiRecognizer(vi_swap_model, SAMPLE_RATE)
                    _new_rec.SetWords(True)
                    recognizers[:] = [_new_rec]
                    vosk_lang_codes[:] = [vi_swap_lang_code]
                    vosk_partial_texts[:] = [""]
                    last_vosk_full_text[:] = [""]
                    _tr = LANG_CODE_TO_TRANSLATE_NAME.get("vi", "Vietnamese")
                    if translator:
                        translator.target_lang = _tr
                    if tts_speaker:
                        tts_speaker.voice = None
                        tts_speaker.lang_name = "vi"
                    print(f"\n[►] {LANGUAGES[vi_swap_lang_code]['name']} → Vietnamese  ")

            # Process with Vosk recognizers
            for i, recognizer in enumerate(recognizers):
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    confidence = result.get("confidence", 0)

                    if text:
                        # Apply language detection boost/penalty
                        if langdetect:
                            if text_matches_language(text, vosk_lang_codes[i]):
                                confidence *= 1.5
                            else:
                                confidence *= 0.3

                        # Save if confidence is good enough and not a duplicate
                        if confidence >= confidence_threshold:
                            is_duplicate = False
                            for _, prev_text in all_results[-5:]:
                                if text.strip().lower() in prev_text.strip().lower() or \
                                   prev_text.strip().lower() in text.strip().lower():
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                all_results.append((vosk_lang_codes[i], text))
                                if not raw_output:
                                    lang_name = LANGUAGES[vosk_lang_codes[i]]['name'][:2].upper()
                                    print(f"\n[{lang_name}] {text}")
                                else:
                                    print(f"\n{text}")

                                # Start translation
                                if translator:
                                    def print_translation(result, text=text):
                                        print(f"    → {result}\n")
                                        if tts_speaker:
                                            tts_speaker.speak(result)
                                    translator.translate_async(text, print_translation)

                                last_vosk_full_text[i] = text
                                vosk_partial_texts[i] = ""

                else:
                    # Handle partial results for Vosk
                    partial_result = json.loads(recognizer.PartialResult())
                    partial_text = partial_result.get("partial", "")

                    if partial_text != vosk_partial_texts[i]:
                        old = vosk_partial_texts[i]

                        if not partial_text and old:
                            # Phrase completed
                            if not raw_output:
                                lang_name = LANGUAGES[vosk_lang_codes[i]]['name'][:2].upper()
                                print(f"\r[{lang_name}] {old}          ")
                            else:
                                print(f"\r{old}          ")
                            all_results.append((vosk_lang_codes[i], old))

                            # Start translation
                            if translator:
                                def print_translation(result, text=old):
                                    print(f"    → {result}\n")
                                    if tts_speaker:
                                        tts_speaker.speak(result)
                                translator.translate_async(old, print_translation)
                        elif partial_text:
                            # Phrase in progress
                            if not raw_output:
                                lang_name = LANGUAGES[vosk_lang_codes[i]]['name'][:2].upper()
                                print(f"\r[{lang_name}] {partial_text}          ", end="", flush=True)
                            else:
                                print(f"\r{partial_text}          ", end="", flush=True)

                        vosk_partial_texts[i] = partial_text

            # Process with PhoWhisper for Vietnamese
            if _pho_active and pho_model and pho_processor:
                pho_audio_buffer.extend(data)

                current_time = time.time()
                min_buffer_size = int(SAMPLE_RATE * pho_min_audio_duration * 2)
                if len(pho_audio_buffer) >= min_buffer_size:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_path = tmp_file.name
                        with wave.open(tmp_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(bytes(pho_audio_buffer))

                    try:
                        if sf:
                            audio_input, _ = sf.read(tmp_path)
                        else:
                            with wave.open(tmp_path, "rb") as tmp_wf:
                                frames = tmp_wf.readframes(tmp_wf.getnframes())
                                audio_input = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

                        rms = np.sqrt(np.mean(audio_input ** 2))
                        if rms < pho_silence_threshold:
                            pho_audio_buffer = pho_audio_buffer[-(SAMPLE_RATE * 1 * 2):]
                            continue

                        inputs = pho_processor(audio_input, sampling_rate=16000, return_tensors="pt")
                        if pho_device:
                            inputs = {k: v.to(pho_device) for k, v in inputs.items()}

                        with torch.no_grad():
                            predicted_ids = pho_model.generate(
                                **inputs,
                                language="vi",
                                task="transcribe",
                                temperature=0.0,
                            )

                        text = pho_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

                        if text and text != pho_last_transcription:
                            if len(text.split()) < 2:
                                pho_audio_buffer = pho_audio_buffer[-(SAMPLE_RATE * 1 * 2):]
                                continue

                            is_duplicate = False
                            for _, prev_text in all_results[-5:]:
                                if text.strip().lower() in prev_text.strip().lower() or \
                                   prev_text.strip().lower() in text.strip().lower():
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                all_results.append(("vi", text))
                                if not raw_output:
                                    print(f"\n[VI] {text}")
                                else:
                                    print(f"\n{text}")
                                if translator:
                                    def print_translation(result, text=text):
                                        print(f"    → {result}\n")
                                        if tts_speaker:
                                            tts_speaker.speak(result)
                                    translator.translate_async(text, print_translation)
                                pho_last_transcription = text

                    except Exception as e:
                        print(f"PhoWhisper error: {e}")
                    finally:
                        os.unlink(tmp_path)

                    pho_audio_buffer = pho_audio_buffer[-(SAMPLE_RATE * 1 * 2):]
                    last_pho_transcription_time = current_time

        # Final results
        print("\n\n--- All Recognized Text ---")

        # Wait for translations to complete
        if translator:
            print("\nWaiting for translations to complete...")
            translator._queue.join()
            translator.stop()

        # Get final results from Vosk recognizers
        for i, recognizer in enumerate(recognizers):
            final_result = json.loads(recognizer.FinalResult())
            text = final_result.get("text", "")
            confidence = final_result.get("confidence", 0)

            if text and confidence >= confidence_threshold:
                is_duplicate = False
                for _, prev_text in all_results:
                    if text.strip().lower() == prev_text.strip().lower():
                        is_duplicate = True
                        break
                if not is_duplicate:
                    all_results.append((vosk_lang_codes[i], text))

        # Add any partial Vosk text
        for i, partial_text in enumerate(vosk_partial_texts):
            if partial_text and partial_text.strip():
                is_duplicate = False
                for _, prev_text in all_results:
                    if partial_text.strip().lower() in prev_text.strip().lower() or \
                       prev_text.strip().lower() in partial_text.strip().lower():
                        is_duplicate = True
                        break
                if not is_duplicate:
                    all_results.append((vosk_lang_codes[i], partial_text))

        # Final PhoWhisper transcription (remaining buffer)
        if use_phowhisper and pho_model and pho_processor and len(pho_audio_buffer) > 0:
            if len(pho_audio_buffer) >= int(SAMPLE_RATE * pho_min_audio_duration * 2):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    with wave.open(tmp_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(bytes(pho_audio_buffer))
                try:
                    if sf:
                        audio_input, _ = sf.read(tmp_path)
                    else:
                        with wave.open(tmp_path, "rb") as tmp_wf:
                            frames = tmp_wf.readframes(tmp_wf.getnframes())
                            audio_input = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = np.sqrt(np.mean(audio_input ** 2))
                    if rms >= pho_silence_threshold:
                        inputs = pho_processor(audio_input, sampling_rate=16000, return_tensors="pt")
                        if pho_device:
                            inputs = {k: v.to(pho_device) for k, v in inputs.items()}
                        with torch.no_grad():
                            predicted_ids = pho_model.generate(
                                **inputs, language="vi", task="transcribe", temperature=0.0,
                            )
                        text = pho_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
                        if text and text != pho_last_transcription and len(text.split()) >= 2:
                            if not any(text.strip().lower() == p.strip().lower() for _, p in all_results):
                                all_results.append(("vi", text))
                except Exception as e:
                    print(f"Final PhoWhisper error: {e}")
                finally:
                    os.unlink(tmp_path)

        # Output results
        if all_results:
            # Remove duplicates
            seen = set()
            unique_results = []
            for lang_code, text in all_results:
                key = (lang_code, text.strip().lower())
                if key not in seen:
                    seen.add(key)
                    unique_results.append((lang_code, text))

            for i, (lang_code, text) in enumerate(unique_results, 1):
                if raw_output:
                    print(f"{i}. {text}")
                else:
                    if lang_code == "vi":
                        print(f"{i}. [Vietnamese] {text}")
                    else:
                        lang_name = LANGUAGES[lang_code]['name']
                        print(f"{i}. [{lang_name}] {text}")

            # Save to file if requested
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    for lang_code, text in unique_results:
                        f.write(f"{text}\n")
                print(f"\nResults saved to: {output_file}")
        else:
            print("No speech detected")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if keyboard:
            keyboard.stop()
        if tts_speaker:
            tts_speaker.stop()
        stream.stop_stream()
        stream.close()
        pa.terminate()


def list_languages():
    """List all supported languages."""
    print("\nSupported languages:")
    for code, info in LANGUAGES.items():
        print(f"  {code} - {info['name']}")
    print()


def list_models():
    """List available local models."""
    print("\nAvailable models in 'models' folder:")
    if not os.path.exists(MODELS_DIR):
        print("  No models folder found.")
        print("\nDownload models from: https://alphacephei.com/vosk/models")
        print("Place them in the 'models' folder.\n")
        return
    
    models = os.listdir(MODELS_DIR)
    if not models:
        print("  No models found.")
        print("\nDownload models from: https://alphacephei.com/vosk/models")
        print("Place them in the 'models' folder.\n")
        return
    
    for model in models:
        model_path = os.path.join(MODELS_DIR, model)
        if os.path.isdir(model_path):
            print(f"  ✓ {model}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Vosk Speech-to-Text Transcriber (Vosk for en/ru, PhoWhisper for vi)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vosk_transcriber.py -l en -i audio.wav       # Single language (file)
  python vosk_transcriber.py -l en,ru --mic           # Two languages (mic)
  python vosk_transcriber.py -l ru,vi --mic           # Russian (Vosk) + Vietnamese (PhoWhisper)
  python vosk_transcriber.py -l en,ru,vi --mic        # All three languages
  python vosk_transcriber.py -l en,ru --mic --raw     # Raw output (for neural network)
  python vosk_transcriber.py -l en,ru --mic --confidence 0.5
  python vosk_transcriber.py -l en,ru --mic -o result.txt  # Save to file
  python vosk_transcriber.py -l ru --mic --translate English  # Translate to English
  python vosk_transcriber.py -l vi --mic              # Vietnamese only (PhoWhisper)
  python vosk_transcriber.py --list-languages
  python vosk_transcriber.py --list-models
        """
    )
    
    parser.add_argument(
        "-l", "--language",
        help="Language code(s), comma-separated (e.g., 'en', 'en,ru', 'ru,vi')"
    )
    parser.add_argument(
        "-i", "--input",
        help="Input audio file (WAV format, mono, 16-bit PCM)"
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Use microphone for real-time transcription"
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all supported languages"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available local models"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output only text without language tags (for neural network input)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.1,
        help="Confidence threshold (default: 0.1)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file to save transcription results"
    )

    parser.add_argument(
        "--translate",
        metavar="LANG",
        help="Переводить на язык (например: English, Vietnamese, French)"
    )

    parser.add_argument(
        "--speak",
        action="store_true",
        help="Озвучивать перевод через системный TTS (macOS say)"
    )

    parser.add_argument(
        "--tts-speed",
        type=float,
        default=1.4,
        metavar="N",
        help="Скорость TTS: 1.0 = нормально, >1 = медленнее (по умолчанию: 1.4)"
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        metavar="N",
        help="Усиление микрофона: 1.0 = без усиления, 2.0 = вдвое громче (по умолчанию: 1.0)"
    )

    args = parser.parse_args()
    
    if args.list_languages:
        list_languages()
        return
    
    if args.list_models:
        list_models()
        return
    
    if not args.language:
        parser.print_help()
        sys.exit(1)
    
    lang_codes = parse_languages(args.language)
    
    # Validate languages
    for lang in lang_codes:
        if lang not in LANGUAGES:
            print(f"Error: Unknown language '{lang}'")
            print(f"Supported: {', '.join(LANGUAGES.keys())}")
            sys.exit(1)
    
    if args.mic:
        transcribe_microphone(lang_codes, args.confidence, args.raw, args.output, args.translate, args.speak, args.tts_speed, args.gain)
    elif args.input:
        if not os.path.exists(args.input):
            print(f"Error: Audio file '{args.input}' not found.")
            sys.exit(1)
        print(f"Transcribing {args.input}...")
        try:
            results = transcribe_audio(args.input, lang_codes)
            print("\n--- Transcription Results ---")
            for result in results:
                print(result)
            print("-----------------------------\n")
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
