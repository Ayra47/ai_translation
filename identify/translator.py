#!/usr/bin/env python3
"""
Speech Translator between English, Russian, and Vietnamese
Uses Vosk for speech-to-text, langdetect for language detection, and Ollama for translation
"""

import json
import sys
import os
import subprocess
import wave
import datetime
from pathlib import Path

import pyaudio
from vosk import Model, KaldiRecognizer

# Language codes mapping
LANG_NAMES = {
    "en": "English",
    "ru": "Russian",
    "vi": "Vietnamese"
}


class SpeechTranslator:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.source_languages = self.config["source_languages"]
        self.sample_rate = self.config["sample_rate"]
        self.vosk_models = self.config.get("vosk_models", {})
        self.audio_output_dir = self.config.get("audio_output_dir", "./recordings")
        self.language_detect_model = self.config.get("language_detect_model", "qwen2.5:latest")
        self.translation_model = self.config.get("translation_model", "translategemma:latest")
        
        # Create recordings directory
        Path(self.audio_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load all Vosk models
        print("Loading Vosk models...")
        self.models = {}
        for lang, model_path in self.vosk_models.items():
            if lang in self.source_languages:
                try:
                    self.models[lang] = Model(model_path)
                    print(f"  Loaded model for {LANG_NAMES.get(lang, lang)}: {model_path}")
                except Exception as e:
                    print(f"  Warning: Could not load model for {lang}: {e}")
        
        if not self.models:
            print("Error: No Vosk models loaded. Check your config and model paths.")
            sys.exit(1)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        
        print(f"\nSource languages (bidirectional): {[LANG_NAMES.get(l, l) for l in self.source_languages]}")
        print(f"Speaking Russian → translates to English")
        print(f"Speaking English → translates to Russian")

    def detect_language(self, text):
        """Detect the language of the input text using Ollama"""
        prompt = f"""Detect the language of this text. Return ONLY the language code (en, ru, or vi), nothing else:

Text: "{text}"

Language code:"""
        
        try:
            result = subprocess.run(
                ["ollama", "run", self.language_detect_model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                detected = result.stdout.strip().lower()
                # Extract language code from response
                for code in ["en", "ru", "vi"]:
                    if code in detected:
                        return code
                # Fallback: check first 2 chars
                if detected[:2] in self.source_languages:
                    return detected[:2]
            return None
        except subprocess.TimeoutExpired:
            print("  Language detection timeout")
            return None
        except Exception as e:
            print(f"  Language detection error: {e}")
            return None

    def get_target_for_language(self, spoken_language):
        """
        Determine the target language based on the spoken language.
        For 2 source languages: translate to the other one.
        """
        if len(self.source_languages) == 2:
            if spoken_language == self.source_languages[0]:
                return self.source_languages[1]
            elif spoken_language == self.source_languages[1]:
                return self.source_languages[0]
        
        # Fallback: return first language that's not the spoken one
        for lang in self.source_languages:
            if lang != spoken_language:
                return lang
        
        return self.source_languages[0]

    def save_audio(self, audio_data):
        """Save audio data to a WAV file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(self.audio_output_dir, filename)
        
        try:
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            return filepath
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

    def transcribe_audio(self, audio_data, lang_code):
        """Transcribe audio using the appropriate Vosk model"""
        if lang_code not in self.models:
            print(f"No model available for {lang_code}")
            return None
        
        model = self.models[lang_code]
        recognizer = KaldiRecognizer(model, self.sample_rate)
        
        # Feed all audio data to recognizer
        recognizer.AcceptWaveform(audio_data)
        
        # Get final result
        result = json.loads(recognizer.Result())
        text = result.get("text", "").strip()
        
        # If no final result, try partial
        if not text:
            partial = json.loads(recognizer.PartialResult())
            text = partial.get("partial", "").strip()
        
        return text

    def transcribe_with_all_models(self, audio_data):
        """
        Try to transcribe with all available models and return the best result.
        Returns (text, detected_language)
        """
        results = {}

        for lang_code, model in self.models.items():
            recognizer = KaldiRecognizer(model, self.sample_rate)
            
            # Feed all audio data
            recognizer.AcceptWaveform(audio_data)
            
            # Get final result
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            
            # If no final result, try partial
            if not text:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "").strip()
            
            if text:
                results[lang_code] = text
                print(f"  [{lang_code}] Transcribed: {text}")
        
        if not results:
            return "", None
        
        # If only one model produced results, use that
        if len(results) == 1:
            lang = list(results.keys())[0]
            return results[lang], lang
        
        # If multiple models produced results, use langdetect on the text
        best_lang = None
        best_text = ""
        best_confidence = 0
        
        for lang, text in results.items():
            try:
                detected = detect(text)
                if detected == lang:
                    return text, lang  # Perfect match
                elif detected in self.source_languages:
                    return text, detected
            except:
                pass
            
            # Fallback: use the longest text
            if len(text) > len(best_text):
                best_text = text
                best_lang = lang
        
        return best_text, best_lang

    def translate_with_ollama(self, text, source_lang, target_lang):
        """Translate text using Ollama translation model"""
        prompt = f"Translate the following text from {LANG_NAMES.get(source_lang, source_lang)} to {LANG_NAMES.get(target_lang, target_lang)}. Return only the translation, no explanations:\n\n{text}"
        
        try:
            result = subprocess.run(
                ["ollama", "run", self.translation_model, prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"Translation error: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            print("Translation timeout")
            return None
        except Exception as e:
            print(f"Translation error: {e}")
            return None

    def record_audio_chunk(self, duration=3.0):
        """Record a chunk of audio"""
        frames = []
        
        # Record for specified duration
        # At 16000 Hz with 4000 buffer, we get 4 chunks per second
        chunks_per_second = self.sample_rate // 4000
        total_chunks = int(duration * chunks_per_second)
        
        for _ in range(total_chunks):
            if not self.is_running:
                break
            data = self.stream.read(4000, exception_on_overflow=False)
            frames.append(data)
        
        return b"".join(frames)

    def detect_speech(self, audio_data, threshold=300):
        """Detect if audio contains speech (simple energy-based detection)"""
        import struct
        shorts = struct.unpack(f"{len(audio_data)//2}h", audio_data)
        rms = (sum(s*s for s in shorts) / len(shorts)) ** 0.5
        print(f"  Audio RMS: {rms:.1f} (threshold: {threshold})")
        return rms > threshold

    def process_and_translate(self):
        """Main loop for recording and translating speech"""
        print("\n=== Speech Translator Ready ===")
        print("Speak in any of the source languages. Press Ctrl+C to stop.\n")

        self.is_running = True
        
        # Open audio stream once
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=4000
        )
        
        self.stream.start_stream()

        silence_count = 0
        max_silence = 3  # Stop recording after 3 silent chunks

        try:
            while self.is_running:
                print("Listening...")

                # Record audio
                audio_data = self.record_audio_chunk(duration=3.0)
                
                # Check if audio contains speech
                if not self.detect_speech(audio_data):
                    print("  (no speech detected)")
                    continue
                
                # Save audio to file
                audio_file = self.save_audio(audio_data)
                if audio_file:
                    print(f"  Audio saved: {audio_file}")
                
                # Transcribe with all models
                print("  Transcribing...")
                text, detected_lang = self.transcribe_with_all_models(audio_data)
                
                if not text:
                    print("  (no text recognized)")
                    continue
                
                # If model-based detection failed, use langdetect
                if not detected_lang or detected_lang not in self.source_languages:
                    detected_lang = self.detect_language(text)
                
                if detected_lang and detected_lang in self.source_languages:
                    # Determine target language
                    target_lang = self.get_target_for_language(detected_lang)
                    
                    print(f"\n[{LANG_NAMES.get(detected_lang, detected_lang)}] You said: {text}")
                    print(f"Translating to {LANG_NAMES.get(target_lang, target_lang)}...")
                    
                    # Translate
                    translation = self.translate_with_ollama(text, detected_lang, target_lang)
                    
                    if translation:
                        print(f"[{LANG_NAMES.get(target_lang, target_lang)}] Translation: {translation}")
                    else:
                        print("Translation failed.")
                    
                    print()
                else:
                    print(f"\nDetected unsupported language: {detected_lang}")
                    print(f"Text: {text}\n")
                
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop the translator"""
        self.is_running = False
        if self.stream:
            self.stream.close()
        self.audio.terminate()
        print("Translator stopped.")


def check_dependencies(config):
    """Check if required dependencies are available"""
    errors = []
    language_detect_model = config.get("language_detect_model", "qwen2.5:latest")
    translation_model = config.get("translation_model", "translategemma:latest")
    
    # Check Ollama
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append("Ollama is not installed or not in PATH")
    except FileNotFoundError:
        errors.append("Ollama is not installed")
    
    # Check if required models are available
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if language_detect_model.replace(":latest", "") not in result.stdout.lower():
            print(f"Warning: {language_detect_model} model may not be installed.")
            print(f"Run: ollama pull {language_detect_model}")
        if translation_model.replace(":latest", "") not in result.stdout.lower():
            print(f"Warning: {translation_model} model may not be installed.")
            print(f"Run: ollama pull {translation_model}")
    except:
        pass
    
    return errors


def main():
    # Check config file
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Check dependencies
    errors = check_dependencies(config)
    if errors:
        print("Dependency errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Create and run translator
    translator = SpeechTranslator(config_path)
    translator.process_and_translate()


if __name__ == "__main__":
    main()
