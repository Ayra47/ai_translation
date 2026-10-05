#!/usr/bin/env python3
"""
Speech-to-Text Translation Tool

Records speech in Russian, English, or Vietnamese using Whisper,
then translates to the target language using Ollama's translategemma model.
"""

import whisper
import ollama
import pyaudio
import wave
import tempfile
import os
from datetime import datetime

from config import TARGET_LANGUAGE, WHISPER_MODEL, OLLAMA_MODEL, RECORDING_DURATION


class SpeechTranslator:
    def __init__(self):
        print(f"Loading Whisper model: {WHISPER_MODEL}...")
        self.model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded!")
        
        self.target_language = TARGET_LANGUAGE
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.chunk_size = 1024
        
    def record_audio(self, duration: int = RECORDING_DURATION) -> str:
        """Record audio from microphone and save to temporary file."""
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        
        pyaudio_instance = pyaudio.PyAudio()
        
        stream = pyaudio_instance.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        frames = []
        
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        pyaudio_instance.terminate()
        
        # Save to temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio_instance.get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print("✅ Recording complete!")
        return temp_filename
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text using Whisper."""
        print("📝 Transcribing speech...")
        
        result = self.model.transcribe(audio_path)
        text = result["text"].strip()
        
        print(f"🗣️  Detected text: {text}")
        return text
    
    def translate(self, text: str) -> str:
        """Translate text using Ollama's translategemma model."""
        print(f"🌐 Translating to {self.target_language}...")
        
        prompt = f"Translate the following text to {self.target_language}:\n\n{text}"
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        translation = response['message']['content'].strip()
        print(f"✅ Translation: {translation}")
        return translation
    
    def process(self):
        """Main processing loop."""
        print("\n" + "="*50)
        print("🎙️  Speech-to-Text Translation Tool")
        print(f"📍 Target language: {self.target_language}")
        print(f"🌍 Supported input languages: Russian, English, Vietnamese")
        print("="*50)
        print("\nPress Ctrl+C to exit\n")
        
        try:
            while True:
                input("\n⏹️  Press Enter to start recording...")
                
                # Record audio
                audio_file = self.record_audio()
                
                try:
                    # Transcribe
                    text = self.transcribe(audio_file)
                    
                    if text:
                        # Translate
                        translation = self.translate(text)
                        
                        # Save to file with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"translation_{timestamp}.txt"
                        
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"Original ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n")
                            f.write(f"{text}\n\n")
                            f.write(f"Translation (to {self.target_language}):\n")
                            f.write(f"{translation}\n")
                        
                        print(f"💾 Saved to: {filename}")
                    else:
                        print("⚠️  No speech detected. Try again.")
                
                finally:
                    # Clean up temp file
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")


def main():
    translator = SpeechTranslator()
    translator.process()


if __name__ == "__main__":
    main()
