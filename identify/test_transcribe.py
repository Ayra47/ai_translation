#!/usr/bin/env python3
"""Test transcription of existing audio files"""

import json
import wave
from vosk import Model, KaldiRecognizer

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Test with one of the recordings
test_file = "recordings/recording_20260328_160749.wav"

print(f"Testing with: {test_file}")
print()

# Check WAV file info
with wave.open(test_file, "rb") as wf:
    print(f"WAV Info:")
    print(f"  Channels: {wf.getnchannels()}")
    print(f"  Sample width: {wf.getsampwidth()}")
    print(f"  Frame rate: {wf.getframerate()}")
    print(f"  Frames: {wf.getnframes()}")
    print()

# Load each model and try to transcribe
for lang, model_path in config["vosk_models"].items():
    if lang not in config["source_languages"]:
        continue
    
    print(f"Testing model: {lang} - {model_path}")
    try:
        model = Model(model_path)
        
        with wave.open(test_file, "rb") as wf:
            recognizer = KaldiRecognizer(model, wf.getframerate())
            
            # Read all audio data
            audio_data = wf.readframes(wf.getnframes())
            
            # Process audio
            if recognizer.AcceptWaveform(audio_data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
            else:
                partial = json.loads(recognizer.PartialResult())
                text = partial.get("partial", "")
            
            print(f"  Result: '{text}'")
            print()
    except Exception as e:
        print(f"  Error: {e}")
        print()
