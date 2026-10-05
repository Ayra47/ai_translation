# Speech Translator (EN ↔ RU ↔ VI)

A real-time speech translator between English, Russian, and Vietnamese using Vosk for speech-to-text and Ollama (translategemma) for translation.

## Features

- 🎤 Real-time speech recognition using Vosk
- 🌍 Automatic language detection (langdetect)
- 🔄 Smart translation between 3 languages
- 🏠 Local processing with Ollama (no API keys needed)

## How It Works

1. You configure 2 source languages you'll speak (e.g., Russian and English)
2. The system detects which language you're speaking
3. It automatically translates to the other language

**Example:**
- You speak Russian: "привет как дела?" → Translates to English: "hello how are you?"
- You speak English: "I'm good" → Translates to Russian: "я хорошо"

## Prerequisites

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

### 2. Pull the translation model

```bash
ollama pull translategemma:latest
```

### 3. Download Vosk Model

Download a multilingual Vosk model:

```bash
# Option 1: Smaller model (good for testing)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 vosk-model

# Option 2: Larger multilingual model (recommended)
# Download from: https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip
# Or other models at: https://alphacephei.com/vosk/models
```

For best results with multiple languages, download a model that supports your languages:
- English: `vosk-model-small-en-us-0.15`
- Russian: `vosk-model-ru-0.22`
- Vietnamese: (use a multilingual model or combine models)

### 4. Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

**Note for macOS:** You may need to install PortAudio first:
```bash
brew install portaudio
```

## Configuration

Edit `config.json` to set your languages:

```json
{
    "source_languages": ["ru", "en"],
    "target_language": "vi",
    "vosk_model_path": "./vosk-model",
    "sample_rate": 16000
}
```

**Parameters:**
- `source_languages`: The 2 languages you'll speak (array of language codes)
- `target_language`: Currently not used in 2-language mode (auto-calculated)
- `vosk_model_path`: Path to your Vosk model directory
- `sample_rate`: Audio sample rate (default: 16000)

**Language codes:**
- `en` - English
- `ru` - Russian
- `vi` - Vietnamese

## Usage

```bash
python3 translator.py
```

Then:
1. Allow microphone access when prompted
2. Start speaking in either of your source languages
3. The translator will detect the language and translate to the other
4. Press `Ctrl+C` to stop

## Example Session

```
=== Speech Translator Ready ===
Source languages: [Russian, English]
Target language: Vietnamese

Listening...

[Russian] You said: привет как дела?
Translating to English...
[English] Translation: hello how are you?

Listening...

[English] You said: I'm good thanks
Translating to Russian...
[Russian] Translation: я хорошо спасибо

Listening...
```

## Troubleshooting

### "Ollama is not installed"
```bash
brew install ollama
ollama serve  # Start Ollama server in background
```

### "translategemma model not found"
```bash
ollama pull translategemma:latest
```

### "Vosk model not found"
Download a model from https://alphacephei.com/vosk/models and update `config.json`

### "No microphone detected"
- Check System Preferences → Security & Privacy → Privacy → Microphone
- Ensure your microphone is selected and working

### Poor recognition accuracy
- Try a larger Vosk model
- Speak clearly and reduce background noise
- Consider using a language-specific model

## Project Structure

```
.
├── config.json          # Configuration file
├── translator.py        # Main translator script
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── vosk-model/         # Vosk model directory (download separately)
```

## License

MIT
