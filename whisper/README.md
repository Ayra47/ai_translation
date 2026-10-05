# Speech-to-Text Translation Tool

A Python application that listens to your speech in **Russian**, **English**, or **Vietnamese** and translates it to your target language using:
- **Whisper** - for speech recognition
- **Ollama (translategemma)** - for translation

## Prerequisites

1. **Python 3.8+** installed
2. **ffmpeg** installed:
   ```bash
   brew install ffmpeg  # macOS
   # or: sudo apt install ffmpeg  # Linux
   ```
3. **Ollama** installed and running with the `translategemma` model:
   ```bash
   ollama pull translategemma
   ```

## Installation

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or: venv\Scripts\activate  # On Windows
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. On macOS, you may need to install PortAudio:
   ```bash
   brew install portaudio
   ```

## Configuration

Edit `config.py` to customize settings:

```python
TARGET_LANGUAGE = "English"  # Change to your desired output language
WHISPER_MODEL = "base"       # Options: tiny, base, small, medium, large
RECORDING_DURATION = 5       # Seconds to record per session
```

Supported target languages: Russian, English, Vietnamese, Spanish, French, German, Chinese, Japanese, Korean, and more.

## Usage

Run the application:
```bash
python main.py
```

1. Press **Enter** to start recording
2. Speak in Russian, English, or Vietnamese (5 seconds)
3. The app will transcribe and translate your speech
4. Results are saved to timestamped text files

Press **Ctrl+C** to exit.

## Output

Each translation is saved to a file like:
```
translation_20260328_143022.txt
```

## Troubleshooting

- **Microphone not working**: Check system audio permissions
- **Ollama connection error**: Ensure Ollama is running (`ollama serve`)
- **Poor transcription**: Try a larger Whisper model in `config.py`
