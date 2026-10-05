# 🌍 Voice Translator

Local voice translator using **Whisper** (STT + language detection) and **Ollama** (translation).

## How it works

```
Microphone → [VAD] → Audio → [Whisper] → Text + Language
                                              ↓
                                    Detected language?
                                    ru → translate to en
                                    en → translate to ru
                                              ↓
                                         [Ollama]
                                              ↓
                                        Translation ✅
```

## Setup

### 1. System dependencies

**macOS:**
```bash
brew install portaudio
```

**Ubuntu/Debian:**
```bash
sudo apt install portaudio19-dev python3-dev
```

**Windows:** PortAudio usually installs automatically with sounddevice.

---

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Ollama + translation model

Make sure [Ollama](https://ollama.com) is installed and running:

```bash
ollama serve                       # start Ollama daemon
ollama pull translategemma:latest  # download translation model
```

Alternative models that work well for translation:
```bash
ollama pull aya:8b          # Cohere Aya — great multilingual model
ollama pull gemma3:4b       # General purpose, handles translation well
ollama pull qwen2.5:7b      # Good CJK + multilingual support
```

Then update `ollama_model` in `config.yaml` to match.

---

## Configuration (`config.yaml`)

```yaml
# The 2 languages you'll speak — system detects which and translates to the other
languages:
  - ru   # Russian
  - en   # English

# For 3-way: speak in first 2, always translate to the 3rd
# languages:
#   - ru
#   - en
#   - vi  # always outputs in Vietnamese

whisper_model: "base"          # tiny / base / small / medium / large-v3
ollama_model: "translategemma:latest"
```

---

## Usage

```bash
# Standard voice mode
python main.py

# List available microphone devices
python main.py --list-devices

# Use a specific audio device
python main.py --device 2

# Text-only mode (type instead of speaking — good for testing)
python main.py --text

# Use a different config file
python main.py --config my_config.yaml
```

---

## Session example

```
🌍 Voice Translator 🌍
  Languages : Russian 🇷🇺 ↔ English 🇬🇧
  Whisper   : base
  Translator: translategemma:latest

🎤 Listening — speak now...
  🔴 Recording...
  ⏹  Processing...
  🗣  Language : Russian 🇷🇺
  📝 Original : Привет, как дела?
  🔄 Translating to English 🇬🇧...
  ✅ English 🇬🇧: Hello, how are you?
  ⏱  STT: 0.8s | Translation: 1.2s

🎤 Listening — speak now...
  🔴 Recording...
  ⏹  Processing...
  🗣  Language : English 🇬🇧
  📝 Original : I'm doing great, thanks!
  🔄 Translating to Russian 🇷🇺...
  ✅ Russian 🇷🇺: Я чувствую себя отлично, спасибо!
```

---

## Troubleshooting

**"No speech detected"**
→ Lower `silence_threshold` in config.yaml (e.g. `0.008`)

**Microphone triggers on background noise**
→ Raise `silence_threshold` (e.g. `0.025`)

**Wrong language detected**
→ Use a larger Whisper model: `whisper_model: "small"` or `"medium"`

**Ollama timeout**
→ Increase `ollama_timeout` in config, or use a smaller model like `gemma3:2b`

**Check audio devices:**
```bash
python main.py --list-devices
python main.py --device <index>
```

---

## Supported languages (Whisper codes)

| Code | Language    |
|------|-------------|
| `ru` | Russian     |
| `en` | English     |
| `vi` | Vietnamese  |
| `zh` | Chinese     |
| `de` | German      |
| `fr` | French      |
| `es` | Spanish     |
| `ja` | Japanese    |
| `ko` | Korean      |
| `uk` | Ukrainian   |

Whisper supports 99 languages total.
