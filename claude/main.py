#!/usr/bin/env python3
"""
voice-translator — main.py

A local voice translator using:
  - faster-whisper  →  speech-to-text + language detection
  - Ollama          →  translation (translategemma:latest or any model)

Usage:
  python main.py                  # Start with config.yaml
  python main.py --config my.yaml # Use custom config file
  python main.py --list-devices   # Show audio input devices
  python main.py --text           # Text-only mode (type instead of speaking)
"""

import sys
import time
import argparse
import yaml
from pathlib import Path

# ── Colour helpers ──────────────────────────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def c(text, color): return f"{color}{text}{Style.RESET_ALL}"
    RED    = Fore.RED
    GREEN  = Fore.GREEN
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BLUE   = Fore.BLUE
    MAGENTA= Fore.MAGENTA
    BOLD   = Style.BRIGHT
except ImportError:
    def c(text, _): return text
    RED = GREEN = YELLOW = CYAN = BLUE = MAGENTA = BOLD = ""


# ── Load config ─────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Banner ───────────────────────────────────────────────────────────────────
def print_banner(cfg: dict, languages: list[str], stt_mod, translator_mod):
    lang_display = " ↔ ".join(
        c(stt_mod.get_lang_name(l), CYAN) for l in languages
    )
    print()
    print(c("╔══════════════════════════════════════════╗", BLUE))
    print(c("║        🌍  Voice Translator  🌍           ║", BLUE))
    print(c("╚══════════════════════════════════════════╝", BLUE))
    print(f"  Languages : {lang_display}")
    print(f"  Whisper   : {c(cfg['whisper_model'], YELLOW)}")
    print(f"  Translator: {c(cfg['ollama_model'], YELLOW)}")
    print()


# ── How translation routing works ────────────────────────────────────────────
def get_target_language(detected_lang: str, languages: list[str]) -> str | None:
    """
    With 2 languages [A, B]:
      detect A → translate to B
      detect B → translate to A

    With 3 languages [A, B, C]:
      detect A or B → translate to C  (C is the fixed target)
      detect C      → translate to A  (fallback)
    """
    if len(languages) == 2:
        if detected_lang == languages[0]:
            return languages[1]
        elif detected_lang == languages[1]:
            return languages[0]
        else:
            return None  # unknown language

    elif len(languages) >= 3:
        source_langs = languages[:2]
        target_lang  = languages[2]
        if detected_lang in source_langs:
            return target_lang
        else:
            return None

    return None


# ── Single translation round ─────────────────────────────────────────────────
def run_once(cfg: dict, languages: list[str], recorder, stt, translator):
    rec_cfg = cfg.get("recording", {})

    # --- Record ---
    def on_speech_start():
        print(c("  🔴 Recording... (stop speaking to finish)", RED))

    def on_speech_end():
        print(c("  ⏹  Processing...", YELLOW))

    print(c("\n🎤 Listening — speak now...", GREEN))
    audio = recorder.record(
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
    )

    if len(audio) == 0:
        print(c("  ⚠️  No audio captured", YELLOW))
        return

    # --- STT + Language Detection ---
    t0 = time.time()
    detected_lang, text, confidence = stt.transcribe(
        audio,
        sample_rate=rec_cfg.get("sample_rate", 16000),
        allowed_languages=languages,
    )
    stt_time = time.time() - t0

    if not text:
        print(c("  ⚠️  No speech detected (try speaking louder or adjust silence_threshold)", YELLOW))
        return

    if cfg.get("show_detected_language", True):
        conf_str = f"  ({confidence:.0%} confidence)" if confidence < 0.95 else ""
        print(f"  🗣  Language : {c(stt.get_lang_name(detected_lang), CYAN)}{conf_str}")

    if cfg.get("show_transcription", True):
        print(f"  📝 Original : {c(text, BOLD)}")

    # --- Route to target language ---
    target_lang = get_target_language(detected_lang, languages)

    if target_lang is None:
        available = ", ".join(stt.get_lang_name(l) for l in languages)
        print(c(f"  ⚠️  Detected '{stt.get_lang_name(detected_lang)}' which is not in your config.", YELLOW))
        print(c(f"     Expected one of: {available}", YELLOW))
        print(c(f"     Tip: increase whisper_model size in config.yaml for better accuracy", YELLOW))
        return

    # --- Translate ---
    t1 = time.time()
    print(f"  🔄 Translating to {c(stt.get_lang_name(target_lang), CYAN)}...")
    translation = translator.translate(text, detected_lang, target_lang)
    trans_time = time.time() - t1

    print(f"  ✅ {c(stt.get_lang_name(target_lang), CYAN)}: {c(translation, GREEN)}")
    print(c(f"  ⏱  STT: {stt_time:.1f}s | Translation: {trans_time:.1f}s", Style.DIM if hasattr(Style, 'DIM') else ""))


# ── Text mode (for testing without microphone) ────────────────────────────────
def run_text_mode(cfg: dict, languages: list[str], stt, translator):
    print(c("\n📝 Text mode — type your text, press Enter to translate", YELLOW))
    print(c("   (Ctrl+C to quit)\n", YELLOW))

    while True:
        try:
            text = input(c("You: ", GREEN)).strip()
            if not text:
                continue

            # Detect language from text using a simple approach
            import langdetect
            detected_lang = langdetect.detect(text)

            if cfg.get("show_detected_language", True):
                print(f"  🗣  Language : {c(stt.get_lang_name(detected_lang), CYAN)}")

            target_lang = get_target_language(detected_lang, languages)
            if target_lang is None:
                print(c(f"  ⚠️  Language '{detected_lang}' not in config", YELLOW))
                continue

            translation = translator.translate(text, detected_lang, target_lang)
            print(f"  ✅ {c(stt.get_lang_name(target_lang), CYAN)}: {c(translation, GREEN)}\n")

        except KeyboardInterrupt:
            break


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Voice Translator")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--list-devices", action="store_true", help="List audio input devices")
    parser.add_argument("--text", action="store_true", help="Text-only mode (no microphone)")
    parser.add_argument("--device", type=int, default=None, help="Audio device index to use")
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(c(f"❌ Config file not found: {cfg_path}", RED))
        sys.exit(1)

    cfg = load_config(str(cfg_path))
    languages: list[str] = cfg.get("languages", ["ru", "en"])
    rec_cfg = cfg.get("recording", {})

    if len(languages) < 2:
        print(c("❌ Please specify at least 2 languages in config.yaml", RED))
        sys.exit(1)

    # Imports (deferred so --list-devices is fast)
    from recorder import AudioRecorder
    from stt import SpeechToText
    from translator import Translator

    # List devices
    if args.list_devices:
        AudioRecorder().list_devices()
        sys.exit(0)

    # Init components
    print(c("\n⚙️  Initializing...", YELLOW))
    import sounddevice as sd
    if args.device is not None:
        sd.default.device = args.device

    recorder = AudioRecorder(
        sample_rate=rec_cfg.get("sample_rate", 16000),
        channels=rec_cfg.get("channels", 1),
        silence_threshold=rec_cfg.get("silence_threshold", 0.015),
        silence_duration=rec_cfg.get("silence_duration", 1.5),
        max_duration=rec_cfg.get("max_duration", 30),
        pre_buffer_duration=rec_cfg.get("pre_buffer_duration", 0.5),
    )

    stt = SpeechToText(
        model_name=cfg.get("whisper_model", "base"),
        device=cfg.get("whisper_device", "cpu"),
        compute_type=cfg.get("whisper_compute_type", "int8"),
    )

    translator = Translator(
        model=cfg.get("ollama_model", "translategemma:latest"),
        base_url=cfg.get("ollama_url", "http://localhost:11434"),
        timeout=cfg.get("ollama_timeout", 60),
    )

    # Check Ollama
    print(c("  Checking Ollama connection...", YELLOW))
    if not translator.check_connection():
        print(c("\n  ⚠️  Continuing anyway — translations will fail until Ollama is ready.", YELLOW))

    # Load Whisper model eagerly
    stt._load_model()

    print_banner(cfg, languages, stt, translator)

    # --- Text mode ---
    if args.text:
        run_text_mode(cfg, languages, stt, translator)
        return

    # --- Voice mode ---
    print(c("  Press Ctrl+C to quit\n", Style.DIM if hasattr(Style, 'DIM') else ""))

    while True:
        try:
            run_once(cfg, languages, recorder, stt, translator)
        except KeyboardInterrupt:
            print(c("\n\n👋 Goodbye!", GREEN))
            break
        except Exception as e:
            print(c(f"\n  ❌ Error: {e}", RED))
            print(c("     Continuing...", YELLOW))
            time.sleep(0.5)


if __name__ == "__main__":
    main()
