"""
translator.py — Translation via local Ollama

Uses a locally running Ollama model (e.g. translategemma:latest)
to translate text between languages.
"""

import requests
import json
from stt import LANG_NAMES


# Full language names for the translation prompt
PROMPT_LANG_NAMES = {
    "ru": "Russian",
    "en": "English",
    "vi": "Vietnamese",
    "zh": "Chinese (Simplified)",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "it": "Italian",
    "pt": "Portuguese",
    "tr": "Turkish",
    "uk": "Ukrainian",
}


class Translator:
    def __init__(
        self,
        model: str = "translategemma:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_connection(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            # Check if our model is available (handle version tags)
            model_base = self.model.split(":")[0]
            available = any(model_base in m for m in models)
            if not available:
                print(f"\n⚠️  Model '{self.model}' not found in Ollama.")
                print(f"   Available models: {', '.join(models) or 'none'}")
                print(f"   Run: ollama pull {self.model}")
            return available
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Cannot connect to Ollama at {self.base_url}")
            print("   Make sure Ollama is running: ollama serve")
            return False

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source_lang to target_lang.
        Returns the translated string.
        """
        src_name = PROMPT_LANG_NAMES.get(source_lang, source_lang)
        tgt_name = PROMPT_LANG_NAMES.get(target_lang, target_lang)

        prompt = self._build_prompt(text, src_name, tgt_name)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,   # Low temp for consistent translation
                        "top_p": 0.9,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            translation = result.get("response", "").strip()

            # Clean up common model artifacts
            translation = self._clean_output(translation, text)
            return translation

        except requests.exceptions.Timeout:
            return f"[Translation timeout — try a smaller model or increase ollama_timeout in config]"
        except requests.exceptions.RequestException as e:
            return f"[Translation error: {e}]"

    def _build_prompt(self, text: str, src_name: str, tgt_name: str) -> str:
        return (
            f"Translate the following {src_name} text into {tgt_name}.\n"
            f"Rules:\n"
            f"- Output ONLY the translated text\n"
            f"- No explanations, no notes, no quotation marks around the result\n"
            f"- Preserve the tone and style of the original\n"
            f"- If the text is very short (1-3 words), still translate it faithfully\n\n"
            f"Text to translate:\n{text}"
        )

    def _clean_output(self, translation: str, original: str) -> str:
        """Remove common model artifacts from translation output."""
        # Remove surrounding quotes if model added them
        if (translation.startswith('"') and translation.endswith('"')) or \
           (translation.startswith("'") and translation.endswith("'")):
            translation = translation[1:-1].strip()

        # Remove common preamble patterns
        for prefix in [
            "Translation:", "Translated:", "Here is the translation:",
            "The translation is:", "Result:", "Output:",
        ]:
            if translation.lower().startswith(prefix.lower()):
                translation = translation[len(prefix):].strip()

        return translation

    def get_lang_display(self, code: str) -> str:
        return LANG_NAMES.get(code, code.upper())
