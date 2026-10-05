"""
stt.py — Dual-model Speech-to-Text + Language Detection

Architecture
============

  Case A — Vietnamese NOT in config:
    faster-whisper only, hard-restricted to configured languages.
    Uses detect_language() to get per-language probabilities,
    picks the best among allowed langs, forces that language for
    transcription (prevents hallucinated Japanese/Korean chars).

  Case B — Vietnamese IS in config:
    ┌─────────────────────────────────────────────────────────┐
    │  Audio                                                  │
    │    ├─→ PhoWhisper (vinai/PhoWhisper-*)                 │
    │    │      fine-tuned specifically on Vietnamese         │
    │    │      → vi_text, vi_avg_log_prob                    │
    │    │                                                    │
    │    └─→ faster-whisper (restricted to non-vi langs only) │
    │           detect_language() among non-vi langs          │
    │           → best_lang, best_text, other_avg_log_prob    │
    │                                                         │
    │  Winner = higher avg_log_prob (same scale, fair race)   │
    └─────────────────────────────────────────────────────────┘

Why avg_log_prob?
  Both models produce per-token log-probabilities during generation.
  avg_log_prob is the mean across all output tokens — closer to 0
  means the model was more certain about its output.
  -0.1 (Vietnamese) > -0.6 (Russian) → Vietnamese wins.
"""

import numpy as np
import tempfile
import os
import math
from typing import Optional

# ── faster-whisper ────────────────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    _HAS_FASTER_WHISPER = True
except ImportError:
    _HAS_FASTER_WHISPER = False

# ── openai-whisper fallback ───────────────────────────────────────────────────
try:
    import whisper as _openai_whisper
    _HAS_OPENAI_WHISPER = True
except ImportError:
    _HAS_OPENAI_WHISPER = False

if not _HAS_FASTER_WHISPER and not _HAS_OPENAI_WHISPER:
    raise ImportError(
        "No Whisper backend found. Install one:\n"
        "  pip install faster-whisper        # recommended\n"
        "  pip install openai-whisper"
    )

# ── PhoWhisper (transformers) ─────────────────────────────────────────────────
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# ── Language display names ────────────────────────────────────────────────────
LANG_NAMES = {
    "ru": "Russian 🇷🇺",
    "en": "English 🇬🇧",
    "vi": "Vietnamese 🇻🇳",
    "zh": "Chinese 🇨🇳",
    "de": "German 🇩🇪",
    "fr": "French 🇫🇷",
    "es": "Spanish 🇪🇸",
    "ja": "Japanese 🇯🇵",
    "ko": "Korean 🇰🇷",
    "ar": "Arabic 🇸🇦",
    "it": "Italian 🇮🇹",
    "pt": "Portuguese 🇵🇹",
    "tr": "Turkish 🇹🇷",
    "uk": "Ukrainian 🇺🇦",
}

# PhoWhisper size → HuggingFace model ID
PHOWHISPER_MODELS = {
    "tiny":   "vinai/PhoWhisper-tiny",
    "base":   "vinai/PhoWhisper-base",
    "small":  "vinai/PhoWhisper-small",
    "medium": "vinai/PhoWhisper-medium",
    "large":  "vinai/PhoWhisper-large",
}


# ═══════════════════════════════════════════════════════════════════════════════

class SpeechToText:
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        phowhisper_size: str = "base",
        use_phowhisper: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.phowhisper_size = phowhisper_size
        self.use_phowhisper = use_phowhisper

        self._whisper = None        # faster-whisper model
        self._ow_model = None       # openai-whisper model
        self._pho_processor = None  # PhoWhisper processor
        self._pho_model = None      # PhoWhisper model
        self._pho_device = None     # torch device for PhoWhisper

        self._backend = "faster_whisper" if _HAS_FASTER_WHISPER else "openai_whisper"
        print(f"  Whisper backend : {self._backend}")

    # ── Public API ────────────────────────────────────────────────────────────

    def load_models(self, languages: list[str]):
        """Pre-load all models. Call once at startup."""
        print(f"  Loading Whisper '{self.model_name}'...")
        self._load_whisper()
        print(f"  Whisper loaded ✓")

        if "vi" in languages and self.use_phowhisper:
            if _HAS_TRANSFORMERS:
                pho_id = PHOWHISPER_MODELS.get(
                    self.phowhisper_size, PHOWHISPER_MODELS["base"]
                )
                print(f"  Loading PhoWhisper '{pho_id}' (Vietnamese specialist)...")
                self._load_phowhisper(pho_id)
                print(f"  PhoWhisper loaded ✓")
            else:
                print(
                    "  ⚠️  PhoWhisper needs 'transformers' + 'torch':\n"
                    "      pip install transformers torch\n"
                    "  Falling back to Whisper-only for Vietnamese."
                )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        allowed_languages: Optional[list[str]] = None,
    ) -> tuple[str, str, float]:
        """
        Transcribe audio and detect language.
        Returns: (lang_code, text, confidence_0_to_1)
        """
        if len(audio) < sample_rate * 0.3:
            return "", "", 0.0

        audio = _normalize(audio)
        langs = allowed_languages or []

        if "vi" in langs and self._pho_model is not None:
            return self._dual_model_transcribe(audio, sample_rate, langs)
        else:
            return self._whisper_only_transcribe(audio, sample_rate, langs)

    def get_lang_name(self, code: str) -> str:
        return LANG_NAMES.get(code, code.upper())

    # ── Model loaders ─────────────────────────────────────────────────────────

    def _load_whisper(self):
        if self._whisper or self._ow_model:
            return
        if self._backend == "faster_whisper":
            self._whisper = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        else:
            self._ow_model = _openai_whisper.load_model(self.model_name)

    def _load_phowhisper(self, model_id: str):
        self._pho_device = torch.device(
            "cuda"
            if torch.cuda.is_available() and self.device == "cuda"
            else "cpu"
        )
        self._pho_processor = WhisperProcessor.from_pretrained(model_id)
        self._pho_model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        self._pho_model.to(self._pho_device)
        self._pho_model.eval()

    # ── Core strategies ───────────────────────────────────────────────────────

    def _dual_model_transcribe(
        self, audio: np.ndarray, sample_rate: int, allowed_languages: list[str]
    ) -> tuple[str, str, float]:
        """
        Run PhoWhisper (vi) vs faster-whisper (non-vi), compare avg_log_prob.
        """
        # ── Vietnamese hypothesis ────────────────────────────────────────────
        vi_text, vi_score = self._phowhisper_transcribe(audio, sample_rate)

        # ── Non-Vietnamese hypothesis ────────────────────────────────────────
        non_vi = [l for l in allowed_languages if l != "vi"]
        if not non_vi:
            # Only Vietnamese configured — nothing to compare
            return "vi", vi_text, _logprob_to_conf(vi_score)

        other_lang, other_text, other_score = self._whisper_best_of(audio, non_vi)

        # ── Decision ─────────────────────────────────────────────────────────
        # Small prior bonus for PhoWhisper: it's specialized, so we trust it
        # slightly more when the race is close.  Tune 0.0–0.15.
        VI_BONUS = 0.05
        vi_adj = vi_score + VI_BONUS

        if vi_adj >= other_score:
            winner_lang, winner_text, winner_score = "vi", vi_text, vi_score
        else:
            winner_lang, winner_text, winner_score = other_lang, other_text, other_score

        gap = abs(vi_adj - other_score)
        if gap < 0.15:
            print(
                f"  [dual-model] close call — "
                f"vi: {vi_score:+.3f} (adj {vi_adj:+.3f}) | "
                f"{other_lang}: {other_score:+.3f}  →  {winner_lang} wins"
            )

        return winner_lang, winner_text, _logprob_to_conf(winner_score)

    def _whisper_only_transcribe(
        self, audio: np.ndarray, sample_rate: int, allowed_languages: list[str]
    ) -> tuple[str, str, float]:
        """faster-whisper hard-restricted to allowed_languages."""
        if self._backend == "faster_whisper":
            lang, text, raw_conf = self._fw_restricted(audio, allowed_languages)
            return lang, text, raw_conf
        else:
            return self._ow_restricted(audio, sample_rate, allowed_languages)

    # ── PhoWhisper inference ─────────────────────────────────────────────────

    def _phowhisper_transcribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[str, float]:
        """Returns (vietnamese_text, avg_log_prob)."""
        inputs = self._pho_processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        features = inputs.input_features.to(self._pho_device)

        with torch.no_grad():
            out = self._pho_model.generate(
                features,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
            )

        text = self._pho_processor.batch_decode(
            out.sequences, skip_special_tokens=True
        )[0].strip()

        # Compute avg_log_prob over generated tokens
        generated_ids = out.sequences[:, features.shape[-1]:]
        avg_lp = _compute_avg_logprob(out.scores, generated_ids)

        return text, avg_lp

    # ── faster-whisper helpers ────────────────────────────────────────────────

    def _fw_restricted(
        self, audio: np.ndarray, allowed_languages: list[str]
    ) -> tuple[str, str, float]:
        """
        1. detect_language() → full probability dict
        2. pick best lang within allowed set (hard restriction)
        3. force-transcribe with winner (prevents hallucination)
        Returns (lang, text, raw_language_probability).
        """
        raw = self._whisper.detect_language(audio)
        # faster-whisper ≥1.0 returns list of [(lang, prob), ...] per batch
        if isinstance(raw, list):
            lang_probs = dict(raw[0])
        else:
            lang_probs = dict(raw)

        if allowed_languages:
            filtered = {l: p for l, p in lang_probs.items() if l in allowed_languages}
            if filtered:
                detected = max(filtered, key=filtered.get)
                raw_conf = filtered[detected]
                if raw_conf < 0.35:
                    top3 = sorted(lang_probs.items(), key=lambda x: -x[1])[:3]
                    top3_str = ", ".join(f"{l}({p:.0%})" for l, p in top3)
                    print(
                        f"  [whisper] low confidence among config langs "
                        f"({raw_conf:.0%}). Global top3: {top3_str}"
                    )
            else:
                detected = max(lang_probs, key=lang_probs.get)
                raw_conf = lang_probs[detected]
        else:
            detected = max(lang_probs, key=lang_probs.get)
            raw_conf = lang_probs[detected]

        text, _ = self._fw_force_transcribe(audio, detected)
        return detected, text, raw_conf

    def _whisper_best_of(
        self, audio: np.ndarray, languages: list[str]
    ) -> tuple[str, str, float]:
        """
        Among `languages`, pick the one with best avg_log_prob.
        Uses detect_language() first to narrow, then force-transcribes.
        Returns (lang, text, avg_log_prob).
        """
        raw = self._whisper.detect_language(audio)
        if isinstance(raw, list):
            lang_probs = dict(raw[0])
        else:
            lang_probs = dict(raw)

        filtered = {l: p for l, p in lang_probs.items() if l in languages}
        best_by_prob = (
            max(filtered, key=filtered.get) if filtered else languages[0]
        )

        text, avg_lp = self._fw_force_transcribe(audio, best_by_prob)
        return best_by_prob, text, avg_lp

    def _fw_force_transcribe(
        self, audio: np.ndarray, language: str
    ) -> tuple[str, float]:
        """Transcribe with forced language. Returns (text, avg_log_prob)."""
        segments_gen, _ = self._whisper.transcribe(
            audio,
            language=language,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        segs = list(segments_gen)
        text = " ".join(s.text for s in segs).strip()
        avg_lp = (
            sum(s.avg_logprob for s in segs) / len(segs) if segs else -1.0
        )
        return text, avg_lp

    # ── openai-whisper fallback ───────────────────────────────────────────────

    def _ow_restricted(
        self, audio: np.ndarray, sample_rate: int, allowed_languages: list[str]
    ) -> tuple[str, str, float]:
        import scipy.io.wavfile as wav_io

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            wav_io.write(tmp, sample_rate, (audio * 32767).astype(np.int16))
            if not allowed_languages:
                r = self._ow_model.transcribe(tmp)
                return r["language"], r["text"].strip(), 1.0

            best_lang, best_text, best_score = None, "", float("-inf")
            for lang in allowed_languages:
                r = self._ow_model.transcribe(tmp, language=lang)
                segs = r.get("segments", [])
                score = (
                    sum(s.get("avg_logprob", -1.0) for s in segs) / len(segs)
                    if segs else -1.0
                )
                if score > best_score:
                    best_score, best_lang, best_text = score, lang, r["text"].strip()

            return best_lang or allowed_languages[0], best_text, 1.0
        finally:
            os.unlink(tmp)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _normalize(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    peak = np.abs(audio).max()
    return audio / peak * 0.95 if peak > 0 else audio


def _compute_avg_logprob(scores, token_ids) -> float:
    """
    Compute avg log-prob of generated tokens from PhoWhisper output.
    scores:    tuple of (batch, vocab) tensors, one per step
    token_ids: (batch, seq_len) chosen token ids
    """
    if not scores:
        return -1.0
    import torch
    log_probs = []
    for step, step_scores in enumerate(scores):
        if step >= token_ids.shape[1]:
            break
        chosen = token_ids[0, step].item()
        lp = torch.log_softmax(step_scores[0], dim=-1)[chosen].item()
        if lp != float("-inf") and lp == lp:  # skip -inf and NaN
            log_probs.append(lp)
    return sum(log_probs) / len(log_probs) if log_probs else -1.0


def _logprob_to_conf(avg_log_prob: float) -> float:
    """
    Map avg_log_prob (≤ 0) → display confidence (0–1).
    Uses exp() mapping:  0.0 → 1.0,  -0.5 → 0.61,  -1.0 → 0.37
    """
    return min(1.0, max(0.0, math.exp(avg_log_prob)))
