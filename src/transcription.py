"""
Transcription pipeline — Groq Whisper + optional OpenRouter cleanup.

Two entry points:
- `transcribe(audio_data)` — snippet path (Koe hotkey). Returns flat polished string.
- `transcribe_groq_segments(audio_data, label)` — meeting path (Scribe). Returns
  list of {start, end, text, label} with chunk-offset-corrected timestamps.

Long audio (>10 min at 16kHz) is auto-chunked under Groq's 25MB upload limit.
"""

import io
import os
import wave
from datetime import datetime
from pathlib import Path
import numpy as np
import requests

from utils import ConfigManager

# ---------- debug logging ----------

_DEBUG_LOG = Path(__file__).parent.parent / "logs" / "debug.log"
_DEBUG_LOG.parent.mkdir(exist_ok=True)


def _debug(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [transcription] {msg}\n")
    except Exception:
        pass


# ---------- rolling snippet storage ----------

MAX_SNIPPETS = 5


def _get_snippets_dir() -> Path:
    snippets_folder = ConfigManager.get_config_value('misc', 'snippets_folder')
    if snippets_folder:
        snippets_dir = Path(snippets_folder)
    else:
        snippets_dir = Path(__file__).parent.parent / "Snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    return snippets_dir


def save_rolling_transcription(text: str):
    """Save snippet to rolling markdown files (keeps last 5). Newest is 1, oldest is 5."""
    if not text or not text.strip():
        return
    try:
        snippets_dir = _get_snippets_dir()

        # Delete oldest, shift the rest up
        oldest = snippets_dir / f"snippet_{MAX_SNIPPETS}.md"
        if oldest.exists():
            oldest.unlink()
        for i in range(MAX_SNIPPETS - 1, 0, -1):
            old = snippets_dir / f"snippet_{i}.md"
            new = snippets_dir / f"snippet_{i+1}.md"
            if old.exists():
                old.rename(new)

        # Write new as snippet_1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"# Snippet\n\n**Time:** {timestamp}\n\n---\n\n{text.strip()}\n"
        (snippets_dir / "snippet_1.md").write_text(content, encoding='utf-8')
    except Exception as e:
        _debug(f"  save_rolling_transcription error: {e}")


# ---------- Groq transcription ----------

# Groq's 25MB upload cap = ~13 min at 16kHz mono int16. Chunk at 10 min for headroom.
GROQ_CHUNK_MAX_SAMPLES = 10 * 60 * 16000
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def _audio_to_wav_bytes(audio_int16: np.ndarray, sample_rate: int = 16000) -> io.BytesIO:
    """Pack int16 PCM samples into an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return buf


def _ensure_int16(audio_data: np.ndarray) -> np.ndarray:
    """Convert audio to int16 PCM (Whisper's expected format)."""
    if audio_data.dtype == np.float32:
        return np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
    return audio_data.astype(np.int16)


def _groq_post(buf: io.BytesIO, data: dict, api_key: str, timeout: float):
    """Single Groq POST. Returns (parsed_response | None, error_str | None).

    For text/json responses, parsed_response is whatever Groq returned.
    On retry-eligible errors (5xx, timeout), retries once.
    """
    for attempt in range(2):
        try:
            response = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", buf, "audio/wav")},
                data=data,
                timeout=timeout,
            )
            if response.status_code == 200:
                if data.get("response_format") == "verbose_json":
                    return response.json(), None
                return response.text if data.get("response_format") == "text" else response.json().get("text", ""), None

            if response.status_code >= 500 and attempt == 0:
                _debug(f"  Groq {response.status_code}, retrying...")
                buf.seek(0)
                continue

            err = f"Groq HTTP {response.status_code}: {response.text[:200]}"
            _debug(f"  {err}")
            return None, err

        except requests.Timeout:
            if attempt == 0:
                _debug(f"  Groq timeout, retrying...")
                buf.seek(0)
                continue
            return None, "Groq timeout"
        except requests.RequestException as e:
            return None, f"Groq request error: {e}"

    return None, "Groq request failed"


def _groq_request_data(response_format: str = "text") -> dict:
    """Build the form data for a Groq Whisper request from current config."""
    model_options = ConfigManager.get_config_section('model_options') or {}
    common = model_options.get('common', {}) or {}
    data = {
        "model": "whisper-large-v3",
        "language": common.get('language') or 'en',
        "response_format": response_format,
    }
    initial_prompt = common.get('initial_prompt')
    if initial_prompt:
        data["prompt"] = initial_prompt
    return data


def transcribe_groq(audio_data: np.ndarray) -> str:
    """Snippet-style transcription. Returns flat text. Auto-chunks long audio."""
    _debug("transcribe_groq() STARTED")
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        _debug("  ERROR: GROQ_API_KEY not set")
        ConfigManager.console_print("Error: GROQ_API_KEY not set in .env")
        return ''

    audio_int16 = _ensure_int16(audio_data)
    data = _groq_request_data(response_format="text")
    total = len(audio_int16)

    if total <= GROQ_CHUNK_MAX_SAMPLES:
        buf = _audio_to_wav_bytes(audio_int16)
        result, err = _groq_post(buf, data, api_key, timeout=60)
        if err:
            ConfigManager.console_print(f"Groq error: {err}")
            return ''
        return result if isinstance(result, str) else ''

    num_chunks = (total + GROQ_CHUNK_MAX_SAMPLES - 1) // GROQ_CHUNK_MAX_SAMPLES
    _debug(f"  Long audio ({total/16000:.0f}s), {num_chunks} chunks")
    parts = []
    for i in range(num_chunks):
        start = i * GROQ_CHUNK_MAX_SAMPLES
        end = min(start + GROQ_CHUNK_MAX_SAMPLES, total)
        buf = _audio_to_wav_bytes(audio_int16[start:end])
        text, err = _groq_post(buf, data, api_key, timeout=120)
        if err:
            _debug(f"  Chunk {i+1} failed: {err}")
            continue
        if text:
            parts.append(text if isinstance(text, str) else '')
    return ' '.join(p.strip() for p in parts if p)


def transcribe_groq_segments(audio_data: np.ndarray, label: str = "Speaker") -> list[dict]:
    """Meeting-style transcription with sentence-level timestamps.

    Returns a list of {start, end, text, label} dicts. Long audio is chunked
    at 10-min boundaries; chunk segment timestamps are offset back to the
    original timeline so the caller can interleave streams by start time.
    """
    _debug(f"transcribe_groq_segments() STARTED label={label}")
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        _debug("  ERROR: GROQ_API_KEY not set")
        return []

    audio_int16 = _ensure_int16(audio_data)
    data = _groq_request_data(response_format="verbose_json")
    total = len(audio_int16)
    sample_rate = 16000

    segments_out: list[dict] = []
    chunk_count = (total + GROQ_CHUNK_MAX_SAMPLES - 1) // GROQ_CHUNK_MAX_SAMPLES if total else 0

    for i in range(chunk_count):
        start_sample = i * GROQ_CHUNK_MAX_SAMPLES
        end_sample = min(start_sample + GROQ_CHUNK_MAX_SAMPLES, total)
        chunk_offset_sec = start_sample / sample_rate
        buf = _audio_to_wav_bytes(audio_int16[start_sample:end_sample])
        result, err = _groq_post(buf, data, api_key, timeout=120)
        if err:
            _debug(f"  Chunk {i+1}/{chunk_count} failed: {err}")
            continue
        if not isinstance(result, dict):
            continue

        # Apply hallucination regex per chunk before joining
        from utils import TextProcessor

        for seg in result.get("segments", []):
            text = TextProcessor.remove_filler_words(seg.get("text", "").strip())
            if not text:
                continue
            segments_out.append({
                "start": float(seg.get("start", 0.0)) + chunk_offset_sec,
                "end": float(seg.get("end", 0.0)) + chunk_offset_sec,
                "text": text,
                "label": label,
            })
        _debug(f"  Chunk {i+1}/{chunk_count}: {len(result.get('segments', []))} segments")

    _debug(f"transcribe_groq_segments() FINISHED, {len(segments_out)} total segments")
    return segments_out


# ---------- post-processing & cleanup ----------

def post_process_transcription(transcription: str) -> str:
    """Apply regex post-processing (filler words, hallucination tail strip)."""
    from utils import TextProcessor
    return TextProcessor.process(transcription, add_trailing_space=True)


def _provider_pin_for_model(model_id: str):
    """OpenRouter provider pin per model. None = let OpenRouter pick."""
    pins = {
        "google/gemini-3-flash-preview":         ["Google AI Studio"],
        "google/gemini-3.1-flash-lite-preview":  ["Google AI Studio"],
        "anthropic/claude-haiku-4-5":            ["Anthropic"],
        "anthropic/claude-sonnet-4-6":           ["Anthropic"],
        "openai/gpt-5.4-mini":                   ["OpenAI"],
        "deepseek/deepseek-v3.2":                ["Friendli"],
    }
    order = pins.get(model_id)
    return {"order": order, "allow_fallbacks": False} if order else None


def ai_cleanup_transcription(text: str) -> str:
    """Cleanup grammar/punctuation via OpenRouter. Falls back to original on any failure."""
    if not text or not text.strip():
        return text
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            _debug("  No OPENROUTER_API_KEY, skipping cleanup")
            return text

        model = ConfigManager.get_config_value('post_processing', 'ai_cleanup_model') or 'google/gemini-3-flash-preview'
        prompt_prefix = ConfigManager.get_config_value('post_processing', 'ai_cleanup_prompt') or (
            "Clean up this transcription. Fix grammar, add proper punctuation, and remove filler words.\n\n"
            "Output ONLY the cleaned text, nothing else (no quotes, no explanation).\n\nTranscription:\n"
        )
        prompt = prompt_prefix + text.strip()

        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 16384,
        }
        pin = _provider_pin_for_model(model)
        if pin:
            body["provider"] = pin

        _debug(f"  Calling OpenRouter ({model}, provider={pin})")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )

        if response.status_code != 200:
            _debug(f"  OpenRouter HTTP {response.status_code}: {response.text[:300]}")
            return text

        data = response.json()
        if data.get("error"):
            _debug(f"  OpenRouter API error: {data['error']}")
            return text

        cleaned = data["choices"][0]["message"]["content"].strip()
        _debug(f"  cleanup ok: {len(text)} -> {len(cleaned)} chars")
        if not cleaned.endswith(' '):
            cleaned += ' '
        return cleaned

    except Exception as e:
        _debug(f"  cleanup error: {e}")
        return text


# ---------- top-level snippet entry point ----------

def transcribe(audio_data: np.ndarray) -> str:
    """Snippet path: Groq → regex post-process → optional cleanup → save → return."""
    _debug("transcribe() STARTED")
    if audio_data is None:
        return ''

    audio_duration_sec = len(audio_data) / 16000
    _debug(f"  Duration: {audio_duration_sec:.1f}s")

    raw = transcribe_groq(audio_data)
    _debug(f"  Raw: {len(raw)} chars")

    result = post_process_transcription(raw)
    _debug(f"  Post-processed: {len(result)} chars")

    cleanup_enabled = ConfigManager.get_config_value('post_processing', 'ai_cleanup_enabled')
    threshold = ConfigManager.get_config_value('post_processing', 'ai_cleanup_threshold') or 10
    if cleanup_enabled and audio_duration_sec >= threshold:
        result = ai_cleanup_transcription(result)
        from utils import TextProcessor
        result = TextProcessor.remove_filler_words(result)
        if result and not result.endswith(' '):
            result += ' '
    else:
        _debug(f"  Cleanup skipped (enabled={cleanup_enabled}, dur={audio_duration_sec:.1f}s, threshold={threshold}s)")

    save_rolling_transcription(result)
    _debug("transcribe() FINISHED")
    return result
