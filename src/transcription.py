"""
Transcription pipeline — ElevenLabs Scribe v2/Groq + optional OpenRouter cleanup.

Two entry points:
- `transcribe(audio_data)` — snippet path (Koe hotkey). Returns flat polished string.
- `transcribe_segments(audio_data, label)` — meeting path (Scribe). Returns
  list of {start, end, text, label} with chunk-offset-corrected timestamps.

Long audio (>10 min at 16kHz) is auto-chunked for stable cloud requests.
"""

import io
import os
import re
import wave
from datetime import datetime
from pathlib import Path
import numpy as np
import requests
from dotenv import load_dotenv

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
MAX_DEBUG_SNIPPETS = 5


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


def save_transcription_debug(raw: str, post_processed: str, final: str, duration_sec: float):
    """Save recent raw/processed/final text locally so cutoff reports are diagnosable."""
    try:
        debug_dir = Path(__file__).parent.parent / "logs" / "transcription_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        oldest = debug_dir / f"snippet_debug_{MAX_DEBUG_SNIPPETS}.md"
        if oldest.exists():
            oldest.unlink()
        for i in range(MAX_DEBUG_SNIPPETS - 1, 0, -1):
            old = debug_dir / f"snippet_debug_{i}.md"
            new = debug_dir / f"snippet_debug_{i+1}.md"
            if old.exists():
                old.rename(new)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = (
            f"# Snippet Debug\n\n"
            f"**Time:** {timestamp}\n"
            f"**Duration:** {duration_sec:.2f}s\n\n"
            "## Raw Transcription\n\n"
            f"{(raw or '').strip()}\n\n"
            "## Post Processed\n\n"
            f"{(post_processed or '').strip()}\n\n"
            "## Final\n\n"
            f"{(final or '').strip()}\n"
        )
        (debug_dir / "snippet_debug_1.md").write_text(content, encoding='utf-8')
    except Exception as e:
        _debug(f"  save_transcription_debug error: {e}")


# ---------- Groq transcription ----------

# Groq's 25MB upload cap = ~13 min at 16kHz mono int16. Chunk at 10 min for headroom.
GROQ_CHUNK_SECONDS = 10 * 60
GROQ_CHUNK_MAX_SAMPLES = GROQ_CHUNK_SECONDS * 16000
GROQ_TRAILING_SILENCE_SEC = 2.0
GROQ_TAIL_RETRY_SECONDS = 15.0
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_API_KEY_NAMES = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "XI_API_KEY")


def _load_env():
    load_dotenv(Path(__file__).parent.parent / ".env")


def _transcription_provider() -> str:
    provider = ConfigManager.get_config_value('model_options', 'transcription_provider')
    return (provider or 'elevenlabs').strip().lower()


def _api_key_from_env(*names: str) -> str:
    _load_env()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ''


def _initial_prompt_keyterms() -> list[str]:
    """Convert the shared vocab hint into ElevenLabs keyterms."""
    model_options = ConfigManager.get_config_section('model_options') or {}
    common = model_options.get('common', {}) or {}
    initial_prompt = common.get('initial_prompt') or ''
    if not initial_prompt:
        return []

    keyterms: list[str] = []
    seen: set[str] = set()
    for raw_term in re.split(r"[,;\n]", str(initial_prompt)):
        term = raw_term.strip().strip(".")
        term = re.sub(r"\s+", " ", term)
        if not term or re.search(r"[<>{}\[\]\\]", term):
            continue
        if len(term.split()) > 5 or len(term) >= 50:
            continue
        lower = term.lower()
        if lower in seen:
            continue
        keyterms.append(term)
        seen.add(lower)
        if len(keyterms) >= 1000:
            break
    return keyterms


def _chunk_max_samples(sample_rate: int) -> int:
    ten_minutes_at_rate = GROQ_CHUNK_SECONDS * int(sample_rate or 16000)
    return min(ten_minutes_at_rate, GROQ_CHUNK_MAX_SAMPLES)


def _audio_to_wav_bytes(
    audio_int16: np.ndarray,
    sample_rate: int = 16000,
    trailing_silence_sec: float = 0.0,
) -> io.BytesIO:
    """Pack int16 PCM samples into an in-memory WAV file."""
    sample_rate = int(sample_rate or 16000)
    if trailing_silence_sec > 0 and audio_int16.size > 0:
        silence_samples = int(sample_rate * trailing_silence_sec)
        if silence_samples > 0:
            audio_int16 = np.concatenate([
                audio_int16,
                np.zeros(silence_samples, dtype=np.int16),
            ])

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


def _boost_quiet_audio_for_whisper(
    audio_int16: np.ndarray,
    target_rms: float = 3000.0,
    max_gain: float = 8.0,
) -> np.ndarray:
    """Boost quiet mic snippets before Whisper without clipping loud audio."""
    if audio_int16.size == 0:
        return audio_int16.astype(np.int16, copy=False)

    audio_f = audio_int16.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio_f ** 2)))
    if rms <= 1e-3 or rms >= target_rms:
        return audio_int16.astype(np.int16, copy=False)

    peak = float(np.max(np.abs(audio_f)))
    clip_limited_gain = (32767.0 / peak) if peak > 0 else max_gain
    gain = min(target_rms / rms, max_gain, clip_limited_gain)
    if gain <= 1.0:
        return audio_int16.astype(np.int16, copy=False)

    _debug(f"  Boosting quiet audio for Whisper: rms={rms:.1f}, gain={gain:.2f}x")
    return np.clip(audio_f * gain, -32768, 32767).astype(np.int16)


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


def _groq_result_text(result) -> str:
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return ''
    if result.get("text"):
        return str(result["text"])
    return ' '.join(
        str(seg.get("text", "")).strip()
        for seg in result.get("segments", [])
        if str(seg.get("text", "")).strip()
    )


def _elevenlabs_request_data() -> list[tuple[str, str]]:
    """Build form fields for ElevenLabs Scribe v2 from current config."""
    model_options = ConfigManager.get_config_section('model_options') or {}
    common = model_options.get('common', {}) or {}
    elevenlabs = model_options.get('elevenlabs', {}) or {}

    data = [
        ("model_id", elevenlabs.get("model_id") or "scribe_v2"),
        ("language_code", common.get('language') or 'en'),
        ("tag_audio_events", "false"),
        ("timestamps_granularity", "word"),
        ("diarize", "false"),
        ("num_speakers", "1"),
        ("temperature", str(elevenlabs.get("temperature", 0.0))),
        ("file_format", "other"),
        ("no_verbatim", "false"),
    ]

    if elevenlabs.get("keyterms_enabled", True):
        data.extend(("keyterms", term) for term in _initial_prompt_keyterms())
    return data


def _elevenlabs_result_text(result) -> str:
    if not isinstance(result, dict):
        return ''
    return str(result.get("text") or '').strip()


def _elevenlabs_post(buf: io.BytesIO, data: list[tuple[str, str]], api_key: str, timeout: float):
    """Single ElevenLabs STT POST. Returns (parsed_response | None, error_str | None)."""
    try:
        response = requests.post(
            ELEVENLABS_URL,
            headers={"xi-api-key": api_key},
            files={"file": ("audio.wav", buf, "audio/wav")},
            data=data,
            timeout=timeout,
        )
    except requests.Timeout:
        return None, "ElevenLabs timeout"
    except requests.RequestException as e:
        return None, f"ElevenLabs request error: {e}"

    try:
        result = response.json()
    except ValueError:
        result = {"_raw": response.text[:500]}

    if response.status_code != 200:
        return None, f"ElevenLabs HTTP {response.status_code}: {response.text[:300]}"
    return result, None


def _transcribe_elevenlabs_audio(
    audio_int16: np.ndarray,
    data: list[tuple[str, str]],
    api_key: str,
    sample_rate: int,
    timeout: float,
) -> dict:
    audio_int16 = _boost_quiet_audio_for_whisper(audio_int16)
    buf = _audio_to_wav_bytes(audio_int16, sample_rate=sample_rate)
    result, err = _elevenlabs_post(buf, data, api_key, timeout=timeout)
    if err:
        _debug(f"  ElevenLabs error: {err}")
        return {}
    return result if isinstance(result, dict) else {}


def _segments_from_elevenlabs_words(result: dict, label: str, offset_sec: float = 0.0) -> list[dict]:
    words = result.get("words") if isinstance(result, dict) else None
    if not isinstance(words, list):
        text = _elevenlabs_result_text(result)
        if not text:
            return []
        return [{"start": offset_sec, "end": offset_sec, "text": text, "label": label}]

    segments: list[dict] = []
    current_words: list[str] = []
    current_start: float | None = None
    current_end = 0.0
    last_end: float | None = None

    def flush():
        nonlocal current_words, current_start, current_end
        text = " ".join(current_words).strip()
        if text and current_start is not None:
            segments.append({
                "start": current_start + offset_sec,
                "end": current_end + offset_sec,
                "text": text,
                "label": label,
            })
        current_words = []
        current_start = None
        current_end = 0.0

    for word in words:
        if not isinstance(word, dict) or word.get("type") not in (None, "word"):
            continue
        text = str(word.get("text") or '').strip()
        if not text:
            continue
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)

        if current_words and last_end is not None and start - last_end > 1.2:
            flush()

        if current_start is None:
            current_start = start
        current_words.append(text)
        current_end = end
        last_end = end

        if text.endswith((".", "?", "!")) or len(current_words) >= 28:
            flush()

    flush()
    return segments


def _transcribe_groq_audio(
    audio_int16: np.ndarray,
    data: dict,
    api_key: str,
    sample_rate: int,
    timeout: float,
) -> str:
    audio_int16 = _boost_quiet_audio_for_whisper(audio_int16)
    buf = _audio_to_wav_bytes(
        audio_int16,
        sample_rate=sample_rate,
        trailing_silence_sec=GROQ_TRAILING_SILENCE_SEC,
    )
    result, err = _groq_post(buf, data, api_key, timeout=timeout)
    if err:
        _debug(f"  Groq error: {err}")
        return ''
    return _groq_result_text(result)


def _word_matches(text: str) -> list[re.Match]:
    return list(re.finditer(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower()))


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return True
    pos = 0
    for word in haystack:
        if word == needle[pos]:
            pos += 1
            if pos == len(needle):
                return True
    return False


def _merge_tail_retry(full_text: str, tail_text: str) -> str:
    """Append final words from the tail retry only when the full pass missed them."""
    full_text = (full_text or '').strip()
    tail_text = (tail_text or '').strip()
    if not full_text or not tail_text:
        return full_text or tail_text

    full_words = _normalise_words(full_text)
    tail_words = _normalise_words(tail_text)
    if len(tail_words) < 4:
        return full_text

    tail_last_words = tail_words[-min(6, len(tail_words)):]
    full_tail_window = full_words[-max(40, len(tail_words) + 8):]
    if _is_subsequence(tail_last_words, full_tail_window):
        return full_text

    max_overlap = min(len(full_words), len(tail_words))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if full_words[-size:] == tail_words[:size]:
            overlap = size
            break

    matches = _word_matches(tail_text)
    if overlap >= len(matches):
        return full_text

    suffix_start = matches[overlap].start() if overlap > 0 else 0
    suffix = tail_text[suffix_start:].lstrip(" \t\r\n,.;:-")
    if len(_normalise_words(suffix)) < 4:
        return full_text

    _debug(f"  Tail retry added {len(_normalise_words(suffix))} words")
    joiner = '' if full_text.endswith((' ', '\n')) else ' '
    return f"{full_text.rstrip()}{joiner}{suffix}"


def transcribe_groq(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Snippet-style transcription. Returns flat text. Auto-chunks long audio."""
    _debug("transcribe_groq() STARTED")
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        _debug("  ERROR: GROQ_API_KEY not set")
        ConfigManager.console_print("Error: GROQ_API_KEY not set in .env")
        return ''

    sample_rate = int(sample_rate or 16000)
    audio_int16 = _ensure_int16(audio_data)
    data = _groq_request_data(response_format="verbose_json")
    total = len(audio_int16)
    max_samples = _chunk_max_samples(sample_rate)

    if total <= max_samples:
        full_text = _transcribe_groq_audio(audio_int16, data, api_key, sample_rate, timeout=60)
    else:
        num_chunks = (total + max_samples - 1) // max_samples
        _debug(f"  Long audio ({total/sample_rate:.0f}s), {num_chunks} chunks")
        parts = []
        for i in range(num_chunks):
            start = i * max_samples
            end = min(start + max_samples, total)
            chunk_text = _transcribe_groq_audio(
                audio_int16[start:end],
                data,
                api_key,
                sample_rate,
                timeout=120,
            )
            if chunk_text:
                parts.append(chunk_text)
        full_text = ' '.join(p.strip() for p in parts if p)

    if total:
        tail_samples = min(total, int(sample_rate * GROQ_TAIL_RETRY_SECONDS))
        tail_audio = audio_int16[-tail_samples:]
        tail_text = _transcribe_groq_audio(tail_audio, data, api_key, sample_rate, timeout=60)
        merged = _merge_tail_retry(full_text, tail_text)
        if merged != full_text:
            _debug(f"  Tail retry merged: full={len(full_text)} chars, tail={len(tail_text)} chars, merged={len(merged)} chars")
        return merged

    return full_text


def transcribe_elevenlabs(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Snippet-style transcription via ElevenLabs Scribe v2."""
    _debug("transcribe_elevenlabs() STARTED")
    api_key = _api_key_from_env(*ELEVENLABS_API_KEY_NAMES)
    if not api_key:
        _debug("  ERROR: ELEVENLABS_API_KEY not set")
        ConfigManager.console_print("Error: ELEVENLABS_API_KEY not set in .env")
        return ''

    sample_rate = int(sample_rate or 16000)
    audio_int16 = _ensure_int16(audio_data)
    data = _elevenlabs_request_data()
    total = len(audio_int16)
    max_samples = _chunk_max_samples(sample_rate)

    if total <= max_samples:
        result = _transcribe_elevenlabs_audio(audio_int16, data, api_key, sample_rate, timeout=120)
        text = _elevenlabs_result_text(result)
    else:
        num_chunks = (total + max_samples - 1) // max_samples
        _debug(f"  Long audio ({total/sample_rate:.0f}s), {num_chunks} chunks")
        parts = []
        for i in range(num_chunks):
            start = i * max_samples
            end = min(start + max_samples, total)
            result = _transcribe_elevenlabs_audio(
                audio_int16[start:end],
                data,
                api_key,
                sample_rate,
                timeout=180,
            )
            chunk_text = _elevenlabs_result_text(result)
            if chunk_text:
                parts.append(chunk_text)
        text = ' '.join(p.strip() for p in parts if p)

    _debug(f"transcribe_elevenlabs() FINISHED, {len(text)} chars")
    return text


def transcribe_groq_segments(audio_data: np.ndarray, label: str = "Speaker", sample_rate: int = 16000) -> list[dict]:
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

    sample_rate = int(sample_rate or 16000)
    audio_int16 = _ensure_int16(audio_data)
    data = _groq_request_data(response_format="verbose_json")
    total = len(audio_int16)
    max_samples = _chunk_max_samples(sample_rate)

    segments_out: list[dict] = []
    chunk_count = (total + max_samples - 1) // max_samples if total else 0

    for i in range(chunk_count):
        start_sample = i * max_samples
        end_sample = min(start_sample + max_samples, total)
        chunk_offset_sec = start_sample / sample_rate
        buf = _audio_to_wav_bytes(
            audio_int16[start_sample:end_sample],
            sample_rate=sample_rate,
            trailing_silence_sec=GROQ_TRAILING_SILENCE_SEC if end_sample == total else 0.0,
        )
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


def transcribe_elevenlabs_segments(audio_data: np.ndarray, label: str = "Speaker", sample_rate: int = 16000) -> list[dict]:
    """Meeting-style transcription with word-derived timestamps via ElevenLabs."""
    _debug(f"transcribe_elevenlabs_segments() STARTED label={label}")
    api_key = _api_key_from_env(*ELEVENLABS_API_KEY_NAMES)
    if not api_key:
        _debug("  ERROR: ELEVENLABS_API_KEY not set")
        return []

    sample_rate = int(sample_rate or 16000)
    audio_int16 = _ensure_int16(audio_data)
    data = _elevenlabs_request_data()
    total = len(audio_int16)
    max_samples = _chunk_max_samples(sample_rate)
    chunk_count = (total + max_samples - 1) // max_samples if total else 0

    segments_out: list[dict] = []
    for i in range(chunk_count):
        start_sample = i * max_samples
        end_sample = min(start_sample + max_samples, total)
        chunk_offset_sec = start_sample / sample_rate
        result = _transcribe_elevenlabs_audio(
            audio_int16[start_sample:end_sample],
            data,
            api_key,
            sample_rate,
            timeout=180,
        )
        if not result:
            continue

        from utils import TextProcessor

        segments = _segments_from_elevenlabs_words(result, label=label, offset_sec=chunk_offset_sec)
        for seg in segments:
            text = TextProcessor.remove_filler_words(seg.get("text", "").strip())
            if not text:
                continue
            seg = dict(seg)
            seg["text"] = text
            segments_out.append(seg)
        _debug(f"  Chunk {i+1}/{chunk_count}: {len(segments)} segments")

    _debug(f"transcribe_elevenlabs_segments() FINISHED, {len(segments_out)} total segments")
    return segments_out


def transcribe_segments(audio_data: np.ndarray, label: str = "Speaker", sample_rate: int = 16000) -> list[dict]:
    """Meeting path routed by configured transcription provider."""
    provider = _transcription_provider()
    if provider == "groq":
        return transcribe_groq_segments(audio_data, label=label, sample_rate=sample_rate)
    if provider == "elevenlabs":
        return transcribe_elevenlabs_segments(audio_data, label=label, sample_rate=sample_rate)

    _debug(f"  Unknown transcription provider: {provider}")
    ConfigManager.console_print(f"Error: unknown transcription provider '{provider}'")
    return []


# ---------- post-processing & cleanup ----------

def post_process_transcription(transcription: str) -> str:
    """Apply regex post-processing (filler words, hallucination tail strip)."""
    from utils import TextProcessor
    return TextProcessor.process(transcription, add_trailing_space=True)


def remove_filler_words(text: str) -> str:
    """Compatibility wrapper for tests and callers that used the old module API."""
    from utils import TextProcessor
    return TextProcessor.remove_filler_words(text)


def ensure_ending_punctuation(text: str) -> str:
    """Compatibility wrapper for tests and callers that used the old module API."""
    from utils import TextProcessor
    return TextProcessor.ensure_ending_punctuation(text)


def _normalise_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def _cleanup_preserves_tail(original: str, cleaned: str, tail_words: int = 12) -> bool:
    """Return False when cleanup appears to have dropped the end of the snippet."""
    original_tail = _normalise_words(original)[-tail_words:]
    if len(original_tail) < 4:
        return True

    cleaned_words = _normalise_words(cleaned)
    if not cleaned_words:
        return False
    cleaned_words = cleaned_words[-max(24, tail_words * 2):]
    required_suffix = original_tail[-min(4, len(original_tail)):]
    if not _is_subsequence(required_suffix, cleaned_words):
        return False

    matched = 0
    search_from = 0
    for word in original_tail:
        try:
            found_at = cleaned_words.index(word, search_from)
        except ValueError:
            continue
        matched += 1
        search_from = found_at + 1

    required_matches = max(3, min(6, len(original_tail) // 2))
    return matched >= required_matches


def _provider_pin_for_model(model_id: str):
    """OpenRouter provider pin per model. None = let OpenRouter pick."""
    pins = {
        "google/gemini-3.5-flash":               ["Google AI Studio"],
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

        model = ConfigManager.get_config_value('post_processing', 'ai_cleanup_model') or 'google/gemini-3.5-flash'
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
        if not _cleanup_preserves_tail(text, cleaned):
            _debug("  cleanup dropped tail, using pre-cleanup text")
            return text
        if not cleaned.endswith(' '):
            cleaned += ' '
        return cleaned

    except Exception as e:
        _debug(f"  cleanup error: {e}")
        return text


# ---------- top-level snippet entry point ----------

def transcribe(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Snippet path: provider STT -> regex post-process -> optional cleanup -> save -> return."""
    _debug("transcribe() STARTED")
    if audio_data is None:
        return ''

    sample_rate = int(sample_rate or 16000)
    audio_duration_sec = len(audio_data) / sample_rate
    _debug(f"  Duration: {audio_duration_sec:.1f}s")

    provider = _transcription_provider()
    _debug(f"  Provider: {provider}")
    if provider == "groq":
        raw = transcribe_groq(audio_data, sample_rate=sample_rate)
    elif provider == "elevenlabs":
        raw = transcribe_elevenlabs(audio_data, sample_rate=sample_rate)
    else:
        _debug(f"  Unknown transcription provider: {provider}")
        ConfigManager.console_print(f"Error: unknown transcription provider '{provider}'")
        raw = ''
    _debug(f"  Raw: {len(raw)} chars")

    post_processed = post_process_transcription(raw)
    result = post_processed
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
    save_transcription_debug(raw, post_processed, result, audio_duration_sec)
    _debug("transcribe() FINISHED")
    return result
