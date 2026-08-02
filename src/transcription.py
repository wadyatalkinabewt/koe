"""ElevenLabs Scribe v2 transcription for Koe snippets and Scribe meetings.

The module has two public entry points:
- ``transcribe`` returns paste-ready text for a hotkey snippet.
- ``transcribe_file_segments`` returns timestamped, diarized meeting segments.

All speech-to-text requests use ElevenLabs Scribe v2 with no-verbatim mode.
Snippets may be split into ten-minute requests. Scribe streams its full mixed
meeting file once so diarized speaker IDs stay coherent.
"""

import io
import os
import re
import wave
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from dotenv import load_dotenv

from paths import default_snippets_dir, env_path, logs_dir
from utils import ConfigManager


_DEBUG_LOG = logs_dir() / "debug.log"
_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)

MAX_SNIPPETS = 5
MAX_DEBUG_SNIPPETS = 5
CHUNK_SECONDS = 10 * 60
GROUP_TRANSCRIPTION_TIMEOUT = 15 * 60
MAX_ELEVENLABS_FILE_BYTES = 5_000_000_000
MAX_ELEVENLABS_DURATION_SECONDS = 10 * 60 * 60
MIN_ELEVENLABS_DURATION_SECONDS = 0.1
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_API_KEY_NAMES = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "XI_API_KEY")

# Exact-token corrections for stable Scribe substitutions that vocabulary
# keyterms do not prevent. Keep this list deliberately small and evidence-led.
TRANSCRIPT_CORRECTIONS = {
    "groq": "Grok",
    "Taylor": "Taylor",
}


def _debug(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] [transcription] {message}\n")
    except Exception:
        pass


# ---------- snippet storage ----------


def _get_snippets_dir() -> Path:
    configured = ConfigManager.get_config_value("misc", "snippets_folder")
    snippets_dir = Path(configured) if configured else default_snippets_dir()
    snippets_dir.mkdir(parents=True, exist_ok=True)
    return snippets_dir


def save_rolling_transcription(text: str) -> None:
    """Save the newest snippet as snippet_1.md and retain five Markdown files."""
    if not text or not text.strip():
        return
    try:
        snippets_dir = _get_snippets_dir()
        oldest = snippets_dir / f"snippet_{MAX_SNIPPETS}.md"
        if oldest.exists():
            oldest.unlink()
        for index in range(MAX_SNIPPETS - 1, 0, -1):
            old = snippets_dir / f"snippet_{index}.md"
            new = snippets_dir / f"snippet_{index + 1}.md"
            if old.exists():
                old.rename(new)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"# Snippet\n\n**Time:** {timestamp}\n\n---\n\n{text.strip()}\n"
        (snippets_dir / "snippet_1.md").write_text(content, encoding="utf-8")
    except Exception as exc:
        _debug(f"save_rolling_transcription error: {exc}")


def save_transcription_debug(raw: str, final: str, duration_sec: float) -> None:
    """Keep five local text snapshots for transcription diagnostics."""
    try:
        debug_dir = logs_dir() / "transcription_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        oldest = debug_dir / f"snippet_debug_{MAX_DEBUG_SNIPPETS}.md"
        if oldest.exists():
            oldest.unlink()
        for index in range(MAX_DEBUG_SNIPPETS - 1, 0, -1):
            old = debug_dir / f"snippet_debug_{index}.md"
            new = debug_dir / f"snippet_debug_{index + 1}.md"
            if old.exists():
                old.rename(new)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = (
            f"# Snippet Debug\n\n"
            f"**Time:** {timestamp}\n"
            f"**Duration:** {duration_sec:.2f}s\n\n"
            "## ElevenLabs Response\n\n"
            f"{(raw or '').strip()}\n\n"
            "## Final\n\n"
            f"{(final or '').strip()}\n"
        )
        (debug_dir / "snippet_debug_1.md").write_text(content, encoding="utf-8")
    except Exception as exc:
        _debug(f"save_transcription_debug error: {exc}")


# ---------- ElevenLabs request path ----------


def _load_env() -> None:
    load_dotenv(env_path(), override=True)


def _api_key_from_env(*names: str) -> str:
    _load_env()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _initial_prompt_keyterms() -> list[str]:
    """Convert the configured vocabulary hint into ElevenLabs keyterms."""
    model_options = ConfigManager.get_config_section("model_options") or {}
    common = model_options.get("common", {}) or {}
    vocabulary = common.get("initial_prompt") or ""
    if not vocabulary:
        return []

    keyterms: list[str] = []
    seen: set[str] = set()
    for raw_term in re.split(r"[,;\n]", str(vocabulary)):
        term = re.sub(r"\s+", " ", raw_term.strip().strip("."))
        if not term or re.search(r"[<>{}\[\]\\]", term):
            continue
        if len(term.split()) > 5 or len(term) >= 50:
            continue
        normalized = term.lower()
        if normalized in seen:
            continue
        keyterms.append(term)
        seen.add(normalized)
        if len(keyterms) >= 1000:
            break
    return keyterms


def _chunk_max_samples(sample_rate: int) -> int:
    return CHUNK_SECONDS * int(sample_rate or 16000)


def _ensure_int16(audio_data: np.ndarray) -> np.ndarray:
    """Convert arbitrary numeric audio samples to mono int16 PCM."""
    audio = np.asarray(audio_data).reshape(-1)
    if np.issubdtype(audio.dtype, np.floating):
        return np.clip(audio * 32768, -32768, 32767).astype(np.int16)
    return audio.astype(np.int16)


def _normalize_quiet_audio(
    audio_int16: np.ndarray,
    target_rms: float = 3000.0,
    max_gain: float = 8.0,
) -> np.ndarray:
    """Raise very quiet captures toward a stable level without clipping."""
    if audio_int16.size == 0:
        return audio_int16.astype(np.int16, copy=False)

    audio_float = audio_int16.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio_float ** 2)))
    if rms <= 1e-3 or rms >= target_rms:
        return audio_int16.astype(np.int16, copy=False)

    peak = float(np.max(np.abs(audio_float)))
    clip_limited_gain = (32767.0 / peak) if peak > 0 else max_gain
    gain = min(target_rms / rms, max_gain, clip_limited_gain)
    if gain <= 1.0:
        return audio_int16.astype(np.int16, copy=False)

    _debug(f"Normalizing quiet audio: rms={rms:.1f}, gain={gain:.2f}x")
    return np.clip(audio_float * gain, -32768, 32767).astype(np.int16)


def _audio_to_wav_bytes(audio_int16: np.ndarray, sample_rate: int = 16000) -> io.BytesIO:
    """Pack mono int16 PCM samples into an in-memory WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate or 16000))
        wav_file.writeframes(audio_int16.tobytes())
    buffer.seek(0)
    return buffer


def _elevenlabs_request_data(
    *,
    diarize: bool = False,
    use_speaker_library: bool = False,
    num_speakers: int | None = None,
) -> list[tuple[str, str]]:
    """Build the shared Scribe v2 form fields for every Koe STT request."""
    if num_speakers is not None and not 1 <= int(num_speakers) <= 32:
        raise ValueError("num_speakers must be between 1 and 32")
    model_options = ConfigManager.get_config_section("model_options") or {}
    common = model_options.get("common", {}) or {}
    elevenlabs = model_options.get("elevenlabs", {}) or {}

    data = [
        ("model_id", "scribe_v2"),
        ("tag_audio_events", "false"),
        ("timestamps_granularity", "word"),
        ("file_format", "other"),
        ("no_verbatim", "true"),
        # Keep Scribe billing tied to the duration of one mono timeline. This is
        # deliberately explicit rather than relying on the API default.
        ("use_multi_channel", "false"),
    ]
    language = common.get("language")
    if language:
        data.append(("language_code", str(language)))
    if elevenlabs.get("keyterms_enabled", False):
        data.extend(("keyterms", term) for term in _initial_prompt_keyterms())
    if diarize:
        data.append(("diarize", "true"))
        if num_speakers is not None:
            data.append(("num_speakers", str(int(num_speakers))))
        if use_speaker_library:
            data.append(("use_speaker_library", "true"))
    return data


def _elevenlabs_post(
    buffer: io.BytesIO,
    data: list[tuple[str, str]],
    api_key: str,
    timeout: float,
) -> tuple[dict | None, str | None]:
    """Submit one synchronous ElevenLabs speech-to-text request."""
    try:
        response = requests.post(
            ELEVENLABS_URL,
            headers={"xi-api-key": api_key},
            files={"file": ("audio.wav", buffer, "audio/wav")},
            data=data,
            timeout=timeout,
        )
    except requests.Timeout:
        return None, "ElevenLabs timeout"
    except requests.RequestException as exc:
        return None, f"ElevenLabs request error: {exc}"

    if response.status_code != 200:
        return None, f"ElevenLabs HTTP {response.status_code}: {response.text[:300]}"
    try:
        return response.json(), None
    except ValueError:
        return None, "ElevenLabs returned an invalid JSON response"


def _elevenlabs_post_file(
    file_path: Path,
    data: list[tuple[str, str]],
    api_key: str,
    timeout: float,
) -> tuple[dict | None, str | None]:
    """Stream an existing audio file to ElevenLabs without duplicating it in memory."""
    try:
        with file_path.open("rb") as audio_file:
            response = requests.post(
                ELEVENLABS_URL,
                headers={"xi-api-key": api_key},
                files={"file": (file_path.name, audio_file, "audio/wav")},
                data=data,
                timeout=timeout,
            )
    except requests.Timeout:
        return None, "ElevenLabs timeout"
    except (OSError, requests.RequestException) as exc:
        return None, f"ElevenLabs request error: {exc}"

    if response.status_code != 200:
        return None, f"ElevenLabs HTTP {response.status_code}: {response.text[:300]}"
    try:
        return response.json(), None
    except ValueError:
        return None, "ElevenLabs returned an invalid JSON response"


def _transcribe_elevenlabs_audio(
    audio_int16: np.ndarray,
    data: list[tuple[str, str]],
    api_key: str,
    sample_rate: int,
    timeout: float,
) -> dict:
    normalized = _normalize_quiet_audio(audio_int16)
    buffer = _audio_to_wav_bytes(normalized, sample_rate=sample_rate)
    result, error = _elevenlabs_post(buffer, data, api_key, timeout=timeout)
    if error:
        _debug(error)
        return {}
    return result if isinstance(result, dict) else {}


def _elevenlabs_result_text(result: dict) -> str:
    return str(result.get("text") or "").strip() if isinstance(result, dict) else ""


def apply_transcript_corrections(text: str) -> str:
    """Correct known whole-token Scribe substitutions while preserving case."""
    if not text:
        return text

    pattern = re.compile(
        r"(?<!\w)(" + "|".join(map(re.escape, TRANSCRIPT_CORRECTIONS)) + r")(?!\w)",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match) -> str:
        original = match.group(0)
        corrected = TRANSCRIPT_CORRECTIONS[original.lower()]
        if original.isupper():
            return corrected.upper()
        if original.islower():
            return corrected.lower()
        return corrected

    return pattern.sub(replace, text)


def _segments_from_elevenlabs_words(
    result: dict,
    label: str,
    offset_sec: float = 0.0,
    use_speaker_labels: bool = False,
    label_resolver: Callable[[float, float], str] | None = None,
) -> list[dict]:
    words = result.get("words") if isinstance(result, dict) else None
    if not isinstance(words, list):
        text = apply_transcript_corrections(_elevenlabs_result_text(result))
        return [{"start": offset_sec, "end": offset_sec, "text": text, "label": label}] if text else []

    segments: list[dict] = []
    current_words: list[str] = []
    current_start: float | None = None
    current_end = 0.0
    last_end: float | None = None
    current_label: str | None = None

    def flush() -> None:
        nonlocal current_words, current_start, current_end, current_label
        text = apply_transcript_corrections(" ".join(current_words).strip())
        if text and current_start is not None:
            segments.append({
                "start": current_start + offset_sec,
                "end": current_end + offset_sec,
                "text": text,
                "label": current_label or label,
            })
        current_words = []
        current_start = None
        current_end = 0.0
        current_label = None

    for word in words:
        if not isinstance(word, dict) or word.get("type") not in (None, "word"):
            continue
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        if use_speaker_labels:
            word_label = _display_speaker_label(word.get("speaker_id"), fallback=label)
        elif label_resolver is not None:
            word_label = str(
                label_resolver(start + offset_sec, end + offset_sec) or label
            )
        else:
            word_label = label
        if current_words and current_label != word_label:
            flush()
        if current_words and last_end is not None and start - last_end > 1.2:
            flush()
        if current_start is None:
            current_start = start
            current_label = word_label
        current_words.append(text)
        current_end = end
        last_end = end
        if text.endswith((".", "?", "!")) or len(current_words) >= 28:
            flush()

    flush()
    return segments


def _display_speaker_label(speaker_id, fallback: str = "Speaker") -> str:
    """Render generic Scribe IDs readably while preserving library identifiers."""
    raw = str(speaker_id or "").strip()
    if not raw:
        return fallback
    match = re.fullmatch(r"speaker_(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return f"Speaker {int(match.group(1)) + 1}"
    return raw


def _wav_duration_seconds(file_path: Path) -> float:
    try:
        with wave.open(str(file_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            return wav_file.getnframes() / frame_rate if frame_rate else 0.0
    except (OSError, wave.Error) as exc:
        raise ValueError(f"Could not inspect meeting audio: {exc}") from exc


def transcribe_file_segments(
    file_path: Path,
    label: str = "Speaker",
    *,
    diarize: bool = False,
    use_speaker_library: bool = False,
    num_speakers: int | None = None,
    label_resolver: Callable[[float, float], str] | None = None,
    timeout: float = GROUP_TRANSCRIPTION_TIMEOUT,
) -> list[dict]:
    """Stream one file-backed Scribe request, preserving diarized speaker identity."""
    file_path = Path(file_path)
    if file_path.stat().st_size > MAX_ELEVENLABS_FILE_BYTES:
        raise ValueError("Meeting audio exceeds ElevenLabs' 5 GB file limit.")
    duration_seconds = _wav_duration_seconds(file_path)
    if duration_seconds < MIN_ELEVENLABS_DURATION_SECONDS:
        _debug(
            "Skipping file-backed transcription because the stream contains "
            f"only {duration_seconds:.3f}s of audio"
        )
        return []
    if duration_seconds > MAX_ELEVENLABS_DURATION_SECONDS:
        raise ValueError("Meeting audio exceeds ElevenLabs' 10-hour duration limit.")

    api_key = _api_key_from_env(*ELEVENLABS_API_KEY_NAMES)
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set")

    request_data = _elevenlabs_request_data(
        diarize=diarize,
        use_speaker_library=use_speaker_library,
        num_speakers=num_speakers,
    )
    result, error = _elevenlabs_post_file(
        file_path,
        request_data,
        api_key,
        timeout=timeout,
    )
    if error:
        _debug(error)
        raise RuntimeError(error)
    return _segments_from_elevenlabs_words(
        result or {},
        label=label,
        use_speaker_labels=diarize,
        label_resolver=label_resolver,
    )


def transcribe_elevenlabs(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Return flat Scribe v2 text for snippet-style audio."""
    api_key = _api_key_from_env(*ELEVENLABS_API_KEY_NAMES)
    if not api_key:
        _debug("ELEVENLABS_API_KEY not set")
        ConfigManager.console_print("Error: ELEVENLABS_API_KEY not set in .env")
        return ""

    sample_rate = int(sample_rate or 16000)
    audio_int16 = _ensure_int16(audio_data)
    request_data = _elevenlabs_request_data()
    max_samples = _chunk_max_samples(sample_rate)
    parts: list[str] = []
    for start in range(0, len(audio_int16), max_samples):
        result = _transcribe_elevenlabs_audio(
            audio_int16[start:start + max_samples],
            request_data,
            api_key,
            sample_rate,
            timeout=180,
        )
        text = _elevenlabs_result_text(result)
        if text:
            parts.append(text)
    final = " ".join(parts).strip()
    _debug(f"transcribe_elevenlabs finished: {len(final)} chars")
    return final


def post_process_transcription(transcription: str) -> str:
    """Apply approved corrections and clipboard-friendly snippet formatting."""
    from utils import TextProcessor

    corrected = apply_transcript_corrections(transcription)
    return TextProcessor.process(corrected, add_trailing_space=True)


def transcribe(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
) -> str:
    """Transcribe, format, save, and return one Koe hotkey snippet."""
    if audio_data is None:
        return ""
    sample_rate = int(sample_rate or 16000)
    duration_sec = len(audio_data) / sample_rate
    raw = transcribe_elevenlabs(audio_data, sample_rate=sample_rate)
    final = post_process_transcription(raw)
    save_rolling_transcription(final)
    save_transcription_debug(raw, final, duration_sec)
    return final
