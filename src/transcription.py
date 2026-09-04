"""Provider-backed transcription for Koe snippets and Scribe meetings.

The module has two public entry points:
- ``transcribe`` returns paste-ready text for a hotkey snippet.
- ``transcribe_file_segments`` returns timestamped, diarized meeting segments.

ElevenLabs Scribe v2 remains the default. Deepgram Nova-3 and Mistral Voxtral
implement the same two public contracts. Snippets may be split into ten-minute
requests. Scribe sends its full mixed meeting file once so diarized speaker IDs
stay coherent.
"""

import io
import os
import re
import threading
import time
import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from dotenv import load_dotenv

from paths import default_snippets_dir, env_path, logs_dir
from providers import deepgram, mistral
from utils import ConfigManager

_DEBUG_LOG = logs_dir() / "debug.log"
_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)

MAX_SNIPPETS = 5
MAX_DEBUG_SNIPPETS = 5
CHUNK_SECONDS = 10 * 60
GROUP_TRANSCRIPTION_TIMEOUT = 15 * 60
ELEVENLABS_UPLOAD_MAX_ATTEMPTS = 3
ELEVENLABS_UPLOAD_RETRY_DELAYS = (2.0, 5.0)
MIN_MEETING_UPLOAD_BYTES_PER_SECOND = 12_000
MEETING_UPLOAD_TIMEOUT_MARGIN = 5 * 60
MAX_ELEVENLABS_FILE_BYTES = 3_000_000_000
MAX_ELEVENLABS_DURATION_SECONDS = 10 * 60 * 60
MIN_ELEVENLABS_DURATION_SECONDS = 0.1
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_TRANSCRIPT_URL = f"{ELEVENLABS_URL}/transcripts"
ELEVENLABS_API_KEY_NAMES = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "XI_API_KEY")
DEEPGRAM_API_KEY_NAMES = ("DEEPGRAM_API_KEY",)
MISTRAL_API_KEY_NAMES = ("MISTRAL_API_KEY",)
SUPPORTED_TRANSCRIPTION_PROVIDERS = ("elevenlabs", "deepgram", "mistral")
ELEVENLABS_DELETE_MAX_ATTEMPTS = 3
ELEVENLABS_DELETE_RETRY_DELAYS = (1.0, 3.0)
ELEVENLABS_DELETE_TIMEOUT = 15.0
MAX_CORRECTION_LENGTH = 100
MAX_DEEPGRAM_FILE_BYTES = 2_000_000_000
MAX_MISTRAL_FILE_BYTES = 500_000_000
MAX_MISTRAL_DURATION_SECONDS = 60 * 60


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
            "## Provider Response\n\n"
            f"{(raw or '').strip()}\n\n"
            "## Final\n\n"
            f"{(final or '').strip()}\n"
        )
        (debug_dir / "snippet_debug_1.md").write_text(content, encoding="utf-8")
    except Exception as exc:
        _debug(f"save_transcription_debug error: {exc}")


# ---------- provider selection ----------


def _load_env() -> None:
    load_dotenv(env_path(), override=True)


def _api_key_from_env(*names: str) -> str:
    _load_env()
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def transcription_provider() -> str:
    """Return the validated configured provider, defaulting to ElevenLabs."""
    provider = str(
        ConfigManager.get_config_value("transcription_options", "provider")
        or "elevenlabs"
    ).strip().casefold()
    if provider not in SUPPORTED_TRANSCRIPTION_PROVIDERS:
        raise ValueError(f"Unsupported transcription provider: {provider}")
    return provider


def _provider_language() -> str | None:
    language = ConfigManager.get_config_value("model_options", "common", "language")
    return str(language).strip() if language else None


def _provider_api_key(provider: str) -> str:
    names = {
        "elevenlabs": ELEVENLABS_API_KEY_NAMES,
        "deepgram": DEEPGRAM_API_KEY_NAMES,
        "mistral": MISTRAL_API_KEY_NAMES,
    }[provider]
    return _api_key_from_env(*names)


# ---------- ElevenLabs request path ----------


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
    rms = float(np.sqrt(np.mean(audio_float**2)))
    if rms <= 1e-3 or rms >= target_rms:
        return audio_int16.astype(np.int16, copy=False)

    peak = float(np.max(np.abs(audio_float)))
    clip_limited_gain = (32767.0 / peak) if peak > 0 else max_gain
    gain = min(target_rms / rms, max_gain, clip_limited_gain)
    if gain <= 1.0:
        return audio_int16.astype(np.int16, copy=False)

    _debug(f"Normalizing quiet audio: rms={rms:.1f}, gain={gain:.2f}x")
    return np.clip(audio_float * gain, -32768, 32767).astype(np.int16)


def _audio_to_wav_bytes(
    audio_int16: np.ndarray, sample_rate: int = 16000
) -> io.BytesIO:
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
    started = time.perf_counter()
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
        result = response.json()
    except ValueError:
        return None, "ElevenLabs returned an invalid JSON response"
    _debug(
        f"ElevenLabs snippet request completed in {time.perf_counter() - started:.2f}s "
        "(upload + recognition + response decoding; excludes deletion)"
    )
    _delete_elevenlabs_transcript_in_background(result, api_key)
    return result, None


def _delete_elevenlabs_transcript_in_background(result: object, api_key: str) -> None:
    """Start cleanup without holding up snippet delivery or retaining its text."""
    transcription_id = (
        str(result.get("transcription_id") or "").strip()
        if isinstance(result, dict)
        else ""
    )
    if not transcription_id:
        _debug("ElevenLabs response had no deletable transcription_id")
        return

    def cleanup() -> None:
        started = time.perf_counter()
        deleted = False
        try:
            deleted = _delete_elevenlabs_transcript(
                {"transcription_id": transcription_id}, api_key
            )
        except Exception as exc:
            _debug(f"ElevenLabs background deletion failed: {type(exc).__name__}")
        finally:
            _debug(
                f"ElevenLabs background deletion finished in "
                f"{time.perf_counter() - started:.2f}s (success={deleted})"
            )

    try:
        # Let bounded cleanup/retries finish on normal exit instead of abandoning
        # remote data. This thread never blocks text delivery or the next snippet.
        threading.Thread(
            target=cleanup, name="koe-transcript-cleanup", daemon=False
        ).start()
    except RuntimeError:
        _debug("ElevenLabs background deletion could not start; transcript retained remotely")


def _delete_elevenlabs_transcript(result: object, api_key: str) -> bool:
    """Best-effort removal of a completed STT input/output from ElevenLabs.

    The deletion runs only after a successful response has been decoded locally.
    A missing ID or failed deletion must not discard the transcript already
    received by Koe, but the failure is recorded in the local diagnostic log.
    """
    if not isinstance(result, dict):
        _debug("ElevenLabs response had no deletable transcription_id")
        return False
    transcription_id = str(result.get("transcription_id") or "").strip()
    if not transcription_id:
        _debug("ElevenLabs response had no deletable transcription_id")
        return False

    delete_url = f"{ELEVENLABS_TRANSCRIPT_URL}/{quote(transcription_id, safe='')}"
    for attempt in range(1, ELEVENLABS_DELETE_MAX_ATTEMPTS + 1):
        try:
            response = requests.delete(
                delete_url,
                headers={"xi-api-key": api_key},
                timeout=ELEVENLABS_DELETE_TIMEOUT,
            )
        except requests.RequestException as exc:
            if attempt >= ELEVENLABS_DELETE_MAX_ATTEMPTS:
                _debug(
                    "ElevenLabs transcript deletion failed after "
                    f"{attempt} attempts: {exc}"
                )
                return False
        else:
            if 200 <= response.status_code < 300 or response.status_code == 404:
                _debug("ElevenLabs transcript deleted after successful receipt")
                return True
            if response.status_code != 429 and response.status_code < 500:
                _debug(
                    "ElevenLabs transcript deletion failed: "
                    f"HTTP {response.status_code}"
                )
                return False
            if attempt >= ELEVENLABS_DELETE_MAX_ATTEMPTS:
                _debug(
                    "ElevenLabs transcript deletion failed after "
                    f"{attempt} attempts: HTTP {response.status_code}"
                )
                return False

        delay = ELEVENLABS_DELETE_RETRY_DELAYS[attempt - 1]
        _debug(
            "Retrying ElevenLabs transcript deletion "
            f"({attempt}/{ELEVENLABS_DELETE_MAX_ATTEMPTS}) in {delay:.0f}s"
        )
        time.sleep(delay)

    return False


def _elevenlabs_post_file(
    file_path: Path,
    data: list[tuple[str, str]],
    api_key: str,
    timeout: float,
) -> tuple[dict | None, str | None]:
    """Stream a meeting file, retrying only failures safe to resubmit.

    A fresh file handle is opened for every attempt so a retry always starts at
    byte zero. Connection failures (including urllib3's wrapped socket write
    timeout) are retryable. A read timeout is deliberately not retried because
    ElevenLabs may already have received and processed the complete recording.
    """
    try:
        file_size = file_path.stat().st_size
    except OSError as exc:
        return None, f"ElevenLabs request error: {exc}"

    upload_timeout = max(
        float(timeout),
        file_size / MIN_MEETING_UPLOAD_BYTES_PER_SECOND + MEETING_UPLOAD_TIMEOUT_MARGIN,
    )
    request_timeout = (upload_timeout, float(timeout))
    if upload_timeout > float(timeout):
        _debug(
            "Extended ElevenLabs upload timeout to "
            f"{upload_timeout / 60:.1f} minutes for {file_size} bytes"
        )

    response = None
    for attempt in range(1, ELEVENLABS_UPLOAD_MAX_ATTEMPTS + 1):
        try:
            with file_path.open("rb") as audio_file:
                response = requests.post(
                    ELEVENLABS_URL,
                    headers={"xi-api-key": api_key},
                    files={"file": (file_path.name, audio_file, "audio/wav")},
                    data=data,
                    # urllib3 uses the connect timeout while writing the request
                    # body, then switches to the read timeout for the response.
                    timeout=request_timeout,
                )
            break
        except requests.RequestException as exc:
            retryable = isinstance(
                exc,
                (requests.ConnectionError, requests.ConnectTimeout),
            ) and not isinstance(exc, requests.ReadTimeout)
            if not retryable or attempt >= ELEVENLABS_UPLOAD_MAX_ATTEMPTS:
                suffix = (
                    f" after {attempt} attempts" if retryable and attempt > 1 else ""
                )
                return None, f"ElevenLabs request error{suffix}: {exc}"

            delay = ELEVENLABS_UPLOAD_RETRY_DELAYS[attempt - 1]
            _debug(
                "Transient ElevenLabs upload failure "
                f"({attempt}/{ELEVENLABS_UPLOAD_MAX_ATTEMPTS}): {exc}; "
                f"retrying in {delay:.0f}s"
            )
            time.sleep(delay)
        except OSError as exc:
            return None, f"ElevenLabs request error: {exc}"

    if response is None:
        return None, "ElevenLabs request error: no response"

    if response.status_code != 200:
        return None, f"ElevenLabs HTTP {response.status_code}: {response.text[:300]}"
    try:
        result = response.json()
    except ValueError:
        return None, "ElevenLabs returned an invalid JSON response"
    _delete_elevenlabs_transcript(result, api_key)
    return result, None


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


def load_transcript_corrections() -> dict[str, str]:
    """Load private exact-token corrections from Koe's existing config."""
    raw_corrections = ConfigManager.get_config_section(
        "transcription_options", "corrections"
    )
    if not isinstance(raw_corrections, dict):
        _debug("Ignoring transcript corrections because they are not a mapping")
        return {}

    corrections: dict[str, str] = {}
    for raw_source, raw_target in raw_corrections.items():
        if not isinstance(raw_source, str) or not isinstance(raw_target, str):
            continue
        source = re.sub(r"\s+", " ", raw_source.strip())
        target = re.sub(r"\s+", " ", raw_target.strip())
        if not source or not target:
            continue
        if len(source) > MAX_CORRECTION_LENGTH or len(target) > MAX_CORRECTION_LENGTH:
            continue
        if any(character in source or character in target for character in "\r\n\t"):
            continue
        corrections[source.casefold()] = target
    return corrections


def apply_transcript_corrections(
    text: str,
    corrections: dict[str, str] | None = None,
) -> str:
    """Apply configured whole-token substitutions while preserving case."""
    if not text:
        return text
    active_corrections = (
        load_transcript_corrections() if corrections is None else corrections
    )
    if not active_corrections:
        return text

    pattern = re.compile(
        r"(?<!\w)("
        + "|".join(map(re.escape, sorted(active_corrections, key=len, reverse=True)))
        + r")(?!\w)",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match) -> str:
        original = match.group(0)
        corrected = active_corrections[original.casefold()]
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
        return (
            [{"start": offset_sec, "end": offset_sec, "text": text, "label": label}]
            if text
            else []
        )

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
            segments.append(
                {
                    "start": current_start + offset_sec,
                    "end": current_end + offset_sec,
                    "text": text,
                    "label": current_label or label,
                }
            )
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
    """Send one file-backed Scribe request and preserve speaker identity."""
    file_path = Path(file_path)
    provider = transcription_provider()
    file_size = file_path.stat().st_size
    duration_seconds = _wav_duration_seconds(file_path)
    if duration_seconds < MIN_ELEVENLABS_DURATION_SECONDS:
        _debug(
            "Skipping file-backed transcription because the stream contains "
            f"only {duration_seconds:.3f}s of audio"
        )
        return []
    if provider == "elevenlabs":
        if file_size > MAX_ELEVENLABS_FILE_BYTES:
            raise ValueError("Meeting audio exceeds ElevenLabs' 3 GB file limit.")
        if duration_seconds > MAX_ELEVENLABS_DURATION_SECONDS:
            raise ValueError("Meeting audio exceeds ElevenLabs' 10-hour duration limit.")
    elif provider == "deepgram":
        if file_size > MAX_DEEPGRAM_FILE_BYTES:
            raise ValueError("Meeting audio exceeds Deepgram's 2 GB file limit.")
    elif provider == "mistral":
        if file_size > MAX_MISTRAL_FILE_BYTES:
            raise ValueError("Meeting audio exceeds Mistral's 500 MB file limit.")
        if duration_seconds > MAX_MISTRAL_DURATION_SECONDS:
            raise ValueError("Meeting audio exceeds Mistral's 60-minute limit.")

    api_key = _provider_api_key(provider)
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY not set")

    if provider == "elevenlabs":
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
    elif provider == "deepgram":
        result, error = deepgram.transcribe_file(
            file_path,
            api_key,
            language=_provider_language(),
            diarize=diarize,
            timeout=timeout,
        )
    else:
        result, error = mistral.transcribe_file(
            file_path,
            api_key,
            language=_provider_language(),
            diarize=diarize,
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
            audio_int16[start : start + max_samples],
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


def transcribe_deepgram(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Return flat Nova-3 text for snippet-style audio."""
    api_key = _provider_api_key("deepgram")
    if not api_key:
        _debug("DEEPGRAM_API_KEY not set")
        ConfigManager.console_print("Error: DEEPGRAM_API_KEY not set in .env")
        return ""
    sample_rate = int(sample_rate or 16000)
    audio_int16 = _ensure_int16(audio_data)
    max_samples = _chunk_max_samples(sample_rate)
    parts: list[str] = []
    for start in range(0, len(audio_int16), max_samples):
        buffer = _audio_to_wav_bytes(
            _normalize_quiet_audio(audio_int16[start : start + max_samples]),
            sample_rate=sample_rate,
        )
        result, error = deepgram.transcribe_stream(
            buffer,
            api_key,
            language=_provider_language(),
            diarize=False,
            timeout=180,
        )
        if error:
            _debug(error)
            continue
        text = _elevenlabs_result_text(result or {})
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def transcribe_mistral(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Return flat Voxtral text for snippet-style audio."""
    api_key = _provider_api_key("mistral")
    if not api_key:
        _debug("MISTRAL_API_KEY not set")
        ConfigManager.console_print("Error: MISTRAL_API_KEY not set in .env")
        return ""
    sample_rate = int(sample_rate or 16000)
    audio_int16 = _ensure_int16(audio_data)
    max_samples = _chunk_max_samples(sample_rate)
    parts: list[str] = []
    for start in range(0, len(audio_int16), max_samples):
        buffer = _audio_to_wav_bytes(
            _normalize_quiet_audio(audio_int16[start : start + max_samples]),
            sample_rate=sample_rate,
        )
        result, error = mistral.transcribe_stream(
            buffer,
            "snippet.wav",
            api_key,
            language=_provider_language(),
            diarize=False,
            timeout=180,
        )
        if error:
            _debug(error)
            continue
        text = _elevenlabs_result_text(result or {})
        if text:
            parts.append(text)
    return " ".join(parts).strip()


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
    provider = transcription_provider()
    transcribers = {
        "elevenlabs": transcribe_elevenlabs,
        "deepgram": transcribe_deepgram,
        "mistral": transcribe_mistral,
    }
    raw = transcribers[provider](audio_data, sample_rate=sample_rate)
    final = post_process_transcription(raw)
    save_rolling_transcription(final)
    save_transcription_debug(raw, final, duration_sec)
    return final
