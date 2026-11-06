"""Transcribe an existing audio file into a Koe transcript document.

Any container ffmpeg can decode is accepted, so a phone recorder's Ogg/Opus
output needs no manual conversion. When the capture guarantees one party per
stereo channel, speakers are attributed from channel energy rather than
acoustic diarization: the identity is ground truth, and the provider is not
asked to diarize. Files that are mono or dual-mono fall back to diarization.

This module is deliberately headless. The tray window imports it, and a
pipeline can call ``transcribe_recording()`` directly without Qt.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from meeting.capture import write_mono_wav
from meeting.transcript import render_transcript
from meeting.transcript_pdf import render_transcript_pdf
from paths import default_transcripts_dir
from transcription import transcribe_file_segments
from utils import ConfigManager

# Channel energy below this is treated as "nobody speaking"; it only has to sit
# above capture hiss, because the resolver compares two live channels.
_SILENCE_FLOOR = 20.0
# Below this left/right margin a segment is treated as cross-talk and keeps the
# previous speaker instead of flipping on noise. Measured on a real call: the
# median margin in overlapping windows is 5.1 dB, so 3 dB avoids coin flips.
_DEFAULT_MIN_MARGIN_DB = 3.0
# Correlated channels mean the recorder wrote the same signal twice; the split
# is not real, so acoustic diarization is the honest choice.
_SPLIT_CORRELATION_THRESHOLD = 0.9
_TARGET_RATE = 16000

PHONE_FIELD = re.compile(r"^\+?\d[\d\s\-()]{2,}$")


class DecodeError(RuntimeError):
    """Raised when the recording cannot be decoded for transcription."""


@dataclass(frozen=True)
class RecordingName:
    """What the recorder's filename convention tells us about a recording."""

    recorded_at: datetime | None
    direction: str | None
    other_name: str
    phone_number: str

    @property
    def recognised(self) -> bool:
        return self.recorded_at is not None or bool(self.direction)


@dataclass
class RecordingResult:
    """Everything a caller or the completion window needs."""

    output_dir: Path
    title: str
    participants: list[str]
    duration_seconds: float
    started_at: datetime
    transcript_md: Path | None = None
    transcript_pdf: Path | None = None
    summary_md: Path | None = None
    summary_pdf: Path | None = None
    segments: list[dict] = field(default_factory=list)
    attribution: str = "none"
    reused: bool = False
    warnings: list[str] = field(default_factory=list)


# ---------- ffmpeg ----------


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise DecodeError(
            f"{name} was not found on PATH. Koe needs ffmpeg to decode "
            "recordings; install it or add it to PATH."
        )
    return found


def probe_audio(path: Path | str) -> dict:
    """Return the first audio stream's real format, never guessed by suffix."""
    source = Path(path)
    completed = subprocess.run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(source),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise DecodeError(f"Could not read {source.name}: {detail or 'ffprobe failed'}")

    try:
        payload = json.loads(completed.stdout or b"{}")
    except ValueError as exc:
        raise DecodeError(f"Could not read {source.name}: {exc}") from exc

    streams = [
        stream
        for stream in (payload.get("streams") or [])
        if stream.get("codec_type") == "audio"
    ]
    if not streams:
        raise DecodeError(f"{source.name} has no audio stream.")

    stream = streams[0]
    container = payload.get("format") or {}
    return {
        "container": str(container.get("format_name") or ""),
        "codec": str(stream.get("codec_name") or ""),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "channel_layout": str(stream.get("channel_layout") or ""),
        "duration": float(container.get("duration") or stream.get("duration") or 0.0),
        "bit_rate": int(container.get("bit_rate") or 0),
    }


def _decode_stereo(source: Path, destination: Path) -> None:
    """Decode any container to interleaved 16 kHz int16 stereo."""
    completed = subprocess.run(
        [
            _tool("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(_TARGET_RATE),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise DecodeError(
            f"Could not decode {source.name}: {detail or 'ffmpeg failed'}"
        )


def _read_stereo(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, channels)
    left = samples[:, 0].astype(np.float32)
    right = samples[:, 1 if channels > 1 else 0].astype(np.float32)
    return left, right


def _mix_to_mono(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Average the two call legs so neither side clips the mono upload."""
    if left.size == 0:
        return np.zeros(0, dtype=np.int16)
    mixed = (left + right) / 2.0
    return np.clip(mixed, -32768, 32767).astype(np.int16)


# ---------- filename convention ----------


_STAMP_PATTERN = re.compile(
    r"^(?P<stamp>\d{8}_\d{6}\.\d+)(?P<offset>[+-]\d{4})"
    r"(?:_(?P<direction>in|out))?(?:_(?P<rest>.*))?$"
)


def _parse_stamp(stamp: str, offset: str) -> datetime | None:
    try:
        naive = datetime.strptime(stamp, "%Y%m%d_%H%M%S.%f")
    except ValueError:
        return None
    sign = -1 if offset[0] == "-" else 1
    try:
        delta = timedelta(hours=int(offset[1:3]), minutes=int(offset[3:5]))
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone(sign * delta))


def parse_recording_filename(path: Path | str) -> RecordingName:
    """Parse `{date}_{direction}_{name}_{number}` and the older 3-field form.

    The contact name may itself contain `_`, and is empty when unresolved, which
    the recorder writes as a doubled separator. So the number is taken from the
    last separated field, the direction from the field after the timestamp, and
    everything in between is the name.
    """
    stem = Path(path).stem
    match = _STAMP_PATTERN.match(stem)
    if not match:
        return RecordingName(None, None, "", "")

    rest = match.group("rest") or ""
    segments = rest.split("_") if rest else []

    phone_number = ""
    if segments and PHONE_FIELD.match(segments[-1].replace(" ", "")):
        phone_number = segments.pop()
    other_name = "_".join(segments).strip(" _")
    other_name = re.sub(r"\s+", " ", other_name)

    return RecordingName(
        recorded_at=_parse_stamp(match.group("stamp"), match.group("offset")),
        direction=match.group("direction"),
        other_name=other_name,
        phone_number=phone_number,
    )


# ---------- channel attribution ----------


def _rms(track: np.ndarray, start: float, end: float, rate: int) -> float:
    first = max(0, int(start * rate))
    last = max(first, min(track.size, int(end * rate)))
    if last <= first:
        return 0.0
    window = track[first:last]
    return float(np.sqrt(np.mean(window**2))) if window.size else 0.0


def channels_are_split(left: np.ndarray, right: np.ndarray) -> bool:
    """True when the two channels carry genuinely different signals."""
    if left.size == 0 or left.size != right.size:
        return False
    if not np.any(left) or not np.any(right):
        return False
    correlation = float(np.corrcoef(left, right)[0, 1])
    return not math.isnan(correlation) and correlation < _SPLIT_CORRELATION_THRESHOLD


def build_channel_resolver(
    left: np.ndarray,
    right: np.ndarray,
    rate: int,
    owner: str,
    other: str,
    min_margin_db: float = _DEFAULT_MIN_MARGIN_DB,
) -> Callable[[float, float], str]:
    """Attribute each segment to the louder channel, holding on uncertainty.

    A margin below ``min_margin_db`` means both parties are active (or the line
    is bleeding), so the previous speaker continues instead of the label
    flipping on noise.
    """
    state: dict[str, str | None] = {"last": None}

    def resolve(start: float, end: float) -> str:
        left_level = _rms(left, start, end, rate)
        right_level = _rms(right, start, end, rate)
        if left_level < _SILENCE_FLOOR and right_level < _SILENCE_FLOOR:
            return state["last"] or owner
        louder = max(left_level, right_level)
        quieter = min(left_level, right_level)
        margin = 20.0 * math.log10((louder + 1.0) / (quieter + 1.0))
        if margin < min_margin_db and state["last"]:
            return state["last"]
        label = owner if left_level >= right_level else other
        state["last"] = label
        return label

    return resolve


# ---------- output ----------


def _slug(text: str, fallback: str = "recording") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\r\n\t]', "_", text or "").strip(" ._")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("._") or fallback


# Identity of the audio a folder was built from. Folder names stay readable
# (`26_11_06_Amy`, matching Scribe) so the name cannot carry that identity: the
# marker does, which keeps re-runs idempotent without making names unreadable.
_SOURCE_MARKER = ".source.json"


def _source_identity(source: Path) -> dict[str, object]:
    status = source.stat()
    return {"name": source.name, "bytes": status.st_size}


def _read_source_identity(folder: Path) -> dict[str, object] | None:
    try:
        payload = json.loads((folder / _SOURCE_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _record_source_identity(folder: Path, identity: dict[str, object]) -> None:
    # If this cannot be written the transcript cannot be written either, so let
    # the failure surface rather than degrading idempotency silently.
    folder.mkdir(parents=True, exist_ok=True)
    (folder / _SOURCE_MARKER).write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )


def _output_dir(root: Path, name: RecordingName, source: Path) -> Path:
    """Return this recording's folder, creating no directories.

    Names follow Scribe (`yy_mm_dd_Contact`). Two calls with the same person on
    one day therefore share a base name, so the marker decides whether an
    existing folder is this recording (reuse) or another one (suffix `_2`).
    """
    identity = _source_identity(source)
    when = name.recorded_at or datetime.fromtimestamp(source.stat().st_mtime)
    descriptor = name.other_name or name.phone_number or source.stem
    base = f"{when.strftime('%y_%m_%d')}_{_slug(descriptor)}"

    suffix = 1
    while True:
        candidate = root / (base if suffix == 1 else f"{base}_{suffix}")
        if not candidate.is_dir():
            return candidate
        if _read_source_identity(candidate) == identity:
            return candidate
        suffix += 1


def _resolve_root(output_root: Path | str | None) -> Path:
    if output_root:
        return Path(output_root).expanduser()
    configured = ConfigManager.get_config_value("transcript_options", "root_folder")
    if configured:
        return Path(str(configured)).expanduser()
    return default_transcripts_dir()


# ---------- public entry point ----------


def transcribe_recording(
    source: Path | str,
    *,
    user_name: str | None = None,
    other_name: str | None = None,
    output_root: Path | str | None = None,
    generate_summary: bool = False,
    channel_attribution: bool = True,
    min_margin_db: float = _DEFAULT_MIN_MARGIN_DB,
    force: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> RecordingResult:
    """Transcribe one recording into `<root>/<folder>/transcript.md` and `.pdf`.

    ``output_root`` overrides the configured root. ``generate_summary`` adds
    `summary.md`/`summary.pdf` through OpenRouter. Running again is idempotent
    unless ``force`` is set, so a pipeline can safely retry.
    """

    def report(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    source = Path(source).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Recording not found: {source}")

    report("Inspecting recording...")
    details = probe_audio(source)
    if details["duration"] <= 0:
        raise DecodeError(f"{source.name} reports a zero duration.")

    parsed = parse_recording_filename(source)
    owner = user_name or ConfigManager.get_config_value("profile", "user_name") or "Me"
    identified = bool(parsed.other_name or parsed.phone_number)
    other = (other_name or parsed.other_name or parsed.phone_number or "").strip()
    if not other:
        other = "Second party" if identified else "Recording"
        identified = False
    title = (
        f"{owner} and {other}" if identified else f"Recording: {source.stem}"
    )
    participants = [owner, other] if identified else [owner]
    started_at = _started_at(parsed, source)

    root = _resolve_root(output_root)
    destination = _output_dir(root, parsed, source)
    transcript_md_path = destination / "transcript.md"
    if transcript_md_path.exists() and not force:
        report("Transcript already exists; nothing to do.")
        return RecordingResult(
            output_dir=destination,
            title=title,
            participants=participants,
            duration_seconds=details["duration"],
            started_at=started_at,
            transcript_md=transcript_md_path,
            transcript_pdf=_existing(destination / "transcript.pdf"),
            summary_md=_existing(destination / "summary.md"),
            summary_pdf=_existing(destination / "summary.pdf"),
            attribution="existing",
            reused=True,
        )

    warnings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="koe-transcribe-") as temp_dir:
        stereo_wav = Path(temp_dir) / "stereo.wav"
        report("Converting audio...")
        _decode_stereo(source, stereo_wav)
        left, right = _read_stereo(stereo_wav)

        mono_path = Path(temp_dir) / "mono.wav"
        write_mono_wav(mono_path, _mix_to_mono(left, right), _TARGET_RATE)

        split = channel_attribution and channels_are_split(left, right)
        if channel_attribution and not split:
            warnings.append(
                "Channels carry the same signal, so speakers came from "
                "diarization rather than the channel split."
            )

        report("Transcribing...")
        if split:
            segments = transcribe_file_segments(
                mono_path,
                label=owner,
                diarize=False,
                label_resolver=build_channel_resolver(
                    left,
                    right,
                    _TARGET_RATE,
                    owner,
                    other,
                    min_margin_db=min_margin_db,
                ),
            )
            attribution = "channel"
        else:
            segments = transcribe_file_segments(
                mono_path,
                label=owner,
                diarize=True,
                num_speakers=2,
            )
            attribution = "diarization"

    if not segments:
        raise DecodeError("No speech was detected in this recording.")

    duration = max(
        (segment.get("end") or 0.0 for segment in segments),
        default=details["duration"],
    )
    markdown = render_transcript(
        segments=segments,
        meeting_name=title,
        participants=participants,
        started_at=started_at,
        duration_seconds=duration,
    )

    _record_source_identity(destination, _source_identity(source))
    transcript_md_path.write_text(markdown, encoding="utf-8")

    report("Writing transcript...")
    transcript_pdf_path = render_transcript_pdf(
        segments=segments,
        meeting_name=title,
        participants=participants,
        started_at=started_at,
        duration_seconds=duration,
        recorder_name=owner,
        output_path=destination / "transcript.pdf",
    )

    summary_md_path: Path | None = None
    summary_pdf_path: Path | None = None
    if generate_summary:
        summary_md_path, summary_pdf_path = _write_summary(
            markdown, destination, title, participants, started_at, duration, owner,
            report, warnings,
        )

    return RecordingResult(
        output_dir=destination,
        title=title,
        participants=participants,
        duration_seconds=duration,
        started_at=started_at,
        transcript_md=transcript_md_path,
        transcript_pdf=transcript_pdf_path,
        summary_md=summary_md_path,
        summary_pdf=summary_pdf_path,
        segments=segments,
        attribution=attribution,
        warnings=warnings,
    )


def _write_summary(
    markdown: str,
    destination: Path,
    title: str,
    participants: list[str],
    started_at: datetime,
    duration: float,
    owner: str,
    report: Callable[[str], None],
    warnings: list[str],
) -> tuple[Path | None, Path | None]:
    """Add an optional OpenRouter summary, never losing the transcript."""
    from meeting.summarizer import SummarizerClient
    from meeting.summary_pdf import render_summary_pdf

    report("Summarising...")
    try:
        analysis = SummarizerClient().analyze(markdown, speaker_labels=participants)
    except Exception as exc:  # noqa: BLE001 - the transcript is already safe
        warnings.append(f"Summary unavailable: {exc}")
        return None, None

    if not analysis.summary:
        warnings.append("Summary was empty; the transcript stands alone.")
        return None, None

    summary_md = destination / "summary.md"
    summary_md.write_text(analysis.summary, encoding="utf-8")
    summary_pdf = render_summary_pdf(
        analysis.summary,
        destination / "summary.pdf",
        meeting_name=title,
        participants=participants,
        started_at=started_at,
        duration_seconds=duration,
        recorder_name=owner,
    )
    return summary_md, summary_pdf


def _started_at(parsed: RecordingName, source: Path) -> datetime:
    if parsed.recorded_at is not None:
        return parsed.recorded_at.astimezone()
    return datetime.fromtimestamp(source.stat().st_mtime)


def _existing(path: Path) -> Path | None:
    return path if path.exists() else None
