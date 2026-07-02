from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from scipy.signal import correlate, correlation_lags


KOE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = KOE_DIR / "src"
MEETINGS_DIR = KOE_DIR / "Meetings"
TARGET_SAMPLE_RATE = 16000

HALLUCINATION_TEXT = {
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "we'll be right back",
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return slug or "Meeting"


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def meeting_dir_for(name: str, meeting_date: str | None) -> Path:
    if meeting_date:
        dt = datetime.strptime(meeting_date, "%Y-%m-%d")
    else:
        dt = datetime.now()
    return MEETINGS_DIR / f"{dt.strftime('%y_%m_%d')}_{slugify(name)}"


def run_ffmpeg_convert(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    log(f"Converting {input_path.name} -> {output_path.name}")
    subprocess.run(command, check=True)


def load_audio_float(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def rms_envelope(audio: np.ndarray, sample_rate: int, window_sec: float = 0.02) -> np.ndarray:
    window = max(1, int(sample_rate * window_sec))
    usable = (len(audio) // window) * window
    if usable <= 0:
        return np.zeros(1, dtype=np.float32)
    blocks = audio[:usable].reshape(-1, window)
    envelope = np.sqrt(np.mean(blocks * blocks, axis=1))
    envelope = np.log1p(envelope * 50.0)
    return (envelope - envelope.mean()) / (envelope.std() + 1e-9)


def estimate_offset_seconds(reference: np.ndarray, other: np.ndarray, sample_rate: int) -> float:
    # Use the first 20 minutes. It is enough for reliable alignment and avoids
    # spending time correlating the full meeting.
    max_samples = min(len(reference), len(other), sample_rate * 60 * 20)
    if max_samples < sample_rate:
        return 0.0

    ref_env = rms_envelope(reference[:max_samples], sample_rate)
    other_env = rms_envelope(other[:max_samples], sample_rate)
    corr = correlate(ref_env, other_env, mode="full", method="fft")
    lags = correlation_lags(len(ref_env), len(other_env), mode="full")

    frames_per_second = 1.0 / 0.02
    max_lag_frames = int(120 * frames_per_second)
    mask = np.abs(lags) <= max_lag_frames
    if not np.any(mask):
        return 0.0

    masked_corr = corr[mask]
    masked_lags = lags[mask]
    best_index = int(np.argmax(masked_corr))
    best_lag = float(masked_lags[best_index])

    if 0 < best_index < len(masked_corr) - 1:
        y0, y1, y2 = masked_corr[best_index - 1], masked_corr[best_index], masked_corr[best_index + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            best_lag += float(0.5 * (y0 - y2) / denom)

    return best_lag / frames_per_second


def active_normalize(audio: np.ndarray, target_rms: float = 0.07) -> np.ndarray:
    active = audio[np.abs(audio) > 0.004]
    rms = float(np.sqrt(np.mean(active * active))) if active.size else float(np.sqrt(np.mean(audio * audio)))
    gain = min(target_rms / (rms + 1e-9), 3.0)
    return audio * gain


def build_aligned_mix(wav_paths: list[Path], output_path: Path) -> dict:
    loaded = [load_audio_float(path) for path in wav_paths]
    sample_rates = {sample_rate for _, sample_rate in loaded}
    if sample_rates != {TARGET_SAMPLE_RATE}:
        raise RuntimeError(f"Expected {TARGET_SAMPLE_RATE} Hz WAVs, got {sorted(sample_rates)}")

    offsets = [0.0]
    reference = loaded[0][0]
    for audio, sample_rate in loaded[1:]:
        offsets.append(estimate_offset_seconds(reference, audio, sample_rate))

    min_offset = min(offsets)
    shifted_offsets = [offset - min_offset for offset in offsets]
    sample_offsets = [int(round(offset * TARGET_SAMPLE_RATE)) for offset in shifted_offsets]
    length = max(sample_offset + len(audio) for sample_offset, (audio, _) in zip(sample_offsets, loaded))

    mix = np.zeros(length, dtype=np.float32)
    for sample_offset, (audio, _) in zip(sample_offsets, loaded):
        normalized = active_normalize(audio)
        mix[sample_offset : sample_offset + len(normalized)] += normalized / len(loaded)

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mix, TARGET_SAMPLE_RATE, subtype="PCM_16")

    return {
        "mix_path": str(output_path),
        "duration_seconds": len(mix) / TARGET_SAMPLE_RATE,
        "offsets_seconds": offsets,
        "shifted_offsets_seconds": shifted_offsets,
        "peak": float(np.max(np.abs(mix))) if mix.size else 0.0,
    }


def diarize(audio_path: Path, num_speakers: int) -> list[dict]:
    import torch
    from pyannote.audio import Pipeline

    load_dotenv(KOE_DIR / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set in Koe .env")

    log("Loading pyannote speaker diarization model")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using diarization device: {device}")
    pipeline.to(device)

    audio, sample_rate = load_audio_float(audio_path)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    log(f"Running diarization on {len(audio) / sample_rate:.1f}s audio")
    result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers)
    annotation = getattr(result, "exclusive_speaker_diarization", None) or getattr(
        result, "speaker_diarization", result
    )

    segments: list[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if turn.end - turn.start < 0.12:
            continue
        segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})

    segments.sort(key=lambda item: (item["start"], item["end"]))
    log(f"Diarization produced {len(segments)} speaker turns")
    return segments


def transcribe_with_koe(audio_path: Path) -> list[dict]:
    load_dotenv(KOE_DIR / ".env")
    sys.path.insert(0, str(SRC_DIR))
    from transcription import transcribe_segments

    audio, sample_rate = sf.read(str(audio_path), dtype="int16")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)

    log("Running Koe transcription")
    segments = transcribe_segments(audio, label="Speaker", sample_rate=sample_rate)
    cleaned = [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": str(seg["text"]).strip()}
        for seg in segments
        if str(seg.get("text", "")).strip()
    ]
    log(f"Koe produced {len(cleaned)} transcript segments")
    return cleaned


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def parse_map(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected RAW=Name mapping, got {value!r}")
        raw, name = value.split("=", 1)
        parsed[raw.strip()] = name.strip()
    return parsed


def assign_speakers(
    transcript: list[dict],
    diarization: list[dict],
    speaker_names: list[str],
    primary_speaker: str | None,
    speaker_map: dict[str, str],
    early_primary_minutes: float,
) -> list[dict]:
    assigned: list[dict] = []
    for seg in transcript:
        scores: dict[str, float] = defaultdict(float)
        for dseg in diarization:
            if dseg["end"] < seg["start"] - 0.5:
                continue
            if dseg["start"] > seg["end"] + 0.5:
                break
            scores[dseg["speaker"]] += overlap(seg["start"], seg["end"], dseg["start"], dseg["end"])

        raw_speaker = max(scores.items(), key=lambda item: item[1])[0] if scores else "Unknown"
        assigned.append({**seg, "speaker_raw": raw_speaker})

    raw_by_first_seen: list[str] = []
    for seg in assigned:
        raw = seg["speaker_raw"]
        if raw != "Unknown" and raw not in raw_by_first_seen:
            raw_by_first_seen.append(raw)

    name_map = dict(speaker_map)
    if not name_map and speaker_names:
        remaining_names = list(speaker_names)
        if primary_speaker and primary_speaker in remaining_names:
            early_words: Counter[str] = Counter()
            cutoff = early_primary_minutes * 60
            for seg in assigned:
                if seg["speaker_raw"] != "Unknown" and seg["start"] <= cutoff:
                    early_words[seg["speaker_raw"]] += len(seg["text"].split())
            if early_words:
                primary_raw = early_words.most_common(1)[0][0]
                name_map[primary_raw] = primary_speaker
                remaining_names.remove(primary_speaker)

        for raw in raw_by_first_seen:
            if raw not in name_map:
                name_map[raw] = remaining_names.pop(0) if remaining_names else raw

    if not name_map:
        for index, raw in enumerate(raw_by_first_seen, start=1):
            name_map[raw] = f"Speaker {index}"

    for seg in assigned:
        seg["speaker"] = name_map.get(seg["speaker_raw"], seg["speaker_raw"])

    log(f"Speaker map: {name_map}")
    return assigned


def apply_manual_speaker_fixes(segments: list[dict], fixes: list[str]) -> None:
    for fix in fixes:
        if "=" not in fix:
            raise ValueError(f"Expected START_SECONDS=Speaker, got {fix!r}")
        start_text, speaker = fix.split("=", 1)
        target = float(start_text)
        speaker = speaker.strip()
        match = min(segments, key=lambda seg: abs(float(seg["start"]) - target), default=None)
        if match and abs(float(match["start"]) - target) <= 0.5:
            log(f"Manual speaker fix: {match['start']:.1f}s -> {speaker}")
            match["speaker"] = speaker
        else:
            raise ValueError(f"No transcript segment starts within 0.5s of {target}")


def clean_hallucinations(segments: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    removed = 0
    for seg in segments:
        text = str(seg["text"]).strip()
        normalized = re.sub(r"[.!\s]+$", "", text.lower()).strip()
        if normalized in HALLUCINATION_TEXT:
            removed += 1
            continue
        new_text = re.sub(r"\s+Thank you\.?$", "", text, flags=re.I).strip()
        if not new_text:
            removed += 1
            continue
        seg = dict(seg)
        seg["text"] = new_text
        cleaned.append(seg)
    if removed:
        log(f"Removed/trimmed {removed} likely silence hallucination segment(s)")
    return cleaned


def merge_for_readability(segments: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for seg in sorted(segments, key=lambda item: item["start"]):
        if not seg["text"]:
            continue
        if (
            merged
            and merged[-1]["speaker"] == seg["speaker"]
            and seg["start"] - merged[-1]["end"] <= 2.0
            and len(merged[-1]["text"]) < 1400
        ):
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            joiner = "" if merged[-1]["text"].endswith((" ", "\n")) else " "
            merged[-1]["text"] = f"{merged[-1]['text']}{joiner}{seg['text']}".strip()
        else:
            merged.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"],
                    "text": seg["text"],
                }
            )
    return merged


def render_transcript(
    output_path: Path,
    meeting_name: str,
    meeting_date: str | None,
    participants: list[str],
    duration_seconds: float,
    segments: list[dict],
) -> None:
    date_text = meeting_date or datetime.now().strftime("%Y-%m-%d")
    merged = merge_for_readability(segments)
    lines = [
        f"# {meeting_name}",
        "",
        f"**Date**: {date_text}",
        f"**Duration**: {max(1, round(duration_seconds / 60))} minutes",
    ]
    if participants:
        lines.append(f"**Participants**: {', '.join(participants)}")
    lines.extend(["", "---", "", "## Transcript", ""])

    for seg in merged:
        lines.append(f"**[{format_timestamp(seg['start'])}] {seg['speaker']}**: {seg['text']}")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    log(f"Wrote transcript: {output_path}")


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log(f"Wrote {path.name}")


def run_full(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else meeting_dir_for(args.meeting_name, args.date)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"

    wav_paths: list[Path] = []
    for index, raw_path in enumerate(args.audio_files, start=1):
        raw = Path(raw_path).expanduser().resolve()
        if not raw.exists():
            raise FileNotFoundError(raw)
        wav_path = audio_dir / f"input_{index}.16k.wav"
        run_ffmpeg_convert(raw, wav_path)
        wav_paths.append(wav_path)

    mix_path = audio_dir / "aligned_mix.16k.wav"
    mix_meta = build_aligned_mix(wav_paths, mix_path)
    write_json(output_dir / "audio_alignment.json", mix_meta)

    diarization = diarize(mix_path, args.num_speakers)
    write_json(output_dir / "diarization_segments.json", diarization)

    transcript = transcribe_with_koe(mix_path)
    write_json(output_dir / "koe_transcript_segments.json", transcript)

    assigned = assign_speakers(
        transcript,
        diarization,
        args.speaker,
        args.primary_speaker,
        parse_map(args.speaker_map),
        args.early_primary_minutes,
    )
    apply_manual_speaker_fixes(assigned, args.set_speaker)
    assigned = clean_hallucinations(assigned)
    write_json(output_dir / "assigned_segments.json", assigned)

    render_transcript(
        output_dir / "transcript.md",
        args.meeting_name,
        args.date,
        args.speaker,
        mix_meta["duration_seconds"],
        assigned,
    )


def run_render_only(args: argparse.Namespace) -> None:
    output_dir = Path(args.render_only).resolve()
    assigned_path = output_dir / "assigned_segments.json"
    if assigned_path.exists():
        assigned = json.loads(assigned_path.read_text(encoding="utf-8"))
    else:
        transcript = json.loads((output_dir / "koe_transcript_segments.json").read_text(encoding="utf-8"))
        diarization = json.loads((output_dir / "diarization_segments.json").read_text(encoding="utf-8"))
        assigned = assign_speakers(
            transcript,
            diarization,
            args.speaker,
            args.primary_speaker,
            parse_map(args.speaker_map),
            args.early_primary_minutes,
        )

    speaker_map = parse_map(args.speaker_map)
    if speaker_map:
        for seg in assigned:
            if seg.get("speaker_raw") in speaker_map:
                seg["speaker"] = speaker_map[seg["speaker_raw"]]

    apply_manual_speaker_fixes(assigned, args.set_speaker)
    assigned = clean_hallucinations(assigned)
    write_json(assigned_path, assigned)

    alignment_path = output_dir / "audio_alignment.json"
    if alignment_path.exists():
        duration_seconds = json.loads(alignment_path.read_text(encoding="utf-8")).get("duration_seconds", 0)
    else:
        duration_seconds = max((seg.get("end", 0) for seg in assigned), default=0)

    render_transcript(
        output_dir / "transcript.md",
        args.meeting_name or output_dir.name,
        args.date,
        args.speaker,
        duration_seconds,
        assigned,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe existing meeting recordings with alignment, pyannote diarization, and Koe."
    )
    parser.add_argument("audio_files", nargs="*", help="Input audio/video recordings to align and mix.")
    parser.add_argument("--meeting-name", default="Meeting", help="Name used in transcript heading and folder slug.")
    parser.add_argument("--date", help="Meeting date as YYYY-MM-DD.")
    parser.add_argument("--output-dir", help="Output folder. Defaults to Koe Meetings/YY_MM_DD_<meeting-name>.")
    parser.add_argument("--speaker", action="append", default=[], help="Participant name. Repeat in desired order.")
    parser.add_argument("--primary-speaker", help="Name to assign to dominant early speaker when no raw map is supplied.")
    parser.add_argument("--early-primary-minutes", type=float, default=6.0)
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--speaker-map", action="append", default=[], help="Override raw speaker map, e.g. SPEAKER_00=Alex.")
    parser.add_argument("--set-speaker", action="append", default=[], help="Manual segment fix, e.g. 1035.4=Konrad.")
    parser.add_argument("--render-only", help="Existing output folder to re-render from saved JSON.")
    args = parser.parse_args()
    if not args.render_only and not args.audio_files:
        parser.error("audio_files are required unless --render-only is used")
    return args


def main() -> None:
    args = parse_args()
    if args.render_only:
        run_render_only(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
