"""
ElevenLabs Scribe v2 benchmark for saved Koe audio clips.

This intentionally makes no Groq calls. It can compare ElevenLabs output
against existing Koe debug markdown where that debug text already exists.
Private transcript outputs are written under logs/, which is gitignored.
"""

from __future__ import annotations

import argparse
from array import array
import json
import math
import os
import re
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
TAIL_AUDIO_DIR = ROOT / "logs" / "snippet_tail_audio"
TRANSCRIPTION_DEBUG_DIR = ROOT / "logs" / "transcription_debug"
FAILED_AUDIO_GLOB = "failed_audio_*.wav"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "elevenlabs_stt_bench"

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
API_KEY_ENV_NAMES = ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "XI_API_KEY")

KOE_KEYTERMS = [
    "Alex",
    "Acme",
    "Civis",
    "Hetzner",
    "Hermes",
    "Koe",
    "Codex",
    "OpenRouter",
    "Groq",
    "Whisper",
    "Scribe",
    "ElevenLabs",
    "Supabase",
    "pgvector",
    "MCP",
    "JSONL",
    "SSE",
    "CLAUDE.md",
    "AGENTS.md",
    "PowerShell",
    "Dunedin",
    "Auckland",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_api_key() -> str | None:
    load_dotenv(ROOT / ".env")
    for name in API_KEY_ENV_NAMES:
        value = os.getenv(name)
        if value:
            return value
    return None


def wav_metadata(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        frame_count = wav.getnframes()
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(frame_count)

    duration_s = frame_count / sample_rate if sample_rate else 0
    rms = pcm_rms(frames, sample_width)
    dbfs = round(20 * math.log10(rms / 32768), 1) if rms > 0 else None
    return {
        "duration_s": round(duration_s, 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "rms": rms,
        "dbfs": dbfs,
        "file_size_bytes": path.stat().st_size,
        "last_modified": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
    }


def pcm_rms(frames: bytes, sample_width: int) -> int:
    if not frames:
        return 0
    if sample_width != 2:
        return 0
    samples = array("h")
    samples.frombytes(frames)
    if not samples:
        return 0
    square_sum = sum(sample * sample for sample in samples)
    return int(math.sqrt(square_sum / len(samples)))


def elevenlabs_file_format(meta: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    if (
        meta["sample_rate"] == 16000
        and meta["channels"] == 1
        and meta["sample_width_bytes"] == 2
    ):
        return "pcm_s16le_16"
    return "other"


def parse_debug_markdown(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8", errors="replace")
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"^## (Raw Groq|Raw Transcription|Post Processed|Final)\s*$", text, re.M))
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        key = match.group(1).lower().replace(" ", "_")
        if key == "raw_transcription":
            key = "raw_groq"
        sections[key] = text[start:end].strip()
    return sections


def tail_index(path: Path) -> int:
    match = re.search(r"tail_(\d+)\.wav$", path.name)
    return int(match.group(1)) if match else 9999


def build_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    tail_paths = sorted(TAIL_AUDIO_DIR.glob("tail_*.wav"), key=tail_index)
    for path in tail_paths[: args.limit_tails]:
        idx = tail_index(path)
        meta = wav_metadata(path)
        debug_path = TRANSCRIPTION_DEBUG_DIR / f"snippet_debug_{idx}.md"
        samples.append(
            {
                "id": f"tail_{idx}",
                "kind": "snippet_tail",
                "audio_path": str(path),
                "debug_path": str(debug_path) if debug_path.exists() else None,
                "baseline": parse_debug_markdown(debug_path),
                "audio": meta,
            }
        )

    if args.include_failed:
        failed_candidates: list[tuple[float, Path, dict[str, Any]]] = []
        for path in sorted((ROOT / "logs").glob(FAILED_AUDIO_GLOB)):
            try:
                meta = wav_metadata(path)
            except wave.Error:
                continue
            duration = float(meta["duration_s"])
            dbfs = meta["dbfs"]
            if args.min_failed_seconds <= duration <= args.max_failed_seconds:
                if dbfs is not None and dbfs < args.min_failed_dbfs:
                    continue
                failed_candidates.append((path.stat().st_mtime, path, meta))

        if args.failed_sort == "loudest":
            failed_candidates.sort(key=lambda item: item[2]["dbfs"] or -999, reverse=True)
        else:
            failed_candidates.sort(reverse=True)
        for _mtime, path, meta in failed_candidates[: args.max_failed]:
            samples.append(
                {
                    "id": path.stem,
                    "kind": "failed_audio",
                    "audio_path": str(path),
                    "debug_path": None,
                    "baseline": {},
                    "audio": meta,
                }
            )

    return samples


def form_fields(args: argparse.Namespace, meta: dict[str, Any]) -> list[tuple[str, str]]:
    fields = [
        ("model_id", args.model_id),
        ("language_code", args.language_code),
        ("tag_audio_events", "false"),
        ("timestamps_granularity", "word"),
        ("diarize", "false"),
        ("num_speakers", "1"),
        ("temperature", str(args.temperature)),
        ("file_format", elevenlabs_file_format(meta, args.file_format)),
        ("no_verbatim", "false"),
    ]
    if args.keyterms_profile == "koe":
        fields.extend(("keyterms", term) for term in KOE_KEYTERMS)
    return fields


def transcribe_sample(
    sample: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    response_dir: Path,
) -> dict[str, Any]:
    path = Path(sample["audio_path"])
    response_path = response_dir / f"{sample['id']}.json"

    if response_path.exists() and not args.force:
        response_json = json.loads(response_path.read_text(encoding="utf-8"))
        return {
            "sample_id": sample["id"],
            "kind": sample["kind"],
            "audio_path": sample["audio_path"],
            "cached": True,
            "elapsed_ms": None,
            "status_code": 200,
            "response_path": str(response_path),
            "text": response_json.get("text", ""),
            "language_code": response_json.get("language_code"),
            "language_probability": response_json.get("language_probability"),
            "error": None,
        }

    headers = {"xi-api-key": api_key}
    data = form_fields(args, sample["audio"])
    with path.open("rb") as audio_file:
        files = {"file": (path.name, audio_file, "audio/wav")}
        t0 = time.perf_counter()
        response = requests.post(
            ELEVENLABS_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=args.timeout,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    try:
        response_json = response.json()
    except ValueError:
        response_json = {"_raw": response.text[:2000]}

    response_path.write_text(
        json.dumps(response_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "sample_id": sample["id"],
        "kind": sample["kind"],
        "audio_path": sample["audio_path"],
        "cached": False,
        "elapsed_ms": elapsed_ms,
        "status_code": response.status_code,
        "response_path": str(response_path),
        "text": response_json.get("text", "") if response.ok else "",
        "language_code": response_json.get("language_code"),
        "language_probability": response_json.get("language_probability"),
        "error": None if response.ok else response_json,
    }


def write_manifest(
    output_dir: Path,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    manifest = {
        "created_at": utc_now_iso(),
        "model_id": args.model_id,
        "endpoint": ELEVENLABS_URL,
        "keyterms_profile": args.keyterms_profile,
        "sample_count": len(samples),
        "samples": [
            {
                "id": sample["id"],
                "kind": sample["kind"],
                "audio_path": sample["audio_path"],
                "debug_path": sample["debug_path"],
                "audio": sample["audio"],
                "baseline_available": bool(sample["baseline"]),
            }
            for sample in samples
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def write_summary(
    output_dir: Path,
    samples: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> Path:
    samples_by_id = {sample["id"]: sample for sample in samples}
    summary_path = output_dir / "summary.md"
    lines = [
        "# ElevenLabs STT Benchmark",
        "",
        f"Created: {utc_now_iso()}",
        "",
        "| sample | kind | duration | status | elapsed | baseline | ElevenLabs text |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for result in results:
        sample = samples_by_id[result["sample_id"]]
        baseline = sample["baseline"].get("raw_groq") or sample["baseline"].get("final") or ""
        status = "cached" if result["cached"] else str(result["status_code"])
        elapsed = "" if result["elapsed_ms"] is None else f"{result['elapsed_ms']:.1f}ms"
        lines.append(
            "| {sample_id} | {kind} | {duration:.1f}s | {status} | {elapsed} | {baseline} | {text} |".format(
                sample_id=result["sample_id"],
                kind=result["kind"],
                duration=float(sample["audio"]["duration_s"]),
                status=status,
                elapsed=elapsed,
                baseline=escape_markdown_table(baseline),
                text=escape_markdown_table(result.get("text", "")),
            )
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def escape_markdown_table(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    response_dir = output_dir / "responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(args)
    manifest_path = write_manifest(output_dir, samples, args)
    print(f"Selected {len(samples)} samples.")
    print(f"Wrote manifest: {manifest_path}")

    if args.dry_run:
        for sample in samples:
            print(
                f"  {sample['id']:<38} {sample['kind']:<13} "
                f"{sample['audio']['duration_s']:>7.2f}s  {sample['audio_path']}"
            )
        return 0

    api_key = get_api_key()
    if not api_key:
        print(
            "Missing ElevenLabs API key. Set ELEVENLABS_API_KEY, ELEVEN_API_KEY, "
            "or XI_API_KEY in the environment or .env."
        )
        return 2

    results_path = output_dir / "results.jsonl"
    results: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as results_file:
        for idx, sample in enumerate(samples, start=1):
            result = transcribe_sample(sample, args, api_key, response_dir)
            results.append(result)
            results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            tag = "OK" if not result.get("error") else "ERR"
            cached = " cached" if result["cached"] else ""
            print(f"[{idx:>2}/{len(samples)}] {sample['id']:<38} {tag}{cached}")

    summary_path = write_summary(output_dir, samples, results)
    print(f"Wrote results: {results_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ElevenLabs Scribe v2 against saved Koe audio."
    )
    parser.add_argument("--model-id", default="scribe_v2")
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--file-format",
        choices=("other", "pcm_s16le_16", "auto"),
        default="other",
        help="ElevenLabs input file_format. Keep 'other' for WAV containers.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit-tails", type=int, default=5)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--max-failed", type=int, default=10)
    parser.add_argument("--min-failed-seconds", type=float, default=8.0)
    parser.add_argument("--max-failed-seconds", type=float, default=30.0)
    parser.add_argument("--min-failed-dbfs", type=float, default=-80.0)
    parser.add_argument("--failed-sort", choices=("newest", "loudest"), default="newest")
    parser.add_argument("--keyterms-profile", choices=("none", "koe"), default="none")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
