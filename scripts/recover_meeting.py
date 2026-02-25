"""
Recover meeting transcript from failed audio files + recovery JSONL.
Transcribes failed audio, merges with existing entries, generates markdown.
"""

import sys
import os
import json
import wave
import time
import base64
import numpy as np
import requests
from pathlib import Path
from datetime import datetime

# Config
SERVER_URL = "http://localhost:9876"
LOGS_DIR = Path(__file__).parent.parent / "logs"
RECOVERY_FILE = Path(__file__).parent.parent / ".transcript_recovery.jsonl"
TIMEOUT_PER_SECOND = 4.0  # seconds of timeout per second of audio
CONNECT_TIMEOUT = 5.0
MIN_READ_TIMEOUT = 30.0

def transcribe_wav(wav_path: Path, language="en", initial_prompt=None) -> tuple:
    """Transcribe a WAV file using the local server. Returns (text, success)."""
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        duration = len(audio_data) / sample_rate
        if duration < 0.5:
            return "", False

        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_data.tobytes()).decode('ascii')

        payload = {
            "audio_base64": audio_b64,
            "sample_rate": sample_rate,
            "language": language,
            "vad_filter": True,
        }
        if initial_prompt:
            payload["initial_prompt"] = initial_prompt

        read_timeout = max(MIN_READ_TIMEOUT, duration * TIMEOUT_PER_SECOND)

        response = requests.post(
            f"{SERVER_URL}/transcribe",
            json=payload,
            timeout=(CONNECT_TIMEOUT, read_timeout)
        )

        if response.status_code == 200:
            data = response.json()
            text = data.get("text", "").strip()
            return text, True
        else:
            print(f"  Server error: {response.status_code}")
            return "", False

    except requests.Timeout:
        print(f"  Timeout transcribing {wav_path.name}")
        return "", False
    except Exception as e:
        print(f"  Error: {e}")
        return "", False


def extract_timestamp_from_filename(filename: str) -> float:
    """Extract timestamp from failed_audio filename for ordering."""
    # Format: failed_audio_{reason}_{YYYYMMDD}_{HHMMSS}.wav
    parts = filename.replace(".wav", "").split("_")
    # Last two parts are date and time
    date_str = parts[-2]  # YYYYMMDD
    time_str = parts[-1]  # HHMMSS
    try:
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        return dt.timestamp()
    except:
        return 0


def main():
    # 1. Load existing recovery entries
    print("=" * 60)
    print("MEETING RECOVERY")
    print("=" * 60)

    recovery_entries = []
    meeting_start = None
    meeting_start_ts = None

    if RECOVERY_FILE.exists():
        lines = RECOVERY_FILE.read_text(encoding="utf-8").strip().split("\n")
        header = json.loads(lines[0])
        meeting_start_str = header.get("meeting_start")
        if meeting_start_str:
            meeting_start = datetime.fromisoformat(meeting_start_str)
            meeting_start_ts = meeting_start.timestamp()

        for line in lines[1:]:
            try:
                entry = json.loads(line)
                recovery_entries.append(entry)
            except:
                continue

        print(f"Loaded {len(recovery_entries)} existing transcript entries from recovery")
        print(f"Meeting started: {meeting_start}")
        if recovery_entries:
            last_ts = max(e["timestamp"] for e in recovery_entries)
            print(f"Last entry timestamp: {last_ts:.0f}s ({last_ts/60:.1f} min)")
    else:
        print("No recovery file found!")
        meeting_start = datetime.now()
        meeting_start_ts = meeting_start.timestamp()

    # 2. Find failed audio files from today
    date_str = "20260219"
    failed_files = sorted(
        LOGS_DIR.glob(f"failed_audio_*{date_str}*.wav"),
        key=lambda f: f.name
    )

    print(f"\nFound {len(failed_files)} failed audio files to transcribe")

    if not failed_files:
        print("Nothing to recover!")
        return

    # 3. Check server health
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=3)
        if r.status_code != 200:
            print("Server not healthy!")
            return
        print("Server is healthy\n")
    except:
        print("Server not reachable!")
        return

    # 4. Transcribe each file
    initial_prompt = "Bryce, Calum, Sash, Sritam, Thomas, Push, Bevis, Maxim, Jacky. Use proper punctuation including periods, commas, and question marks."

    recovered_entries = []
    total_duration = 0

    for i, wav_path in enumerate(failed_files):
        # Get file info
        try:
            with wave.open(str(wav_path), 'rb') as wf:
                sr = wf.getframerate()
                nf = wf.getnframes()
                duration = nf / sr
        except:
            print(f"[{i+1}/{len(failed_files)}] SKIP {wav_path.name} (can't read)")
            continue

        total_duration += duration
        file_ts = extract_timestamp_from_filename(wav_path.name)

        # Calculate meeting-relative timestamp
        if meeting_start_ts:
            relative_ts = file_ts - meeting_start_ts
        else:
            relative_ts = 0

        print(f"[{i+1}/{len(failed_files)}] {wav_path.name} ({duration:.1f}s, t={relative_ts:.0f}s)...", end=" ", flush=True)

        text, success = transcribe_wav(wav_path, language="en", initial_prompt=initial_prompt)

        if success and text:
            print(f"OK ({len(text)} chars)")
            recovered_entries.append({
                "timestamp": relative_ts,
                "speaker": "(Recovered)",
                "text": text,
                "source": "recovered"
            })
        elif success:
            print("(empty/silence)")
        else:
            print("FAILED")

    print(f"\n{'=' * 60}")
    print(f"Transcribed {len(recovered_entries)}/{len(failed_files)} files")
    print(f"Total audio duration: {total_duration:.0f}s ({total_duration/60:.1f} min)")

    # 5. Merge all entries
    all_entries = recovery_entries + recovered_entries
    all_entries.sort(key=lambda e: e["timestamp"])

    # Deduplicate (some entries might overlap)
    seen_texts = set()
    deduped = []
    for entry in all_entries:
        text_key = entry["text"].strip().lower()[:50]
        if text_key not in seen_texts and len(entry["text"].strip()) > 3:
            seen_texts.add(text_key)
            deduped.append(entry)

    all_entries = deduped
    print(f"Total entries after merge + dedup: {len(all_entries)}")

    # 6. Collect participants
    participants = set(e["speaker"] for e in all_entries)

    # 7. Generate markdown
    duration_secs = max(e["timestamp"] for e in all_entries) if all_entries else 0
    duration_mins = int(duration_secs // 60)

    def fmt_ts(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    lines = [
        "# Meeting Transcript",
        "",
        f"**Date**: {meeting_start.strftime('%Y-%m-%d %H:%M') if meeting_start else '2026-02-19'}",
        f"**Duration**: {duration_mins} minutes",
        f"**Participants**: {', '.join(sorted(participants))}",
        "",
        "---",
        "",
        "## Full Transcript",
        ""
    ]

    for entry in all_entries:
        ts = fmt_ts(entry["timestamp"])
        lines.append(f"**[{ts}] {entry['speaker']}**: {entry['text']}")
        lines.append("")

    transcript_md = "\n".join(lines)

    # 8. Save transcript
    output_dir = Path("C:/Users/Galbraith/Documents/Work/Haiku/meetings/Transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"26_02_19_Recovered_Meeting.md"
    output_path = output_dir / filename
    output_path.write_text(transcript_md, encoding="utf-8")
    print(f"\nTranscript saved to: {output_path}")

    # Also save raw JSON for the summarizer
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(all_entries, indent=2), encoding="utf-8")
    print(f"Raw entries saved to: {json_path}")

    return str(output_path)


if __name__ == "__main__":
    main()
