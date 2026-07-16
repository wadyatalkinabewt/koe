"""
Build meeting transcript markdown from interleaved speaker segments.

Pure data transformation — no I/O during the meeting itself, no recovery
file. Recording and transcription run separately; this module just takes
the resulting segments and renders the final transcript.
"""

from datetime import datetime
from typing import List, Dict


def format_timestamp(seconds: float) -> str:
    """Seconds → MM:SS or HH:MM:SS depending on length."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def merge_consecutive_same_speaker(segments: List[Dict]) -> List[Dict]:
    """Combine adjacent segments from the same speaker into one utterance.

    Each input segment: {start, end, text, label}.
    Output is sorted by start and merged where label matches the previous.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.get("start", 0))
    merged: List[Dict] = []
    for seg in sorted_segs:
        if merged and merged[-1]["label"] == seg["label"]:
            merged[-1]["end"] = max(merged[-1].get("end", 0), seg.get("end", 0))
            merged[-1]["text"] = (merged[-1]["text"] + " " + seg["text"]).strip()
        else:
            merged.append({
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                "label": seg.get("label", "Speaker"),
            })
    return merged


def render_transcript(
    segments: List[Dict],
    meeting_name: str,
    participants: List[str],
    started_at: datetime,
    duration_seconds: float,
) -> str:
    """Build the final transcript markdown.

    segments: list of {start, end, text, label} after participant rename.
    """
    utterances = merge_consecutive_same_speaker(segments)

    lines = [f"# {meeting_name}", ""]
    lines.append(f"**Date**: {started_at.strftime('%Y-%m-%d %H:%M')}")
    duration_min = max(1, round(duration_seconds / 60))
    lines.append(f"**Duration**: {duration_min} minute{'s' if duration_min != 1 else ''}")
    if participants:
        lines.append(f"**Participants**: {', '.join(participants)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    for utt in utterances:
        ts = format_timestamp(utt["start"])
        lines.append(f"**[{ts}] {utt['label']}**: {utt['text']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
