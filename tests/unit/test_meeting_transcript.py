import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from meeting.transcript import format_timestamp, merge_consecutive_same_speaker, render_transcript


def test_timestamp_formats_minutes_and_hours():
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3661) == "01:01:01"


def test_consecutive_speaker_segments_merge_in_time_order():
    segments = [
        {"start": 4, "end": 5, "text": "Second", "label": "Taylor"},
        {"start": 0, "end": 1, "text": "Hello", "label": "Alex"},
        {"start": 1, "end": 2, "text": "again", "label": "Alex"},
    ]

    assert merge_consecutive_same_speaker(segments) == [
        {"start": 0, "end": 2, "text": "Hello again", "label": "Alex"},
        {"start": 4, "end": 5, "text": "Second", "label": "Taylor"},
    ]


def test_render_transcript_includes_stream_labels():
    rendered = render_transcript(
        segments=[{"start": 0, "end": 1, "text": "Hello.", "label": "Alex"}],
        meeting_name="Planning",
        participants=["Alex", "Taylor"],
        started_at=datetime(2026, 7, 13, 10, 30),
        duration_seconds=60,
    )

    assert "# Planning" in rendered
    assert "**Participants**: Alex, Taylor" in rendered
    assert "**[00:00] Alex**: Hello." in rendered


def test_render_transcript_appends_notes_as_a_final_section():
    rendered = render_transcript(
        segments=[{"start": 0, "end": 1, "text": "Hello.", "label": "Alex"}],
        meeting_name="Planning",
        participants=["Alex"],
        started_at=datetime(2026, 7, 13, 10, 30),
        duration_seconds=60,
        notes_text="Follow up with Jordan.\nConfirm the deadline.",
    )

    assert rendered.index("## Transcript") < rendered.index("## Notes")
    assert rendered.endswith("Follow up with Jordan.\nConfirm the deadline.\n")
