from datetime import datetime
from pathlib import Path


def test_known_speakers_precede_cleanly_renumbered_unknowns():
    from meeting.pdf_theme import normalize_speaker_labels

    ordered, labels = normalize_speaker_labels(
        ["Speaker 7", "Jordan", "Speaker 3", "Alex"],
    )

    assert ordered == ["Jordan", "Alex", "Speaker 7", "Speaker 3"]
    assert [labels[label] for label in ordered] == [
        "Jordan",
        "Alex",
        "Speaker 1",
        "Speaker 2",
    ]


def test_configured_recorder_gets_reserved_koe_green():
    from meeting.pdf_theme import (
        KOE_GREEN,
        assign_speaker_colors,
        normalize_speaker_labels,
    )

    ordered, labels = normalize_speaker_labels(["Jordan", "Alex", "Speaker 8"])
    assigned = assign_speaker_colors(ordered, labels, "Alex")

    assert assigned["Alex"].hexval() == KOE_GREEN.hexval()
    assert assigned["Jordan"].hexval() != KOE_GREEN.hexval()
    assert assigned["Speaker 8"].hexval() != KOE_GREEN.hexval()


def test_render_transcript_pdf_creates_readable_pdf(tmp_path: Path):
    from meeting.transcript_pdf import render_transcript_pdf

    output = tmp_path / "transcript.pdf"
    render_transcript_pdf(
        segments=[
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Welcome to the planning meeting.",
                "label": "Speaker 7",
            },
            {
                "start": 1.2,
                "end": 2.0,
                "text": "Let us review the priorities.",
                "label": "Alex",
            },
        ],
        meeting_name="Planning Meeting",
        participants=["Speaker 7", "Alex"],
        started_at=datetime(2026, 7, 27, 9, 0),
        duration_seconds=120,
        recorder_name="Alex",
        output_path=output,
        notes_text="Confirm the owners before Friday.",
    )

    contents = output.read_bytes()
    assert contents.startswith(b"%PDF-")
    assert len(contents) > 3_000
