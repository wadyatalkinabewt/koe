from datetime import datetime
from pathlib import Path


def test_render_summary_pdf_creates_readable_pdf(tmp_path: Path):
    from meeting.summary_pdf import render_summary_pdf

    output = tmp_path / "summary.pdf"
    render_summary_pdf(
        """# Weekly Sync - 26 Jul 2026
Duration: 34 minutes | Participants: Alex, Jane

---

## Summary
The team reviewed progress and agreed on the next delivery date.

---

## Key Decisions
- Ship the revised proposal on Friday.
- Keep the existing review process.

## Action Items
##### Alex
- Send the revised proposal.
""",
        output,
        meeting_name="Weekly Sync",
        participants=["Alex", "Jane"],
        started_at=datetime(2026, 7, 26, 10, 0),
        duration_seconds=34 * 60,
        recorder_name="Alex",
    )

    contents = output.read_bytes()
    assert contents.startswith(b"%PDF-")
    assert len(contents) > 3_000


def test_summary_source_metadata_parser_reads_legacy_header():
    from meeting.summary_pdf import _source_metadata

    title, started_at, duration, participants, _body = _source_metadata(
        """# Weekly Sync - 26 Jul 2026
Duration: 34 minutes | Participants: Alex, Jane

## Summary
Done.
"""
    )

    assert title == "Weekly Sync"
    assert started_at == datetime(2026, 7, 26)
    assert duration == 34 * 60
    assert participants == ["Alex", "Jane"]


def test_render_one_on_one_summary_uses_compact_header(tmp_path: Path):
    from meeting.summary_pdf import render_summary_pdf

    output = tmp_path / "one-on-one.pdf"
    render_summary_pdf(
        "# Summary\n\nReviewed the invoice workflow.",
        output,
        meeting_name="Invoice Workflow",
        participants=["Alex", "Casey Example"],
        started_at=datetime(2026, 7, 20, 9, 0),
        duration_seconds=42 * 60,
        recorder_name="Alex",
        meeting_mode="online_one_on_one",
        participant_name="Casey Example",
    )

    contents = output.read_bytes()
    assert contents.startswith(b"%PDF-")
    assert len(contents) > 3_000
