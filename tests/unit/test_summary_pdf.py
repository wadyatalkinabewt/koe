from pathlib import Path


def test_render_summary_pdf_creates_readable_pdf(tmp_path: Path):
    from meeting.summary_pdf import render_summary_pdf

    output = tmp_path / "summary.pdf"
    render_summary_pdf(
        """# Weekly Sync - 26 Jul 2026
Duration: 34 minutes | Participants: Operator, Jane

---

## Summary
The team reviewed progress and agreed on the next delivery date.

---

## Key Decisions
- Ship the revised proposal on Friday.
- Keep the existing review process.

## Action Items
##### Operator
- Send the revised proposal.
""",
        output,
    )

    contents = output.read_bytes()
    assert contents.startswith(b"%PDF-")
    assert len(contents) > 2_000
