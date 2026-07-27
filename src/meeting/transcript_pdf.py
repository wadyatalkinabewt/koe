"""Render Koe meeting segments as the approved transcript PDF."""

from __future__ import annotations

import html
from pathlib import Path

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from meeting.pdf_theme import (
    CARD_BG,
    CARD_BORDER,
    CORAL,
    INK,
    MUTED,
    assign_speaker_colors,
    build_header_story,
    font_names,
    footer_callback,
    normalize_speaker_labels,
)
from meeting.transcript import format_timestamp, merge_consecutive_same_speaker


def _split_turn(text: str, max_words: int = 105) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        stop = min(len(words), start + max_words)
        if stop < len(words):
            lower = max(start + 55, stop - 25)
            stop = next(
                (
                    index + 1
                    for index in range(stop - 1, lower - 1, -1)
                    if words[index].endswith((".", "?", "!"))
                ),
                stop,
            )
        chunks.append(" ".join(words[start:stop]))
        start = stop
    return chunks


def _styles() -> dict[str, ParagraphStyle]:
    fonts = font_names()
    return {
        "speaker": ParagraphStyle(
            "KoeTranscriptSpeaker",
            fontName=fonts["sans_bold"],
            fontSize=8.7,
            leading=10.8,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "body": ParagraphStyle(
            "KoeTranscriptBody",
            fontName=fonts["sans"],
            fontSize=9.45,
            leading=13.1,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "notes_heading": ParagraphStyle(
            "KoeTranscriptNotesHeading",
            fontName=fonts["sans_bold"],
            fontSize=13.5,
            leading=17,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "notes_body": ParagraphStyle(
            "KoeTranscriptNotesBody",
            fontName=fonts["sans"],
            fontSize=9.35,
            leading=13.4,
            textColor=INK,
            alignment=TA_LEFT,
        ),
    }


def _speaker_card(
    label: str,
    timestamp: str,
    text: str,
    accent,
    width: float,
    styles: dict[str, ParagraphStyle],
    *,
    continued: bool,
) -> Table:
    continuation = " - continued" if continued else ""
    header = Paragraph(
        f'<font color="{accent.hexval()}">{html.escape(label)}</font>'
        f'<font color="{MUTED.hexval()}">{continuation} &nbsp;&nbsp; '
        f"{html.escape(timestamp)}</font>",
        styles["speaker"],
    )
    body = Paragraph(html.escape(text), styles["body"])
    content = Table(
        [[header], [body]],
        colWidths=[width - 5.5 * mm],
        style=TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5 * mm),
                ("TOPPADDING", (0, 0), (-1, 0), 1.65 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 0.7 * mm),
                ("TOPPADDING", (0, 1), (-1, 1), 0),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 2.1 * mm),
                ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ]
        ),
    )
    return Table(
        [["", content]],
        colWidths=[2 * mm, width - 2 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), accent),
                ("BACKGROUND", (1, 0), (1, 0), CARD_BG),
                ("BOX", (0, 0), (-1, -1), 0.55, CARD_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        ),
    )


def render_transcript_pdf(
    *,
    segments: list[dict],
    meeting_name: str,
    participants: list[str],
    started_at,
    duration_seconds: float,
    recorder_name: str,
    output_path: Path | str,
    notes_text: str = "",
) -> Path:
    """Write the approved coloured-card transcript PDF."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    utterances = merge_consecutive_same_speaker(segments)
    ordered_speakers, display_labels = normalize_speaker_labels(
        participants,
        [utterance["label"] for utterance in utterances],
    )
    speaker_colors = assign_speaker_colors(
        ordered_speakers,
        display_labels,
        recorder_name,
    )
    styles = _styles()

    document = BaseDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=19 * mm,
        title=meeting_name,
        author="Koe",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        id="transcript",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document.addPageTemplates(
        [
            PageTemplate(
                id="transcript",
                frames=[frame],
                onPage=footer_callback(meeting_name, started_at),
            )
        ]
    )
    story = build_header_story(
        meeting_name,
        started_at,
        duration_seconds,
        ordered_speakers,
        display_labels,
        speaker_colors,
        document.width,
    )
    for utterance in utterances:
        raw_label = utterance["label"]
        label = display_labels[raw_label]
        accent = speaker_colors[raw_label]
        for index, chunk in enumerate(_split_turn(utterance["text"])):
            story.append(
                _speaker_card(
                    label,
                    format_timestamp(utterance["start"]),
                    chunk,
                    accent,
                    document.width,
                    styles,
                    continued=index > 0,
                )
            )
            story.append(Spacer(1, 1.35 * mm))

    notes = notes_text.strip()
    if notes:
        story.append(Spacer(1, 3.2 * mm))
        notes_heading = Table(
            [
                [
                    "",
                    Paragraph("Notes", styles["notes_heading"]),
                ]
            ],
            colWidths=[0.9 * mm, document.width - 0.9 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), CORAL),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                    ("LEFTPADDING", (1, 0), (1, 0), 3 * mm),
                ]
            ),
        )
        notes_card = Table(
            [
                [
                    Paragraph(
                        html.escape(notes).replace("\n", "<br/>"),
                        styles["notes_body"],
                    )
                ]
            ],
            colWidths=[document.width],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
                    ("BOX", (0, 0), (-1, -1), 0.55, CARD_BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3.5 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3.5 * mm),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.4 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ]
            ),
        )
        story.extend(
            [
                notes_heading,
                Spacer(1, 2.2 * mm),
                notes_card,
            ]
        )

    document.build(story)
    return destination
