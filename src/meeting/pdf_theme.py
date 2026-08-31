"""Shared visual system for Koe meeting PDFs."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

PAGE_BG = colors.HexColor("#FAF8F4")
INK = colors.HexColor("#272535")
MUTED = colors.HexColor("#696679")
RULE = colors.HexColor("#D9D5CC")
CARD_BG = colors.HexColor("#F4F2F7")
CARD_BORDER = colors.HexColor("#DDD9E5")
CORAL = colors.HexColor("#C15B5F")
KOE_GREEN = colors.HexColor("#22B983")

# The recorder has a reserved green. Other speakers use a high-contrast,
# colour-blind-conscious sequence and remain stable within one document.
SPEAKER_COLORS = (
    colors.HexColor("#565A91"),
    colors.HexColor("#C47A2C"),
    colors.HexColor("#B9576A"),
    colors.HexColor("#3F78AD"),
    colors.HexColor("#865C98"),
    colors.HexColor("#75833C"),
    colors.HexColor("#BD5B3B"),
    colors.HexColor("#597484"),
    colors.HexColor("#9B6B35"),
    colors.HexColor("#7C668D"),
)

_UNKNOWN_SPEAKER = re.compile(r"^speaker(?:_|\s+)\d+$", flags=re.IGNORECASE)


@lru_cache(maxsize=1)
def font_names() -> dict[str, str]:
    """Register Windows fonts when available and return safe font names."""
    definitions = {
        "sans": ("KoeSans", Path(r"C:\Windows\Fonts\segoeui.ttf"), "Helvetica"),
        "sans_bold": (
            "KoeSansBold",
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            "Helvetica-Bold",
        ),
        "serif": ("KoeSerif", Path(r"C:\Windows\Fonts\georgia.ttf"), "Times-Roman"),
        "serif_bold": (
            "KoeSerifBold",
            Path(r"C:\Windows\Fonts\georgiab.ttf"),
            "Times-Bold",
        ),
    }
    resolved: dict[str, str] = {}
    for role, (registered_name, path, fallback) in definitions.items():
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont(registered_name, str(path)))
                resolved[role] = registered_name
                continue
            except Exception:
                pass
        resolved[role] = fallback
    if resolved["sans"] == "KoeSans" and resolved["sans_bold"] == "KoeSansBold":
        pdfmetrics.registerFontFamily(
            "KoeSans",
            normal="KoeSans",
            bold="KoeSansBold",
            italic="KoeSans",
            boldItalic="KoeSansBold",
        )
    if resolved["serif"] == "KoeSerif" and resolved["serif_bold"] == "KoeSerifBold":
        pdfmetrics.registerFontFamily(
            "KoeSerif",
            normal="KoeSerif",
            bold="KoeSerifBold",
            italic="KoeSerif",
            boldItalic="KoeSerifBold",
        )
    return resolved


def display_date(started_at: datetime) -> str:
    return f"{started_at.day} {started_at.strftime('%B %Y')}"


def display_duration(duration_seconds: float) -> str:
    total = max(0, int(round(duration_seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hr")
    if minutes:
        parts.append(f"{minutes} min")
    if seconds or not parts:
        parts.append(f"{seconds} sec")
    return " ".join(parts)


def _unique_labels(labels: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for label in labels:
        value = str(label or "").strip()
        if value and value not in unique:
            unique.append(value)
    return unique


def normalize_speaker_labels(
    participants: Iterable[str],
    segment_labels: Iterable[str] = (),
) -> tuple[list[str], dict[str, str]]:
    """Put known names first and renumber anonymous speakers from one."""
    raw_order = _unique_labels([*participants, *segment_labels])
    known = [label for label in raw_order if not _UNKNOWN_SPEAKER.fullmatch(label)]
    unknown = [label for label in raw_order if _UNKNOWN_SPEAKER.fullmatch(label)]
    ordered = known + unknown
    display = {label: label for label in known}
    display.update(
        {label: f"Speaker {index + 1}" for index, label in enumerate(unknown)}
    )
    return ordered, display


def assign_speaker_colors(
    ordered_speakers: Iterable[str],
    display_labels: dict[str, str],
    recorder_name: str,
) -> dict[str, colors.Color]:
    """Reserve Koe green for the configured recorder, then assign the palette."""
    recorder = str(recorder_name or "").strip().casefold()
    assigned: dict[str, colors.Color] = {}
    next_palette = 0
    for raw_label in ordered_speakers:
        display = display_labels[raw_label]
        if recorder and display.casefold() == recorder:
            assigned[raw_label] = KOE_GREEN
            continue
        assigned[raw_label] = SPEAKER_COLORS[next_palette % len(SPEAKER_COLORS)]
        next_palette += 1
    return assigned


def header_styles() -> dict[str, ParagraphStyle]:
    fonts = font_names()
    return {
        "title": ParagraphStyle(
            "KoePdfTitle",
            fontName=fonts["sans_bold"],
            fontSize=24,
            leading=28,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=1.2 * mm,
        ),
        "date": ParagraphStyle(
            "KoePdfDate",
            fontName=fonts["sans"],
            fontSize=11,
            leading=14,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "meta_heading": ParagraphStyle(
            "KoePdfMetaHeading",
            fontName=fonts["sans_bold"],
            fontSize=8.8,
            leading=10.8,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "meta_value": ParagraphStyle(
            "KoePdfMetaValue",
            fontName=fonts["sans"],
            fontSize=8.6,
            leading=10.8,
            textColor=INK,
            alignment=TA_LEFT,
        ),
    }


class ParticipantBar(Flowable):
    """A short, thin colour marker aligned to a participant name."""

    def __init__(self, accent: colors.Color) -> None:
        super().__init__()
        self.accent = accent
        self.width = 0.65 * mm
        self.height = 3.25 * mm

    def draw(self) -> None:
        self.canv.setFillColor(self.accent)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)


def _participant_chip(
    label: str,
    accent: colors.Color,
    style: ParagraphStyle,
) -> Table:
    return Table(
        [[ParticipantBar(accent), Paragraph(html.escape(label), style)]],
        colWidths=[0.65 * mm, None],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("TOPPADDING", (0, 0), (0, 0), 0),
                ("BOTTOMPADDING", (0, 0), (0, 0), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 1.1 * mm),
                ("RIGHTPADDING", (1, 0), (1, 0), 3 * mm),
                ("TOPPADDING", (1, 0), (1, 0), 0),
                ("BOTTOMPADDING", (1, 0), (1, 0), 0),
            ]
        ),
    )


def _participant_rows(
    ordered_speakers: list[str],
    display_labels: dict[str, str],
    speaker_colors: dict[str, colors.Color],
    style: ParagraphStyle,
    available_width: float,
) -> Table:
    fonts = font_names()
    rows: list[list[Table]] = []
    current: list[Table] = []
    current_width = 0.0
    for speaker in ordered_speakers:
        label = display_labels[speaker]
        estimated_width = (
            pdfmetrics.stringWidth(label, fonts["sans"], style.fontSize)
            + 0.65 * mm
            + 4.1 * mm
        )
        if current and current_width + estimated_width > available_width:
            rows.append(current)
            current = []
            current_width = 0.0
        current.append(_participant_chip(label, speaker_colors[speaker], style))
        current_width += estimated_width
    if current:
        rows.append(current)
    if not rows:
        rows = [[Paragraph("Not identified", style)]]

    row_tables = []
    for row in rows:
        row_tables.append(
            [
                Table(
                    [row],
                    hAlign="LEFT",
                    style=TableStyle(
                        [
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    ),
                )
            ]
        )
    return Table(
        row_tables,
        hAlign="LEFT",
        style=TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8 * mm),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 0),
            ]
        ),
    )


def _metadata_cell(
    heading: str,
    value,
    styles: dict[str, ParagraphStyle],
) -> Table:
    return Table(
        [
            [Paragraph(html.escape(heading), styles["meta_heading"])],
            [value],
        ],
        style=TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 0.8 * mm),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 0),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        ),
    )


def build_header_story(
    meeting_name: str,
    started_at: datetime,
    duration_seconds: float,
    ordered_speakers: list[str],
    display_labels: dict[str, str],
    speaker_colors: dict[str, colors.Color],
    document_width: float,
    *,
    space_after: float = 4 * mm,
) -> list:
    styles = header_styles()
    participant_list = _participant_rows(
        ordered_speakers,
        display_labels,
        speaker_colors,
        styles["meta_value"],
        document_width * 0.7 - 8 * mm,
    )
    metadata = Table(
        [
            [
                _metadata_cell(
                    "Duration",
                    Paragraph(display_duration(duration_seconds), styles["meta_value"]),
                    styles,
                ),
                _metadata_cell("Participants", participant_list, styles),
            ]
        ],
        colWidths=[document_width * 0.3, document_width * 0.7],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEEAE2")),
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4 * mm),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        ),
    )
    return [
        Paragraph(html.escape(meeting_name), styles["title"]),
        Paragraph(display_date(started_at), styles["date"]),
        Spacer(1, 3.2 * mm),
        HRFlowable(
            width="100%",
            thickness=1.2,
            color=CORAL,
            spaceBefore=0,
            spaceAfter=3.2 * mm,
        ),
        metadata,
        Spacer(1, space_after),
    ]


def footer_callback(meeting_name: str, started_at: datetime):
    footer_text = f"{display_date(started_at)} - {meeting_name}"
    fonts = font_names()

    def draw_footer(canvas, document) -> None:
        width, height = A4
        canvas.saveState()
        canvas.setFillColor(PAGE_BG)
        canvas.rect(0, 0, width, height, fill=1, stroke=0)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 13.5 * mm, width - 18 * mm, 13.5 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont(fonts["sans"], 8)
        canvas.drawString(18 * mm, 9 * mm, footer_text)
        canvas.drawRightString(width - 18 * mm, 9 * mm, str(document.page))
        canvas.restoreState()

    return draw_footer
