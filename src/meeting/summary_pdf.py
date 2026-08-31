"""Render Koe's Markdown meeting summary in the shared meeting PDF style."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
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
    RULE,
    ParticipantBar,
    display_date,
    display_duration,
    font_names,
    footer_callback,
    header_styles,
    normalize_speaker_labels,
)


@dataclass
class SummaryBlock:
    kind: str
    text: str = ""
    items: list[str] = field(default_factory=list)


@dataclass
class SummarySection:
    title: str
    blocks: list[SummaryBlock] = field(default_factory=list)


def _inline_markup(text: str) -> str:
    value = html.escape(text.strip(), quote=True)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"__([^_]+)__", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#565A91">\1</link>',
        value,
    )
    return value


def _parse_date(value: str) -> datetime | None:
    for pattern in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), pattern)
        except ValueError:
            continue
    return None


def _parse_duration(value: str) -> float:
    hours = re.search(r"(\d+)\s*(?:hours?|hrs?|hr)\b", value, flags=re.I)
    minutes = re.search(r"(\d+)\s*(?:minutes?|mins?|min)\b", value, flags=re.I)
    seconds = re.search(r"(\d+)\s*(?:seconds?|secs?|sec)\b", value, flags=re.I)
    if not any((hours, minutes, seconds)):
        clock = re.fullmatch(r"\s*(\d+):(\d{2})(?::(\d{2}))?\s*", value)
        if clock:
            if clock.group(3) is None:
                return float(int(clock.group(1)) * 60 + int(clock.group(2)))
            return float(
                int(clock.group(1)) * 3600
                + int(clock.group(2)) * 60
                + int(clock.group(3))
            )
        return 0.0
    return float(
        (int(hours.group(1)) * 3600 if hours else 0)
        + (int(minutes.group(1)) * 60 if minutes else 0)
        + (int(seconds.group(1)) if seconds else 0)
    )


def _source_metadata(
    markdown_text: str,
) -> tuple[str, datetime | None, float, list[str], list[str]]:
    lines = markdown_text.replace("\r\n", "\n").split("\n")
    title = "Meeting Summary"
    started_at = None
    duration_seconds = 0.0
    participants: list[str] = []
    body_lines: list[str] = []
    consumed_title = False

    for raw_line in lines:
        line = raw_line.strip()
        if not consumed_title and line.startswith("# "):
            consumed_title = True
            heading = line[2:].strip()
            dated = re.match(
                r"^(.+?)\s+-\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})$",
                heading,
            )
            if dated:
                title = dated.group(1).strip()
                started_at = _parse_date(dated.group(2))
            else:
                title = heading or title
            continue

        metadata = re.match(
            r"^Duration:\s*([^|]+?)(?:\s*\|\s*Participants:\s*(.+))?$",
            line,
            flags=re.I,
        )
        if metadata:
            duration_seconds = _parse_duration(metadata.group(1))
            if metadata.group(2):
                participants = [
                    participant.strip()
                    for participant in metadata.group(2).split(",")
                    if participant.strip()
                ]
            continue
        if line == "---":
            continue
        body_lines.append(raw_line)
    return title, started_at, duration_seconds, participants, body_lines


def _parse_sections(lines: list[str]) -> list[SummarySection]:
    sections: list[SummarySection] = []
    current: SummarySection | None = None
    paragraph: list[str] = []
    bullets: list[str] = []

    def ensure_section() -> SummarySection:
        nonlocal current
        if current is None:
            current = SummarySection("Summary")
            sections.append(current)
        return current

    def flush_paragraph() -> None:
        if paragraph:
            ensure_section().blocks.append(
                SummaryBlock("paragraph", text=" ".join(paragraph))
            )
            paragraph.clear()

    def flush_bullets() -> None:
        if bullets:
            ensure_section().blocks.append(SummaryBlock("bullets", items=list(bullets)))
            bullets.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_bullets()
            continue

        heading = re.match(r"^(#{2,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_bullets()
            if len(heading.group(1)) == 2:
                current = SummarySection(heading.group(2).strip())
                sections.append(current)
            else:
                ensure_section().blocks.append(
                    SummaryBlock("subheading", text=heading.group(2).strip())
                )
            continue

        bullet = re.match(r"^(?:[-*+]|\d+\.)\s+(.+)$", line)
        if bullet:
            flush_paragraph()
            bullets.append(bullet.group(1).strip())
            continue

        flush_bullets()
        paragraph.append(line)

    flush_paragraph()
    flush_bullets()
    return [section for section in sections if section.blocks]


def _styles() -> dict[str, ParagraphStyle]:
    fonts = font_names()
    return {
        "section": ParagraphStyle(
            "KoeSummarySection",
            fontName=fonts["sans_bold"],
            fontSize=13.5,
            leading=17,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "subsection": ParagraphStyle(
            "KoeSummarySubsection",
            fontName=fonts["sans_bold"],
            fontSize=10.2,
            leading=13,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "body": ParagraphStyle(
            "KoeSummaryBody",
            fontName=fonts["sans"],
            fontSize=9.35,
            leading=13.4,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "KoeSummaryBullet",
            fontName=fonts["sans"],
            fontSize=9.25,
            leading=13.2,
            textColor=INK,
            alignment=TA_LEFT,
        ),
        "participant": ParagraphStyle(
            "KoeSummaryParticipant",
            fontName=fonts["sans_bold"],
            fontSize=13.2,
            leading=16,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "inline_meta": ParagraphStyle(
            "KoeSummaryInlineMeta",
            fontName=fonts["sans"],
            fontSize=9.4,
            leading=12,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
        "empty_state": ParagraphStyle(
            "KoeSummaryEmptyState",
            fontName=fonts["sans"],
            fontSize=9.2,
            leading=12.5,
            textColor=MUTED,
            alignment=TA_LEFT,
        ),
    }


def _section_heading(title: str, width: float, style: ParagraphStyle) -> Table:
    return Table(
        [[ParticipantBar(CORAL), Paragraph(_inline_markup(title), style)]],
        colWidths=[0.65 * mm, width - 0.65 * mm],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("LEFTPADDING", (1, 0), (1, 0), 2.1 * mm),
                ("RIGHTPADDING", (1, 0), (1, 0), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0.45 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0.45 * mm),
            ]
        ),
    )


def _plain_card(
    content,
    width: float,
    *,
    background=CARD_BG,
    border=CARD_BORDER,
) -> Table:
    return Table(
        [[content]],
        colWidths=[width],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), background),
                ("BOX", (0, 0), (-1, -1), 0.65, border),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.1 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
            ]
        ),
    )


def _paragraph_stack(
    paragraphs: list[str],
    width: float,
    style: ParagraphStyle,
) -> Table:
    rows = [[Paragraph(_inline_markup(text), style)] for text in paragraphs]
    return Table(
        rows,
        colWidths=[width],
        style=TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -2), 2.2 * mm),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 0),
            ]
        ),
    )


def _bullet_stack(
    items: list[str],
    width: float,
    style: ParagraphStyle,
    *,
    bullet_color: colors.Color,
) -> Table:
    bullet_style = ParagraphStyle(
        f"{style.name}Marker{bullet_color.hexval()}",
        parent=style,
        textColor=bullet_color,
        fontName=font_names()["sans_bold"],
    )
    rows = [
        [
            Paragraph("•", bullet_style),
            Paragraph(_inline_markup(item), style),
        ]
        for item in items
    ]
    return Table(
        rows,
        colWidths=[3.5 * mm, width - 3.5 * mm],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -2), 1.45 * mm),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 0),
            ]
        ),
    )


def _summary_header_story(
    meeting_name: str,
    started_at: datetime,
    duration_seconds: float,
    participants: list[str],
    width: float,
    styles: dict[str, ParagraphStyle],
    *,
    meeting_mode: str,
    participant_name: str,
) -> list:
    headings = header_styles()
    if meeting_mode.endswith("one_on_one"):
        subtitle = participant_name or next(
            (name for name in participants if name.strip()),
            "Participant",
        )
        return [
            Paragraph(html.escape(meeting_name), headings["title"]),
            Paragraph(html.escape(subtitle), styles["participant"]),
            Spacer(1, 1.3 * mm),
            Paragraph(
                f"{html.escape(display_date(started_at))}"
                f" &nbsp;&nbsp;|&nbsp;&nbsp; {html.escape(display_duration(duration_seconds))}",
                styles["inline_meta"],
            ),
            Spacer(1, 3.2 * mm),
            HRFlowable(
                width="100%",
                thickness=1.2,
                color=CORAL,
                spaceBefore=0,
                spaceAfter=5 * mm,
            ),
        ]

    ordered, display = normalize_speaker_labels(participants)
    participant_text = ", ".join(display[name] for name in ordered) or "Not identified"
    metadata = Table(
        [
            [
                Table(
                    [
                        [Paragraph("Duration", headings["meta_heading"])],
                        [
                            Paragraph(
                                html.escape(display_duration(duration_seconds)),
                                headings["meta_value"],
                            )
                        ],
                    ],
                    style=TableStyle(
                        [
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 0.8 * mm),
                            ("BOTTOMPADDING", (0, 1), (-1, 1), 0),
                        ]
                    ),
                ),
                Table(
                    [
                        [Paragraph("Participants", headings["meta_heading"])],
                        [
                            Paragraph(
                                html.escape(participant_text),
                                headings["meta_value"],
                            )
                        ],
                    ],
                    style=TableStyle(
                        [
                            ("LEFTPADDING", (0, 0), (-1, -1), 0),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                            ("TOPPADDING", (0, 0), (-1, -1), 0),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 0.8 * mm),
                            ("BOTTOMPADDING", (0, 1), (-1, 1), 0),
                        ]
                    ),
                ),
            ]
        ],
        colWidths=[width * 0.24, width * 0.76],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEEAE2")),
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.4 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4 * mm),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        ),
    )
    return [
        Paragraph(html.escape(meeting_name), headings["title"]),
        Paragraph(display_date(started_at), headings["date"]),
        Spacer(1, 3.2 * mm),
        HRFlowable(
            width="100%",
            thickness=1.2,
            color=CORAL,
            spaceBefore=0,
            spaceAfter=3.2 * mm,
        ),
        metadata,
        Spacer(1, 5 * mm),
    ]


def render_summary_pdf(
    markdown_text: str,
    output_path: Path | str,
    *,
    meeting_name: str | None = None,
    participants: list[str] | None = None,
    started_at: datetime | None = None,
    duration_seconds: float | None = None,
    recorder_name: str = "",
    meeting_mode: str = "online_group",
    participant_name: str = "",
) -> Path:
    """Write a polished PDF summary using the shared meeting visual system."""
    parsed_name, parsed_start, parsed_duration, parsed_participants, body = (
        _source_metadata(markdown_text)
    )
    resolved_name = str(meeting_name or parsed_name or "Meeting Summary").strip()
    resolved_start = started_at or parsed_start or datetime.now()
    resolved_duration = (
        float(duration_seconds) if duration_seconds is not None else parsed_duration
    )
    resolved_participants = list(participants or parsed_participants)
    sections = _parse_sections(body)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    document = BaseDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=19 * mm,
        title=f"{resolved_name} summary",
        author="Koe",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        id="summary",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document.addPageTemplates(
        [
            PageTemplate(
                id="summary",
                frames=[frame],
                onPage=footer_callback(resolved_name, resolved_start),
            )
        ]
    )

    story = _summary_header_story(
        resolved_name,
        resolved_start,
        resolved_duration,
        resolved_participants,
        document.width,
        styles,
        meeting_mode=meeting_mode,
        participant_name=participant_name,
    )
    for section_index, section in enumerate(sections):
        if section_index:
            story.append(Spacer(1, 4 * mm))
        section_heading = _section_heading(
            section.title,
            document.width,
            styles["section"],
        )
        section_heading.keepWithNext = True
        section_gap = Spacer(1, 2.2 * mm)
        section_gap.keepWithNext = True
        story.append(section_heading)
        story.append(section_gap)

        section_key = section.title.casefold()

        if section_key == "summary":
            summary_paragraphs = [
                block.text for block in section.blocks if block.kind == "paragraph"
            ]
            if summary_paragraphs:
                content_width = document.width - 6.4 * mm
                story.append(
                    _plain_card(
                        _paragraph_stack(
                            summary_paragraphs,
                            content_width,
                            styles["body"],
                        ),
                        document.width,
                    )
                )
            continue

        if "decision" in section_key or "question" in section_key:
            items = [
                item
                for block in section.blocks
                if block.kind == "bullets"
                for item in block.items
            ]
            paragraphs = [
                block.text for block in section.blocks if block.kind == "paragraph"
            ]
            if "decision" in section_key:
                tint = colors.HexColor("#EDF6F1")
                border = colors.HexColor("#AED6C7")
                marker = colors.HexColor("#278767")
            else:
                tint = colors.HexColor("#FBF4E8")
                border = colors.HexColor("#E4C995")
                marker = colors.HexColor("#A76A18")
            content_width = document.width - 6.4 * mm
            empty_copy = " ".join(paragraphs).strip().rstrip(".").casefold()
            is_empty_state = not items and empty_copy in {
                "no formal decisions recorded",
                "no open questions",
                "none recorded",
            }
            if is_empty_state:
                story.append(
                    Paragraph(
                        _inline_markup(paragraphs[0]),
                        styles["empty_state"],
                    )
                )
                continue
            if items:
                content = _bullet_stack(
                    items,
                    content_width,
                    styles["bullet"],
                    bullet_color=marker,
                )
            else:
                content = _paragraph_stack(
                    paragraphs or ["None recorded."],
                    content_width,
                    styles["body"],
                )
            story.append(
                _plain_card(
                    content,
                    document.width,
                    background=tint,
                    border=border,
                )
            )
            continue

        if "action" in section_key:
            block_index = 0
            first_action_group = True
            while block_index < len(section.blocks):
                block = section.blocks[block_index]
                if block.kind == "subheading":
                    owner = Paragraph(
                        _inline_markup(block.text),
                        styles["subsection"],
                    )
                    next_block = (
                        section.blocks[block_index + 1]
                        if block_index + 1 < len(section.blocks)
                        else None
                    )
                    if next_block and next_block.kind == "bullets":
                        owner_card = _plain_card(
                            _bullet_stack(
                                next_block.items,
                                document.width - 6.4 * mm,
                                styles["bullet"],
                                bullet_color=MUTED,
                            ),
                            document.width,
                            background=colors.HexColor("#F7F5F7"),
                        )
                        action_group = [
                            owner,
                            Spacer(1, 1.4 * mm),
                            owner_card,
                            Spacer(1, 1.7 * mm),
                        ]
                        if first_action_group:
                            story.pop()
                            story.pop()
                            action_group = [
                                section_heading,
                                section_gap,
                                *action_group,
                            ]
                            first_action_group = False
                        story.append(KeepTogether(action_group))
                        block_index += 2
                        continue
                    story.append(owner)
                    story.append(Spacer(1, 1.4 * mm))
                elif block.kind == "paragraph":
                    story.append(
                        _plain_card(
                            Paragraph(_inline_markup(block.text), styles["body"]),
                            document.width,
                            background=colors.HexColor("#F7F5F7"),
                        )
                    )
                elif block.kind == "bullets":
                    story.append(
                        _plain_card(
                            _bullet_stack(
                                block.items,
                                document.width - 6.4 * mm,
                                styles["bullet"],
                                bullet_color=MUTED,
                            ),
                            document.width,
                            background=colors.HexColor("#F7F5F7"),
                        )
                    )
                block_index += 1
            continue

        for block in section.blocks:
            if block.kind == "subheading":
                story.append(
                    Paragraph(_inline_markup(block.text), styles["subsection"])
                )
                story.append(Spacer(1, 1.4 * mm))
                continue

            if block.kind == "paragraph":
                paragraph = Paragraph(_inline_markup(block.text), styles["body"])
                story.append(paragraph)
                story.append(
                    Spacer(
                        1,
                        3.2 * mm if section_key == "topics discussed" else 1.8 * mm,
                    )
                )
                continue

            if block.kind == "bullets":
                story.append(
                    _plain_card(
                        _bullet_stack(
                            block.items,
                            document.width - 6.4 * mm,
                            styles["bullet"],
                            bullet_color=MUTED,
                        ),
                        document.width,
                        background=colors.HexColor("#F7F5F7"),
                    )
                )
                story.append(Spacer(1, 1.7 * mm))

    document.build(story)
    return destination
