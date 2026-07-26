"""Render Koe's Markdown meeting summary as a clean, readable PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Spacer,
)


INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#64748B")
INDIGO = colors.HexColor("#5B6EF5")
ACCENT = colors.HexColor("#22C58B")
RULE = colors.HexColor("#DCE3EE")
PAPER = colors.white


def _inline_markup(text: str) -> str:
    """Translate Koe's small Markdown subset to ReportLab paragraph markup."""
    value = html.escape(text.strip(), quote=True)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"__([^_]+)__", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#5B6EF5">\1</link>',
        value,
    )
    return value


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "KoeTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "section": ParagraphStyle(
            "KoeSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=INK,
            spaceBefore=4 * mm,
            spaceAfter=2.5 * mm,
        ),
        "subsection": ParagraphStyle(
            "KoeSubsection",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=INDIGO,
            spaceBefore=2.5 * mm,
            spaceAfter=1.2 * mm,
        ),
        "metadata": ParagraphStyle(
            "KoeMetadata",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=MUTED,
            spaceAfter=1.5 * mm,
        ),
        "body": ParagraphStyle(
            "KoeBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=14.5,
            textColor=INK,
            spaceAfter=2.5 * mm,
        ),
        "bullet": ParagraphStyle(
            "KoeBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=INK,
            leftIndent=5 * mm,
            firstLineIndent=0,
            bulletIndent=1.5 * mm,
            bulletFontName="Helvetica-Bold",
            bulletFontSize=9,
            bulletColor=ACCENT,
            spaceAfter=1.2 * mm,
        ),
    }


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "Koe meeting summary")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def _flush_paragraph(buffer: list[str], story: list, style: ParagraphStyle) -> None:
    if not buffer:
        return
    story.append(Paragraph(_inline_markup(" ".join(buffer)), style))
    buffer.clear()


def _flush_list(items: list[str], story: list, style: ParagraphStyle) -> None:
    if not items:
        return
    for item in items:
        story.append(Paragraph(_inline_markup(item), style, bulletText="-"))
    story.append(Spacer(1, 1.5 * mm))
    items.clear()


def render_summary_pdf(markdown_text: str, output_path: Path | str) -> Path:
    """Write a polished PDF version of a Koe summary and return its path."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story: list = []
    paragraph_buffer: list[str] = []
    list_items: list[str] = []

    for raw_line in markdown_text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()

        if not line:
            _flush_paragraph(paragraph_buffer, story, styles["body"])
            _flush_list(list_items, story, styles["bullet"])
            continue

        if line == "---":
            _flush_paragraph(paragraph_buffer, story, styles["body"])
            _flush_list(list_items, story, styles["bullet"])
            story.append(Spacer(1, 1.5 * mm))
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            _flush_paragraph(paragraph_buffer, story, styles["body"])
            _flush_list(list_items, story, styles["bullet"])
            level = len(heading.group(1))
            heading_style = (
                styles["title"]
                if level == 1
                else styles["section"]
                if level == 2
                else styles["subsection"]
            )
            story.append(Paragraph(_inline_markup(heading.group(2)), heading_style))
            continue

        bullet = re.match(r"^(?:[-*+]|\d+\.)\s+(.+)$", line)
        if bullet:
            _flush_paragraph(paragraph_buffer, story, styles["body"])
            list_items.append(bullet.group(1))
            continue

        _flush_list(list_items, story, styles["bullet"])
        if not story or (
            len(story) <= 2
            and (line.startswith("Duration:") or line.startswith("Participants:"))
        ):
            _flush_paragraph(paragraph_buffer, story, styles["body"])
            story.append(Paragraph(_inline_markup(line), styles["metadata"]))
        else:
            paragraph_buffer.append(line)

    _flush_paragraph(paragraph_buffer, story, styles["body"])
    _flush_list(list_items, story, styles["bullet"])

    document = BaseDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="Koe meeting summary",
        author="Koe",
    )
    frame = Frame(
        document.leftMargin,
        document.bottomMargin,
        document.width,
        document.height,
        id="summary",
    )
    document.addPageTemplates([PageTemplate(id="summary", frames=[frame], onPage=_footer)])
    document.build(story)
    return destination
