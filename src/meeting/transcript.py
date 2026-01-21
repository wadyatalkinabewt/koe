"""
Transcript output in markdown format.
"""

import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TranscriptEntry:
    """A single entry in the transcript."""
    timestamp: float      # Seconds from start
    speaker: str          # Speaker name
    text: str             # What was said
    source: str = "mic"   # "mic" or "loopback"


@dataclass
class TranscriptWriter:
    """Writes meeting transcripts to markdown files."""

    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "Meetings" / "Transcripts")
    include_timestamps: bool = True

    # Internal state
    entries: List[TranscriptEntry] = field(default_factory=list)
    meeting_start: Optional[datetime] = None
    participants: set = field(default_factory=set)
    meeting_name: Optional[str] = None

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_meeting(self):
        """Mark the start of a new meeting."""
        self.entries = []
        self.meeting_start = datetime.now()
        self.participants = set()

    def add_entry(self, timestamp: float, speaker: str, text: str, source: str = "mic"):
        """Add a transcript entry."""
        if not text or not text.strip():
            return

        self.entries.append(TranscriptEntry(
            timestamp=timestamp,
            speaker=speaker,
            text=text.strip(),
            source=source
        ))
        self.participants.add(speaker)

    def get_duration(self) -> float:
        """Get meeting duration in seconds."""
        if not self.entries:
            return 0
        return max(e.timestamp for e in self.entries)

    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as [HH:MM:SS] or [MM:SS]."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def generate_markdown(self, notes_markdown: Optional[str] = None) -> str:
        """Generate the full markdown transcript with optional notes section."""
        if not self.meeting_start:
            self.meeting_start = datetime.now()

        duration_secs = self.get_duration()
        duration_mins = int(duration_secs // 60)

        # Use custom meeting name as title if provided
        title = f"# {self.meeting_name}" if self.meeting_name else "# Meeting Transcript"

        lines = [
            title,
            "",
            f"**Date**: {self.meeting_start.strftime('%Y-%m-%d %H:%M')}",
            f"**Duration**: {duration_mins} minutes",
            f"**Participants**: {', '.join(sorted(self.participants))}",
            "",
            "---",
            ""
        ]

        # Add notes section if provided
        if notes_markdown and notes_markdown.strip():
            lines.append(notes_markdown.strip())
            lines.append("")
            lines.append("---")
            lines.append("")

        # Add transcript section header if we have entries
        if self.entries:
            lines.append("## Full Transcript")
            lines.append("")

            for entry in self.entries:
                if self.include_timestamps:
                    ts = self.format_timestamp(entry.timestamp)
                    lines.append(f"**[{ts}] {entry.speaker}**: {entry.text}")
                else:
                    lines.append(f"**{entry.speaker}**: {entry.text}")
                lines.append("")

        return "\n".join(lines)

    def save(self, filename: Optional[str] = None, notes_markdown: Optional[str] = None) -> Path:
        """Save transcript to file with optional notes section."""
        if filename is None:
            # Generate filename from date/time
            if self.meeting_start:
                filename = self.meeting_start.strftime("meeting_%Y%m%d_%H%M%S.md")
            else:
                filename = datetime.now().strftime("meeting_%Y%m%d_%H%M%S.md")

        filepath = self.output_dir / filename
        content = self.generate_markdown(notes_markdown=notes_markdown)

        filepath.write_text(content, encoding='utf-8')
        print(f"[Meeting Transcript] Saved to: {filepath}")
        return filepath

    def get_recent_text(self, seconds: float = 30) -> str:
        """Get text from the last N seconds for copying."""
        if not self.entries:
            return ""

        cutoff = self.get_duration() - seconds
        recent = [e for e in self.entries if e.timestamp >= cutoff]

        return "\n".join(f"{e.speaker}: {e.text}" for e in recent)

    def get_full_text(self) -> str:
        """Get all text without formatting."""
        return "\n".join(f"{e.speaker}: {e.text}" for e in self.entries)
