"""
Transcript output in markdown format.

Includes crash recovery via incremental saves to .transcript_recovery.jsonl
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# Recovery file location (in project root, hidden file)
RECOVERY_FILE = Path(__file__).parent.parent.parent / ".transcript_recovery.jsonl"


@dataclass
class TranscriptEntry:
    """A single entry in the transcript."""
    timestamp: float      # Seconds from start
    speaker: str          # Speaker name
    text: str             # What was said
    source: str = "mic"   # "mic" or "loopback"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "text": self.text,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptEntry":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            speaker=data["speaker"],
            text=data["text"],
            source=data.get("source", "mic")
        )


@dataclass
class TranscriptWriter:
    """Writes meeting transcripts to markdown files.

    Includes crash recovery: entries are incrementally saved to a recovery file
    as they arrive. If the app crashes mid-meeting, the recovery file can be
    used to restore the transcript on next startup.
    """

    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "Meetings" / "Transcripts")
    include_timestamps: bool = True

    # Internal state
    entries: List[TranscriptEntry] = field(default_factory=list)
    meeting_start: Optional[datetime] = None
    participants: set = field(default_factory=set)
    meeting_name: Optional[str] = None

    # Recovery state
    _recovery_enabled: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_meeting(self):
        """Mark the start of a new meeting."""
        self.entries = []
        self.meeting_start = datetime.now()
        self.participants = set()

        # Initialize recovery file with meeting metadata
        if self._recovery_enabled:
            self._init_recovery_file()

    def _init_recovery_file(self):
        """Initialize the recovery file with meeting metadata."""
        try:
            # Write header line with meeting info
            header = {
                "_type": "header",
                "meeting_start": self.meeting_start.isoformat() if self.meeting_start else None,
                "meeting_name": self.meeting_name,
                "output_dir": str(self.output_dir)
            }
            RECOVERY_FILE.write_text(json.dumps(header) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"[Transcript] Warning: Failed to init recovery file: {e}")

    def _append_to_recovery(self, entry: TranscriptEntry):
        """Append an entry to the recovery file."""
        if not self._recovery_enabled:
            return
        try:
            with open(RECOVERY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            print(f"[Transcript] Warning: Failed to append to recovery: {e}")

    @classmethod
    def has_recovery_data(cls) -> bool:
        """Check if there's a recovery file with data."""
        if not RECOVERY_FILE.exists():
            return False
        try:
            # Check if file has more than just the header
            lines = RECOVERY_FILE.read_text(encoding="utf-8").strip().split("\n")
            return len(lines) > 1  # More than just the header
        except Exception:
            return False

    @classmethod
    def get_recovery_info(cls) -> Optional[dict]:
        """Get info about the recovery file (meeting name, entry count, etc.)."""
        if not RECOVERY_FILE.exists():
            return None
        try:
            lines = RECOVERY_FILE.read_text(encoding="utf-8").strip().split("\n")
            if not lines:
                return None

            header = json.loads(lines[0])
            if header.get("_type") != "header":
                return None

            entry_count = len(lines) - 1  # Exclude header
            return {
                "meeting_name": header.get("meeting_name", "Unknown"),
                "meeting_start": header.get("meeting_start"),
                "entry_count": entry_count,
                "output_dir": header.get("output_dir")
            }
        except Exception:
            return None

    @classmethod
    def load_from_recovery(cls) -> Optional["TranscriptWriter"]:
        """Load transcript from recovery file."""
        if not RECOVERY_FILE.exists():
            return None

        try:
            lines = RECOVERY_FILE.read_text(encoding="utf-8").strip().split("\n")
            if not lines:
                return None

            # Parse header
            header = json.loads(lines[0])
            if header.get("_type") != "header":
                return None

            # Create writer with recovered settings
            output_dir = Path(header.get("output_dir", cls.output_dir))
            writer = cls(output_dir=output_dir)
            writer.meeting_name = header.get("meeting_name")

            if header.get("meeting_start"):
                writer.meeting_start = datetime.fromisoformat(header["meeting_start"])
            else:
                writer.meeting_start = datetime.now()

            # Load entries
            for line in lines[1:]:
                try:
                    data = json.loads(line)
                    entry = TranscriptEntry.from_dict(data)
                    writer.entries.append(entry)
                    writer.participants.add(entry.speaker)
                except Exception:
                    continue  # Skip malformed entries

            print(f"[Transcript] Recovered {len(writer.entries)} entries from crash recovery file")
            return writer
        except Exception as e:
            print(f"[Transcript] Failed to load recovery file: {e}")
            return None

    @classmethod
    def delete_recovery_file(cls):
        """Delete the recovery file (call after successful save)."""
        try:
            if RECOVERY_FILE.exists():
                RECOVERY_FILE.unlink()
                print("[Transcript] Deleted recovery file")
        except Exception as e:
            print(f"[Transcript] Warning: Failed to delete recovery file: {e}")

    def add_entry(self, timestamp: float, speaker: str, text: str, source: str = "mic"):
        """Add a transcript entry (also appends to recovery file)."""
        if not text or not text.strip():
            return

        entry = TranscriptEntry(
            timestamp=timestamp,
            speaker=speaker,
            text=text.strip(),
            source=source
        )
        self.entries.append(entry)
        self.participants.add(speaker)

        # Append to recovery file for crash protection
        self._append_to_recovery(entry)

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

            # Sort entries by timestamp to ensure correct chronological order
            # (mic and loopback may arrive out of order due to parallel processing)
            sorted_entries = sorted(self.entries, key=lambda e: e.timestamp)

            for entry in sorted_entries:
                if self.include_timestamps:
                    ts = self.format_timestamp(entry.timestamp)
                    lines.append(f"**[{ts}] {entry.speaker}**: {entry.text}")
                else:
                    lines.append(f"**{entry.speaker}**: {entry.text}")
                lines.append("")

        return "\n".join(lines)

    def save(self, filename: Optional[str] = None, notes_markdown: Optional[str] = None) -> Path:
        """Save transcript to file with optional notes section.

        After successful save, deletes the recovery file since data is now safe.
        """
        if filename is None:
            # Generate filename from date/time
            if self.meeting_start:
                filename = self.meeting_start.strftime("meeting_%Y%m%d_%H%M%S.md")
            else:
                filename = datetime.now().strftime("meeting_%Y%m%d_%H%M%S.md")

        filepath = self.output_dir / filename
        content = self.generate_markdown(notes_markdown=notes_markdown)

        # Atomic write: write to temp file then rename
        temp_path = filepath.with_suffix('.tmp')
        temp_path.write_text(content, encoding='utf-8')
        temp_path.replace(filepath)  # Atomic rename
        print(f"[Meeting Transcript] Saved to: {filepath}")

        # Delete recovery file after successful save - data is now safe on disk
        TranscriptWriter.delete_recovery_file()

        return filepath

    def get_recent_text(self, seconds: float = 30) -> str:
        """Get text from the last N seconds for copying."""
        if not self.entries:
            return ""

        cutoff = self.get_duration() - seconds
        # Sort by timestamp and filter to recent entries
        sorted_entries = sorted(self.entries, key=lambda e: e.timestamp)
        recent = [e for e in sorted_entries if e.timestamp >= cutoff]

        return "\n".join(f"{e.speaker}: {e.text}" for e in recent)

    def get_full_text(self) -> str:
        """Get all text without formatting."""
        # Sort by timestamp for correct chronological order
        sorted_entries = sorted(self.entries, key=lambda e: e.timestamp)
        return "\n".join(f"{e.speaker}: {e.text}" for e in sorted_entries)
