"""
Tests for transcript writer and crash recovery.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from meeting.transcript import TranscriptWriter, TranscriptEntry, RECOVERY_FILE


class TestTranscriptEntry:
    """Tests for TranscriptEntry dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        entry = TranscriptEntry(
            timestamp=10.5,
            speaker="Bryce",
            text="Hello world",
            source="mic"
        )
        result = entry.to_dict()
        assert result["timestamp"] == 10.5
        assert result["speaker"] == "Bryce"
        assert result["text"] == "Hello world"
        assert result["source"] == "mic"

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "timestamp": 10.5,
            "speaker": "Bryce",
            "text": "Hello world",
            "source": "mic"
        }
        entry = TranscriptEntry.from_dict(data)
        assert entry.timestamp == 10.5
        assert entry.speaker == "Bryce"
        assert entry.text == "Hello world"
        assert entry.source == "mic"

    def test_from_dict_default_source(self):
        """Should use default source if not provided."""
        data = {
            "timestamp": 10.5,
            "speaker": "Bryce",
            "text": "Hello world"
        }
        entry = TranscriptEntry.from_dict(data)
        assert entry.source == "mic"


class TestTranscriptWriter:
    """Tests for TranscriptWriter."""

    def test_add_entry(self, temp_dir):
        """Should add entries correctly."""
        writer = TranscriptWriter(output_dir=temp_dir)
        writer._recovery_enabled = False  # Disable recovery for unit tests
        writer.start_meeting()

        writer.add_entry(0.0, "Bryce", "Hello")
        writer.add_entry(5.0, "Calum", "Hi there")

        assert len(writer.entries) == 2
        assert "Bryce" in writer.participants
        assert "Calum" in writer.participants

    def test_add_entry_ignores_empty(self, temp_dir):
        """Should ignore empty text entries."""
        writer = TranscriptWriter(output_dir=temp_dir)
        writer._recovery_enabled = False
        writer.start_meeting()

        writer.add_entry(0.0, "Bryce", "")
        writer.add_entry(5.0, "Bryce", "   ")
        writer.add_entry(10.0, "Bryce", None)

        assert len(writer.entries) == 0

    def test_get_duration(self, temp_dir):
        """Should calculate duration from entries."""
        writer = TranscriptWriter(output_dir=temp_dir)
        writer._recovery_enabled = False
        writer.start_meeting()

        writer.add_entry(0.0, "Bryce", "Hello")
        writer.add_entry(30.0, "Calum", "Goodbye")

        assert writer.get_duration() == 30.0

    def test_get_duration_empty(self, temp_dir):
        """Should return 0 for empty transcript."""
        writer = TranscriptWriter(output_dir=temp_dir)
        writer._recovery_enabled = False
        writer.start_meeting()

        assert writer.get_duration() == 0

    def test_format_timestamp_minutes(self, temp_dir):
        """Should format timestamps correctly for minutes."""
        writer = TranscriptWriter(output_dir=temp_dir)
        assert writer.format_timestamp(65) == "01:05"
        assert writer.format_timestamp(0) == "00:00"
        assert writer.format_timestamp(3599) == "59:59"

    def test_format_timestamp_hours(self, temp_dir):
        """Should format timestamps correctly for hours."""
        writer = TranscriptWriter(output_dir=temp_dir)
        assert writer.format_timestamp(3600) == "01:00:00"
        assert writer.format_timestamp(3661) == "01:01:01"

    def test_generate_markdown(self, temp_dir):
        """Should generate valid markdown."""
        writer = TranscriptWriter(output_dir=temp_dir)
        writer._recovery_enabled = False
        writer.meeting_name = "Test Meeting"
        writer.start_meeting()

        writer.add_entry(0.0, "Bryce", "Hello everyone")
        writer.add_entry(5.0, "Calum", "Hi Bryce")

        md = writer.generate_markdown()

        assert "# Test Meeting" in md
        assert "**Date**:" in md
        assert "**Participants**:" in md
        assert "Bryce" in md
        assert "Calum" in md
        assert "Hello everyone" in md
        assert "Hi Bryce" in md

    def test_generate_markdown_with_notes(self, temp_dir):
        """Should include notes section in markdown."""
        writer = TranscriptWriter(output_dir=temp_dir)
        writer._recovery_enabled = False
        writer.meeting_name = "Test Meeting"
        writer.start_meeting()

        writer.add_entry(0.0, "Bryce", "Hello")

        notes = "## Notes\n- Item 1\n- Item 2"
        md = writer.generate_markdown(notes_markdown=notes)

        assert "## Notes" in md
        assert "- Item 1" in md

    def test_entries_sorted_by_timestamp(self, temp_dir):
        """Should sort entries by timestamp in output."""
        writer = TranscriptWriter(output_dir=temp_dir)
        writer._recovery_enabled = False
        writer.start_meeting()

        # Add entries out of order
        writer.add_entry(10.0, "Calum", "Second")
        writer.add_entry(0.0, "Bryce", "First")
        writer.add_entry(20.0, "Bryce", "Third")

        md = writer.generate_markdown()
        first_pos = md.find("First")
        second_pos = md.find("Second")
        third_pos = md.find("Third")

        assert first_pos < second_pos < third_pos

    def test_save_creates_file(self, temp_dir):
        """Should save transcript to file."""
        writer = TranscriptWriter(output_dir=temp_dir)
        writer._recovery_enabled = False
        writer.meeting_name = "Test Meeting"
        writer.start_meeting()

        writer.add_entry(0.0, "Bryce", "Hello")

        filepath = writer.save(filename="test.md")

        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "Hello" in content


class TestCrashRecovery:
    """Tests for crash recovery functionality."""

    def test_recovery_file_created(self, temp_dir):
        """Recovery file should be created when meeting starts."""
        # Use a custom recovery file path for testing
        import meeting.transcript as transcript_module
        original_recovery_file = transcript_module.RECOVERY_FILE
        test_recovery_file = temp_dir / ".test_recovery.jsonl"
        transcript_module.RECOVERY_FILE = test_recovery_file

        try:
            writer = TranscriptWriter(output_dir=temp_dir)
            writer.meeting_name = "Test Meeting"
            writer.start_meeting()

            assert test_recovery_file.exists()

            # Check header
            content = test_recovery_file.read_text()
            header = json.loads(content.strip().split('\n')[0])
            assert header["_type"] == "header"
            assert header["meeting_name"] == "Test Meeting"
        finally:
            transcript_module.RECOVERY_FILE = original_recovery_file

    def test_entries_appended_to_recovery(self, temp_dir):
        """Entries should be appended to recovery file."""
        import meeting.transcript as transcript_module
        original_recovery_file = transcript_module.RECOVERY_FILE
        test_recovery_file = temp_dir / ".test_recovery.jsonl"
        transcript_module.RECOVERY_FILE = test_recovery_file

        try:
            writer = TranscriptWriter(output_dir=temp_dir)
            writer.meeting_name = "Test Meeting"
            writer.start_meeting()

            writer.add_entry(0.0, "Bryce", "Hello")
            writer.add_entry(5.0, "Calum", "Hi")

            lines = test_recovery_file.read_text().strip().split('\n')
            assert len(lines) == 3  # header + 2 entries

            entry1 = json.loads(lines[1])
            assert entry1["speaker"] == "Bryce"
            assert entry1["text"] == "Hello"
        finally:
            transcript_module.RECOVERY_FILE = original_recovery_file

    def test_has_recovery_data(self, temp_dir, sample_recovery_data):
        """Should detect recovery data exists."""
        import meeting.transcript as transcript_module
        original_recovery_file = transcript_module.RECOVERY_FILE
        test_recovery_file = temp_dir / ".test_recovery.jsonl"
        transcript_module.RECOVERY_FILE = test_recovery_file

        try:
            # No file
            assert not TranscriptWriter.has_recovery_data()

            # Empty file
            test_recovery_file.write_text("")
            assert not TranscriptWriter.has_recovery_data()

            # Only header
            test_recovery_file.write_text('{"_type": "header"}\n')
            assert not TranscriptWriter.has_recovery_data()

            # Header + entries
            test_recovery_file.write_text(sample_recovery_data)
            assert TranscriptWriter.has_recovery_data()
        finally:
            transcript_module.RECOVERY_FILE = original_recovery_file

    def test_load_from_recovery(self, temp_dir, sample_recovery_data):
        """Should load transcript from recovery file."""
        import meeting.transcript as transcript_module
        original_recovery_file = transcript_module.RECOVERY_FILE
        test_recovery_file = temp_dir / ".test_recovery.jsonl"
        transcript_module.RECOVERY_FILE = test_recovery_file

        try:
            test_recovery_file.write_text(sample_recovery_data)

            writer = TranscriptWriter.load_from_recovery()

            assert writer is not None
            assert writer.meeting_name == "Test Meeting"
            assert len(writer.entries) == 3
            assert "Bryce" in writer.participants
            assert "Calum" in writer.participants
        finally:
            transcript_module.RECOVERY_FILE = original_recovery_file

    def test_delete_recovery_file(self, temp_dir):
        """Should delete recovery file after save."""
        import meeting.transcript as transcript_module
        original_recovery_file = transcript_module.RECOVERY_FILE
        test_recovery_file = temp_dir / ".test_recovery.jsonl"
        transcript_module.RECOVERY_FILE = test_recovery_file

        try:
            writer = TranscriptWriter(output_dir=temp_dir)
            writer.meeting_name = "Test Meeting"
            writer.start_meeting()
            writer.add_entry(0.0, "Bryce", "Hello")

            assert test_recovery_file.exists()

            writer.save(filename="test.md")

            assert not test_recovery_file.exists()
        finally:
            transcript_module.RECOVERY_FILE = original_recovery_file
