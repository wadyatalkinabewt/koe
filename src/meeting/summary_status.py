"""
Status file infrastructure for tracking summarization progress.

Status files allow the detached subprocess to communicate progress to the parent
process (or any other process) without blocking.
"""

import json
import time
import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


def _get_status_dir() -> Path:
    """Get the centralized status files directory (creates if needed)."""
    # Koe root directory
    koe_root = Path(__file__).parent.parent.parent
    status_dir = koe_root / ".summary_status"
    status_dir.mkdir(exist_ok=True)
    return status_dir


def _get_status_file_path(transcript_path: Path) -> Path:
    """
    Get the status file path for a given transcript.

    Uses a hash of the full transcript path to ensure uniqueness while keeping
    all status files in a centralized folder.

    Args:
        transcript_path: Path to the transcript file

    Returns:
        Path to the status file in .summary_status/ folder
    """
    # Hash the full transcript path to get a unique identifier
    path_hash = hashlib.sha256(str(transcript_path.resolve()).encode('utf-8')).hexdigest()[:16]

    # Include transcript name for easier debugging (sanitize first)
    safe_name = transcript_path.stem.replace(' ', '_')[:50]  # Limit length

    status_dir = _get_status_dir()
    return status_dir / f"{safe_name}_{path_hash}.json"


@dataclass
class SummaryStatus:
    """Status of a summary generation process."""
    status: str  # "in_progress", "complete", "failed"
    stage: str  # Human-readable progress message
    progress_percent: int = 0  # 0-100
    summary_path: Optional[str] = None  # Path to summary when complete
    error: Optional[str] = None  # Error message if failed
    pid: Optional[int] = None  # Process ID
    timestamp: float = 0.0  # Last update time
    transcript_path: Optional[str] = None  # Source transcript
    started_at: float = 0.0  # When subprocess started

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SummaryStatus':
        """Create from dictionary."""
        return cls(**data)


class SummaryStatusWriter:
    """Writes status updates to a JSON file atomically."""

    def __init__(self, transcript_path: Path):
        """
        Initialize status writer.

        Args:
            transcript_path: Path to the transcript being summarized
        """
        self.transcript_path = transcript_path
        # Status file stored in centralized .summary_status/ folder
        self.status_file = _get_status_file_path(transcript_path)
        self.pid = os.getpid()
        self.started_at = time.time()

    def write(self, status: str, stage: str, progress_percent: int = 0,
              summary_path: Optional[Path] = None, error: Optional[str] = None):
        """
        Write status update atomically.

        Args:
            status: "in_progress", "complete", or "failed"
            stage: Human-readable progress message
            progress_percent: 0-100
            summary_path: Path to summary file when complete
            error: Error message if failed
        """
        status_obj = SummaryStatus(
            status=status,
            stage=stage,
            progress_percent=progress_percent,
            summary_path=str(summary_path) if summary_path else None,
            error=error,
            pid=self.pid,
            timestamp=time.time(),
            transcript_path=str(self.transcript_path),
            started_at=self.started_at
        )

        # Write to temp file, then rename (atomic on most filesystems)
        temp_file = self.status_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(status_obj.to_dict(), f, indent=2)

            # Atomic rename
            temp_file.replace(self.status_file)
        except Exception as e:
            # If atomic write fails, try direct write as fallback
            try:
                with open(self.status_file, 'w', encoding='utf-8') as f:
                    json.dump(status_obj.to_dict(), f, indent=2)
            except Exception:
                pass  # Give up silently - parent will detect stale status

    def cleanup(self):
        """Remove status file (call on successful completion)."""
        try:
            if self.status_file.exists():
                self.status_file.unlink()
        except Exception:
            pass  # Non-critical if cleanup fails


class SummaryStatusReader:
    """Reads status updates from a JSON file."""

    STALE_TIMEOUT = 600  # 10 minutes

    def __init__(self, transcript_path: Path):
        """
        Initialize status reader.

        Args:
            transcript_path: Path to the transcript being summarized
        """
        self.transcript_path = transcript_path
        # Status file stored in centralized .summary_status/ folder
        self.status_file = _get_status_file_path(transcript_path)

    def read(self) -> Optional[SummaryStatus]:
        """
        Read current status.

        Returns:
            SummaryStatus object, or None if file doesn't exist or is corrupt
        """
        if not self.status_file.exists():
            return None

        try:
            with open(self.status_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            status = SummaryStatus.from_dict(data)

            # Check for staleness (process may have crashed)
            if status.status == "in_progress":
                age = time.time() - status.timestamp
                if age > self.STALE_TIMEOUT:
                    # Mark as failed due to timeout
                    status.status = "failed"
                    status.error = "Process timed out or crashed"

            return status

        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupt file - treat as failed
            return SummaryStatus(
                status="failed",
                stage="Status file corrupted",
                error="Could not read status file",
                timestamp=time.time()
            )
        except Exception:
            # Other errors (file locked, permissions, etc.)
            return None

    def wait_for_completion(self, timeout: float = 300, poll_interval: float = 2) -> Optional[SummaryStatus]:
        """
        Block until summarization completes or times out.

        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check status file

        Returns:
            Final SummaryStatus, or None if timed out
        """
        start_time = time.time()

        while True:
            status = self.read()

            if status and status.status in ("complete", "failed"):
                return status

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return None

            time.sleep(poll_interval)

    def cleanup(self):
        """Remove status file (call after reading final status)."""
        try:
            if self.status_file.exists():
                self.status_file.unlink()
        except Exception:
            pass  # Non-critical if cleanup fails
