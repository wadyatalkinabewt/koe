"""
Detached subprocess for generating meeting summaries.

This runs independently from the parent process (Scribe window can be closed).
Communicates progress via status file in same directory as transcript.

Usage:
    python -m src.meeting.summarize_detached "C:/path/to/transcript.md"
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Load .env FIRST (before importing modules that need API keys)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from .summary_status import SummaryStatusWriter
from .summarizer import SummarizerClient


# Setup error logging
ERROR_LOG = Path(__file__).parent.parent.parent / "summarization_errors.log"


def setup_logging():
    """Setup logging to file (for errors only)."""
    logging.basicConfig(
        filename=str(ERROR_LOG),
        level=logging.ERROR,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def log_error(transcript_name: str, error: str):
    """Log error to file."""
    logging.error(f"{transcript_name}: {error}")


def main():
    """Entry point for detached summarization process."""
    setup_logging()

    if len(sys.argv) < 2:
        print("Usage: python -m src.meeting.summarize_detached <transcript_path>")
        sys.exit(1)

    transcript_path = Path(sys.argv[1])

    if not transcript_path.exists():
        print(f"Error: Transcript file not found: {transcript_path}")
        sys.exit(1)

    status_writer = SummaryStatusWriter(transcript_path)

    try:
        # Initialize status
        status_writer.write(
            status="in_progress",
            stage="Loading transcript...",
            progress_percent=0
        )

        # Load transcript
        try:
            transcript_content = transcript_path.read_text(encoding='utf-8')
        except Exception as e:
            error_msg = f"Failed to read transcript: {e}"
            status_writer.write(
                status="failed",
                stage="Failed",
                error=error_msg
            )
            log_error(transcript_path.name, error_msg)
            sys.exit(1)

        # Initialize summarizer
        status_writer.write(
            status="in_progress",
            stage="Analyzing transcript...",
            progress_percent=10
        )

        try:
            summarizer = SummarizerClient()
        except Exception as e:
            error_msg = f"Failed to initialize API client: {e}"
            status_writer.write(
                status="failed",
                stage="Failed",
                error=error_msg
            )
            log_error(transcript_path.name, error_msg)
            sys.exit(1)

        # Generate summary
        def update_status(stage: str):
            """Callback for status updates during summarization."""
            # Map stages to progress percentages
            progress_map = {
                "Generating summary...": 20,
                "Summary generated successfully": 80
            }
            # For retry messages, extract attempt number
            if "Retrying" in stage or "attempt" in stage:
                progress = 30
            else:
                progress = progress_map.get(stage, 50)

            status_writer.write(
                status="in_progress",
                stage=stage,
                progress_percent=progress
            )

        try:
            summary = summarizer.summarize(
                transcript_content,
                status_callback=update_status
            )
        except Exception as e:
            error_msg = str(e)
            status_writer.write(
                status="failed",
                stage="Failed",
                error=error_msg
            )
            log_error(transcript_path.name, error_msg)
            sys.exit(1)

        # Save summary
        status_writer.write(
            status="in_progress",
            stage="Saving summary...",
            progress_percent=90
        )

        try:
            summary_path = SummarizerClient.calculate_mirrored_path(transcript_path)
            summary_path.write_text(summary, encoding='utf-8')
        except Exception as e:
            error_msg = f"Failed to save summary: {e}"
            status_writer.write(
                status="failed",
                stage="Failed",
                error=error_msg
            )
            log_error(transcript_path.name, error_msg)
            sys.exit(1)

        # Success!
        status_writer.write(
            status="complete",
            stage="Complete",
            progress_percent=100,
            summary_path=summary_path
        )

        # NOTE: Don't clean up status file - parent needs to read it
        # The parent will detect completion and stop polling
        # Status files are small and will be overwritten on next summarization

        sys.exit(0)

    except Exception as e:
        # Catch-all for unexpected errors
        error_msg = f"Unexpected error: {e}"
        try:
            status_writer.write(
                status="failed",
                stage="Failed",
                error=error_msg
            )
        except:
            pass  # If status write fails, nothing we can do
        log_error(transcript_path.name, error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
