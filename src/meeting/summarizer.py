"""
AI summarization client using Anthropic's Claude API.

Uses Claude Sonnet 4.5 to generate comprehensive meeting summaries with
anti-hallucination safeguards and retry logic.
"""

import time
import os
from typing import Optional, Callable
from pathlib import Path


class SummarizerClient:
    """Client for AI-powered meeting summarization."""

    MODEL = "claude-sonnet-4-5-20250929"
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 2.0  # seconds
    TIMEOUT = 300  # 5 minutes total timeout

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize summarizer client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or provided")

        # Lazy import anthropic to avoid import errors if not installed
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic>=0.40.0"
            )

    def summarize(
        self,
        transcript_content: str,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate a comprehensive summary of a meeting transcript.

        Args:
            transcript_content: Full transcript markdown content
            status_callback: Optional callback for status updates

        Returns:
            Generated summary in markdown format

        Raises:
            Exception: If summarization fails after all retries
        """
        # Status updates handled by parent (summarize_detached.py)
        # Shows user-friendly messages: "Analyzing transcript...", "Generating summary...", etc.

        prompt = self._build_prompt(transcript_content)

        # Retry with exponential backoff
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                if attempt > 0:
                    delay = self.INITIAL_RETRY_DELAY * (2 ** (attempt - 1))
                    if status_callback:
                        status_callback(f"Retrying in {delay:.0f}s...")
                    time.sleep(delay)

                # Don't update status for each attempt - keeps UI clean

                response = self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=4096,
                    temperature=0.0,  # Deterministic for consistency
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                # Extract text from response
                summary = response.content[0].text

                if status_callback:
                    status_callback("Summary generated successfully")

                return summary

            except self.anthropic.APIConnectionError as e:
                last_error = f"Network error: {e}"
            except self.anthropic.RateLimitError as e:
                last_error = f"Rate limit exceeded: {e}"
            except self.anthropic.APIStatusError as e:
                last_error = f"API error ({e.status_code}): {e.message}"
            except Exception as e:
                last_error = f"Unexpected error: {e}"

        # All retries exhausted
        raise Exception(f"Summarization failed after {self.MAX_RETRIES} attempts: {last_error}")

    def _extract_metadata(self, transcript_content: str) -> dict:
        """
        Extract metadata (title, date, duration, participants) from transcript.

        Args:
            transcript_content: Full transcript markdown

        Returns:
            Dict with title, date, duration, participants (all optional)
        """
        metadata = {
            "title": None,
            "date": None,
            "duration": None,
            "participants": None
        }

        lines = transcript_content.split('\n')

        for line in lines[:20]:  # Only check first 20 lines for metadata
            line = line.strip()

            # Title: first H1 heading
            if line.startswith('# ') and metadata["title"] is None:
                metadata["title"] = line[2:].strip()

            # Date: **Date**: value
            if line.startswith('**Date**:'):
                metadata["date"] = line.replace('**Date**:', '').strip()

            # Duration: **Duration**: value
            if line.startswith('**Duration**:'):
                metadata["duration"] = line.replace('**Duration**:', '').strip()

            # Participants: **Participants**: value
            if line.startswith('**Participants**:'):
                metadata["participants"] = line.replace('**Participants**:', '').strip()

        return metadata

    def _build_prompt(self, transcript_content: str) -> str:
        """
        Build the summarization prompt with anti-hallucination guidelines.

        Args:
            transcript_content: Full transcript markdown

        Returns:
            Complete prompt string
        """
        # Extract metadata from transcript
        metadata = self._extract_metadata(transcript_content)

        # Build metadata header for output format
        metadata_lines = []
        if metadata["title"]:
            metadata_lines.append(f"# {metadata['title']}")
        if metadata["date"]:
            metadata_lines.append(f"**Date**: {metadata['date']}")
        if metadata["duration"]:
            metadata_lines.append(f"**Duration**: {metadata['duration']}")
        if metadata["participants"]:
            metadata_lines.append(f"**Participants**: {metadata['participants']}")

        if metadata_lines:
            metadata_header = "\n".join(metadata_lines) + "\n\n---\n"
        else:
            metadata_header = ""

        return f"""You are a meeting summarization assistant. Your task is to create a comprehensive, accurate summary of the following meeting transcript.

**CRITICAL RULES (Anti-Hallucination):**
1. **Only use information from the transcript** - never add external knowledge or assumptions
2. **Preserve exact technical terms, names, and numbers** - don't paraphrase domain-specific terminology
3. **Keep speaker labels as-is** - if transcript shows "Speaker 1", use "Speaker 1" (don't guess real names)
4. **Trust the meeting notes** - the notes section (Agenda/Notes/Action Items) is ground truth written by the meeting host
5. **Don't infer unspoken intent** - if something wasn't explicitly said, don't add it
6. **Preserve uncertainty** - if speakers were uncertain or debating, reflect that
7. **Don't add generic business advice** - no "best practices" or recommendations beyond what was discussed

**OUTPUT FORMAT:**

{metadata_header}## Summary
[2-4 paragraphs capturing the meeting's purpose, key outcomes, and overall context]

## Key Decisions
[Bullet list of concrete decisions made. If none, write "No formal decisions recorded."]

## Topics Discussed
[Natural breakdown of topics with 1-2 sentence summaries for each. Use clear headings.]

## Action Items & Follow-ups
[Table format with columns: Action | Owner | Deadline (if mentioned)]

If there are action items in the meeting notes, prioritize those as authoritative.
If transcript mentions additional tasks, add them.
If no action items exist, write "No action items assigned."

## Open Questions
[Unresolved questions or topics requiring follow-up. If none, write "No open questions."]

---

**MEETING TRANSCRIPT:**

{transcript_content}

---

**Generate the summary now, following the format above. Start with the metadata header (title, date, duration, participants) exactly as shown:**"""

    @staticmethod
    def calculate_mirrored_path(transcript_path: Path) -> Path:
        """
        Calculate the mirrored path in Summaries/ folder.

        Example:
            Transcripts/Standups/26_01_21_Daily.md
            → Summaries/Standups/26_01_21_Daily.md

        Args:
            transcript_path: Path to transcript file

        Returns:
            Path where summary should be saved
        """
        # Find the "Transcripts" folder in the path
        parts = transcript_path.parts

        # Find index of "Transcripts" folder
        transcripts_idx = None
        for i, part in enumerate(parts):
            if part == "Transcripts":
                transcripts_idx = i
                break

        if transcripts_idx is None:
            # Fallback: just save in same directory with .summary.md suffix
            return transcript_path.with_suffix('.summary.md')

        # Replace "Transcripts" with "Summaries"
        new_parts = list(parts)
        new_parts[transcripts_idx] = "Summaries"

        # Change extension to .summary.md
        summary_path = Path(*new_parts).with_suffix('.summary.md')

        # Ensure parent directory exists
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        return summary_path
