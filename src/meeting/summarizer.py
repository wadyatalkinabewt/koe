"""
AI summarization client using OpenRouter.

Default model is google/gemini-3.6-flash via OpenRouter. Provider selection is
left to OpenRouter so account privacy policies and available endpoints are
respected. To switch models, change SummarizerClient.MODEL.
"""

import time
import os
from typing import Optional, Callable

import requests
from dotenv import load_dotenv

from paths import env_path


class SummarizerClient:
    """Client for AI-powered meeting summarization via OpenRouter."""

    MODEL = "google/gemini-3.6-flash"
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 2.0  # seconds
    REQUEST_TIMEOUT = 300  # 5 minutes per attempt
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    REQUIRED_SECTIONS = (
        "## Summary",
        "## Key Decisions",
        "## Topics Discussed",
        "## Action Items",
        "## Open Questions",
    )

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize summarizer client.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        """
        load_dotenv(env_path(), override=True)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment or provided")

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
        prompt = self._build_prompt(transcript_content)

        body = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                if attempt > 0:
                    delay = self.INITIAL_RETRY_DELAY * (2 ** (attempt - 1))
                    if status_callback:
                        status_callback(f"Retrying in {delay:.0f}s...")
                    time.sleep(delay)

                response = requests.post(
                    self.OPENROUTER_URL,
                    headers=headers,
                    json=body,
                    timeout=self.REQUEST_TIMEOUT,
                )

                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    # Don't retry on client errors (4xx other than 429)
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        break
                    continue

                data = response.json()
                if data.get("error"):
                    last_error = f"API error: {data['error']}"
                    continue

                summary = data["choices"][0]["message"]["content"]
                missing_sections = [
                    section
                    for section in self.REQUIRED_SECTIONS
                    if section not in summary
                ]
                if missing_sections:
                    last_error = (
                        "Incomplete summary response; missing "
                        + ", ".join(missing_sections)
                    )
                    continue

                if status_callback:
                    status_callback("Summary generated successfully")

                return summary

            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout: {e}"
            except requests.exceptions.ConnectionError as e:
                last_error = f"Network error: {e}"
            except Exception as e:
                last_error = f"Unexpected error: {e}"

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
        fallback_title = None

        for line in lines:
            line = line.strip()

            # Host notes may precede the transcript. Prefer the meeting's H1
            # over the generated "Notes — ..." heading, while retaining a
            # fallback for documents that contain notes only.
            if line.startswith('# '):
                candidate = line[2:].strip()
                fallback_title = fallback_title or candidate
                if not candidate.casefold().startswith("notes —"):
                    metadata["title"] = candidate

            # Date: **Date**: value
            if line.startswith('**Date**:'):
                metadata["date"] = line.replace('**Date**:', '').strip()

            # Duration: **Duration**: value
            if line.startswith('**Duration**:'):
                metadata["duration"] = line.replace('**Duration**:', '').strip()

            # Participants: **Participants**: value
            if line.startswith('**Participants**:'):
                metadata["participants"] = line.replace('**Participants**:', '').strip()

            if all(metadata.values()):
                break

        metadata["title"] = metadata["title"] or fallback_title

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

        # Build metadata header for output format (proper markdown hierarchy)
        # Format: # Title - DD Mon YYYY
        #         Duration: X min | Participants: A, B, C
        title_line = ""
        if metadata["title"]:
            # Try to format date nicely if available
            date_str = ""
            if metadata["date"]:
                # Try to parse and reformat date (e.g., "2026-01-22 12:34" -> "22 Jan 2026")
                try:
                    from datetime import datetime
                    date_part = metadata["date"].split()[0]  # Get just the date part
                    dt = datetime.strptime(date_part, "%Y-%m-%d")
                    date_str = f" - {dt.strftime('%d %b %Y')}"
                except:
                    date_str = f" - {metadata['date']}"
            title_line = f"# {metadata['title']}{date_str}"

        info_parts = []
        if metadata["duration"]:
            info_parts.append(f"Duration: {metadata['duration']}")
        if metadata["participants"]:
            info_parts.append(f"Participants: {metadata['participants']}")
        info_line = " | ".join(info_parts) if info_parts else ""

        if title_line:
            metadata_header = title_line + "\n" + info_line + "\n" if info_line else title_line + "\n"
        else:
            metadata_header = ""

        return f"""You are a meeting summarization assistant. Your task is to create a comprehensive, accurate summary of the following meeting transcript.

**CRITICAL RULES (Anti-Hallucination):**
1. Only use information from the transcript - never add external knowledge or assumptions
2. Preserve exact technical terms, names, and numbers - don't paraphrase domain-specific terminology
3. Keep speaker labels as-is - if transcript shows "Speaker 1", use "Speaker 1" (don't guess real names)
4. Trust the meeting notes - the notes section (Agenda/Notes/Action Items) is ground truth written by the meeting host
5. Don't infer unspoken intent - if something wasn't explicitly said, don't add it
6. Preserve uncertainty - if speakers were uncertain or debating, reflect that
7. Don't add generic business advice - no "best practices" or recommendations beyond what was discussed

**OUTPUT FORMAT (markdown hierarchy: H1 title, H2 sections, H5 subtopics/owners):**

{metadata_header}
---

## Summary
[No more than 2 concise paragraphs and roughly 120 words total. Surface only
the meeting's highest-value outcomes and current state. Move supporting detail
into Topics Discussed. Tastefully bold 2-4 short, high-signal phrases using
**bold Markdown**; do not bold full sentences or routine context.]

---

## Key Decisions
[Bullet list of concrete decisions made. If none, write "No formal decisions recorded."]

---

## Topics Discussed

##### Example Topic Name
Brief description of what was discussed about this topic.

##### Another Topic
Description here.

---

## Action Items

##### Alex
- Task one
- Task two

##### Sash
- Task three

##### Alex & Sash
- Shared task

If there are action items in the meeting notes, prioritize those as authoritative.
If transcript mentions additional tasks, add them.
If no action items exist, write "No action items assigned."

---

## Open Questions
[Bullet list of unresolved questions or topics requiring follow-up. If none, write "No open questions."]

**FORMATTING RULES:**
- Use --- horizontal rule before each ## H2 section heading
- Use ##### H5 for topic names under Topics Discussed
- Use ##### H5 for owner names under Action Items (for shared tasks use ##### Name & Name)
- Empty line before each heading
- Content starts immediately after heading (no empty line after heading)

**MEETING TRANSCRIPT:**

{transcript_content}

**Generate the summary now, following the format above exactly. Start with the H1 title line:**"""
