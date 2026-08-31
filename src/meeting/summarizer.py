"""Contextual speaker resolution and meeting summarization via OpenRouter.

Default model is google/gemini-3.7-flash via OpenRouter. Provider selection is
left to OpenRouter so account privacy policies and available endpoints are
respected. To switch models, change SummarizerClient.MODEL.
"""

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any, Callable, Optional

import requests
from dotenv import load_dotenv

from paths import env_path


_GENERIC_SPEAKER = re.compile(r"^Speaker\s+\d+$", flags=re.IGNORECASE)
_TRANSCRIPT_UTTERANCE = re.compile(
    r"^\*\*\[[^\]]+\]\s+(.+?)\*\*:\s*(.+)$"
)
_SELF_IDENTIFICATION = re.compile(
    r"\b(?:i\s+am|i\s*m|my\s+name\s+is)\s+(.+)$",
    flags=re.IGNORECASE,
)
_ROLE_LABELS = {
    "ot",
    "client",
    "customer",
    "facilitator",
    "interpreter",
    "lawyer",
    "manager",
    "representative",
    "staff",
    "support",
    "team",
}


@dataclass(frozen=True)
class MeetingAnalysis:
    """Validated model output used by both meeting documents."""

    summary: str
    speaker_mapping: dict[str, str]


def _normalise_evidence(text: str) -> str:
    """Normalise transcript excerpts for conservative exact-quote checks."""
    return re.sub(r"[\W_]+", " ", str(text).casefold()).strip()


def _speaker_labels_from_transcript(transcript_content: str) -> list[str]:
    """Return labels in first-utterance order from rendered transcript Markdown."""
    labels: list[str] = []
    for line in transcript_content.splitlines():
        match = _TRANSCRIPT_UTTERANCE.match(line.strip())
        if not match:
            continue
        label = match.group(1).strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def _transcript_utterances(transcript_content: str) -> list[dict[str, Any]]:
    """Extract auditable label/text pairs from rendered transcript Markdown."""
    utterances: list[dict[str, Any]] = []
    for line in transcript_content.splitlines():
        match = _TRANSCRIPT_UTTERANCE.match(line.strip())
        if not match:
            continue
        utterances.append(
            {
                "label": match.group(1).strip(),
                "text": match.group(2).strip(),
                "normalised": _normalise_evidence(match.group(2)),
            }
        )
    return utterances


def _label_core_tokens(label: str) -> list[str]:
    """Return the identity-bearing portion of a proposed display label."""
    without_qualifier = re.sub(r"\([^)]*\)", " ", label)
    return [
        token
        for token in _normalise_evidence(without_qualifier).split()
        if len(token) >= 2
    ]


def _is_role_label(label: str) -> bool:
    tokens = _label_core_tokens(label)
    return bool(tokens) and (
        label.strip().isupper()
        or all(token in _ROLE_LABELS for token in tokens)
    )


def replace_speaker_labels(text: str, mapping: dict[str, str]) -> str:
    """Replace exact generic labels without touching substrings or ordinary words."""
    rewritten = text
    for source, target in sorted(
        mapping.items(), key=lambda item: len(item[0]), reverse=True
    ):
        rewritten = re.sub(
            rf"(?<![\w]){re.escape(source)}(?![\w])",
            lambda _match, replacement=target: replacement,
            rewritten,
            flags=re.IGNORECASE,
        )
    return rewritten


class SummarizerClient:
    """Client for AI-powered meeting summarization via OpenRouter."""

    MODEL = "google/gemini-3.7-flash"
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
        """Generate only the summary for callers that do not need identity data."""
        return self.analyze(
            transcript_content,
            status_callback=status_callback,
        ).summary

    def analyze(
        self,
        transcript_content: str,
        speaker_labels: Optional[list[str]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> MeetingAnalysis:
        """Generate a summary and a conservatively validated speaker map.

        Only generic labels are eligible for contextual inference. The model's
        evidence quotes are checked against the labelled transcript before a
        proposed name can affect either meeting document.
        """
        labels = list(speaker_labels or _speaker_labels_from_transcript(transcript_content))
        prompt = self._build_prompt(transcript_content, speaker_labels=labels)

        body = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            # Fail closed instead of routing meeting text to a provider endpoint
            # that retains prompts or responses.
            "provider": {"zdr": True},
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

                raw_content = data["choices"][0]["message"]["content"]
                try:
                    payload = self._decode_model_payload(raw_content)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    last_error = f"Invalid structured summary response: {exc}"
                    continue

                summary = payload["summary_markdown"]
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

                speaker_mapping = self._validate_speaker_inferences(
                    payload["speaker_inferences"],
                    transcript_content,
                    labels,
                )
                summary = replace_speaker_labels(summary, speaker_mapping)

                if status_callback:
                    if speaker_mapping:
                        status_callback(
                            "Summary generated; "
                            f"identified {len(speaker_mapping)} speaker"
                            f"{'s' if len(speaker_mapping) != 1 else ''}"
                        )
                    else:
                        status_callback("Summary generated successfully")

                return MeetingAnalysis(
                    summary=summary,
                    speaker_mapping=speaker_mapping,
                )

            except requests.exceptions.Timeout as e:
                last_error = f"Request timeout: {e}"
            except requests.exceptions.ConnectionError as e:
                last_error = f"Network error: {e}"
            except Exception as e:
                last_error = f"Unexpected error: {e}"

        raise Exception(f"Summarization failed after {self.MAX_RETRIES} attempts: {last_error}")

    @staticmethod
    def _decode_model_payload(raw_content: Any) -> dict[str, Any]:
        """Decode the required JSON envelope and reject ambiguous output shapes."""
        if not isinstance(raw_content, str):
            raise TypeError("message content is not text")
        content = raw_content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, count=1)
            content = re.sub(r"\s*```$", "", content, count=1)
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise TypeError("top-level response must be an object")
        summary = payload.get("summary_markdown")
        inferences = payload.get("speaker_inferences")
        if not isinstance(summary, str) or not summary.strip():
            raise TypeError("summary_markdown must be non-empty text")
        if not isinstance(inferences, list):
            raise TypeError("speaker_inferences must be a list")
        return {
            "summary_markdown": summary.strip(),
            "speaker_inferences": inferences,
        }

    @staticmethod
    def _validate_speaker_inferences(
        proposed: list[Any],
        transcript_content: str,
        speaker_labels: list[str],
    ) -> dict[str, str]:
        """Accept only high-confidence mappings backed by transcript excerpts."""
        generic_by_key = {
            label.casefold(): label
            for label in speaker_labels
            if _GENERIC_SPEAKER.fullmatch(label.strip())
        }
        trusted_targets = {
            label.casefold()
            for label in speaker_labels
            if not _GENERIC_SPEAKER.fullmatch(label.strip())
        }
        if not generic_by_key:
            return {}

        utterances = _transcript_utterances(transcript_content)
        transcript_normalised = _normalise_evidence(transcript_content)
        accepted: list[tuple[str, str]] = []
        used_sources: set[str] = set()
        used_named_targets: set[str] = set()

        for item in proposed:
            if not isinstance(item, dict):
                continue
            source_raw = str(item.get("source_label") or "").strip()
            target = str(item.get("proposed_label") or "").strip()
            confidence = str(item.get("confidence") or "").strip().casefold()
            source = generic_by_key.get(source_raw.casefold())
            if (
                not source
                or source.casefold() in used_sources
                or confidence != "high"
                or not target
                or len(target) > 80
                or re.search(r"[\r\n<>*_`\[\]{}#|]", target)
                or _GENERIC_SPEAKER.fullmatch(target)
                or target.casefold() in trusted_targets
            ):
                continue

            core_tokens = _label_core_tokens(target)
            if not core_tokens or not any(
                re.search(rf"\b{re.escape(token)}\b", transcript_normalised)
                for token in core_tokens
            ):
                continue

            evidence = item.get("evidence")
            if not isinstance(evidence, list):
                continue
            verified: list[dict[str, Any]] = []
            seen_quotes: set[tuple[str, str]] = set()
            for evidence_item in evidence:
                if not isinstance(evidence_item, dict):
                    continue
                evidence_label = str(evidence_item.get("speaker_label") or "").strip()
                quote = str(evidence_item.get("exact_quote") or "").strip()
                kind = str(evidence_item.get("kind") or "").strip().casefold()
                quote_normalised = _normalise_evidence(quote)
                quote_key = (evidence_label.casefold(), quote_normalised)
                if len(quote_normalised) < 8 or quote_key in seen_quotes:
                    continue
                for index, utterance in enumerate(utterances):
                    if (
                        utterance["label"].casefold() == evidence_label.casefold()
                        and quote_normalised in utterance["normalised"]
                    ):
                        verified.append(
                            {
                                "label": utterance["label"],
                                "quote": quote_normalised,
                                "kind": kind,
                                "index": index,
                            }
                        )
                        seen_quotes.add(quote_key)
                        break

            source_evidence = [
                item
                for item in verified
                if item["label"].casefold() == source.casefold()
            ]
            name_evidence = [
                item
                for item in verified
                if any(
                    re.search(rf"\b{re.escape(token)}\b", item["quote"])
                    for token in core_tokens
                )
            ]
            self_identified = any(
                item["kind"] == "self_identification"
                and _SELF_IDENTIFICATION.search(item["quote"])
                and any(
                    re.search(rf"\b{re.escape(token)}\b", item["quote"])
                    for token in core_tokens
                )
                for item in source_evidence
            )
            nearby_address_response = any(
                abs(name_item["index"] - source_item["index"]) <= 3
                for name_item in name_evidence
                for source_item in source_evidence
                if name_item["label"].casefold() != source.casefold()
            )
            if not self_identified and not (
                len(verified) >= 2
                and source_evidence
                and name_evidence
                and nearby_address_response
            ):
                continue

            role_target = _is_role_label(target)
            target_key = target.casefold()
            if not role_target and target_key in used_named_targets:
                continue
            accepted.append((source, target))
            used_sources.add(source.casefold())
            if not role_target:
                used_named_targets.add(target_key)

        role_counts: dict[str, int] = {}
        for _source, target in accepted:
            if _is_role_label(target):
                role_counts[target.casefold()] = role_counts.get(target.casefold(), 0) + 1
        role_indexes: dict[str, int] = {}
        mapping: dict[str, str] = {}
        for source, target in accepted:
            target_key = target.casefold()
            if _is_role_label(target) and role_counts[target_key] > 1:
                role_indexes[target_key] = role_indexes.get(target_key, 0) + 1
                target = f"{target} {role_indexes[target_key]}"
            mapping[source] = target
        return mapping

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

    def _build_prompt(
        self,
        transcript_content: str,
        speaker_labels: Optional[list[str]] = None,
    ) -> str:
        """
        Build the summarization prompt with anti-hallucination guidelines.

        Args:
            transcript_content: Full transcript markdown
            speaker_labels: Final deterministic labels present in the transcript

        Returns:
            Complete prompt string
        """
        # Extract metadata from transcript
        metadata = self._extract_metadata(transcript_content)
        labels = list(speaker_labels or _speaker_labels_from_transcript(transcript_content))
        generic_labels = [
            label for label in labels if _GENERIC_SPEAKER.fullmatch(label.strip())
        ]
        trusted_labels = [
            label for label in labels if not _GENERIC_SPEAKER.fullmatch(label.strip())
        ]
        eligible_line = ", ".join(generic_labels) if generic_labels else "None"
        trusted_line = ", ".join(trusted_labels) if trusted_labels else "None"

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

        return f"""You are a meeting analysis assistant. Your tasks are to identify generic speaker labels only when the transcript contains strong evidence, and to create a comprehensive, accurate meeting summary.

**CRITICAL RULES (Anti-Hallucination):**
1. Only use information from the transcript - never add external knowledge or assumptions
2. Preserve exact technical terms, names, and numbers - don't paraphrase domain-specific terminology
3. Only these generic labels are eligible for inference: {eligible_line}
4. These labels are already trusted and must never be renamed: {trusted_line}
5. Infer an exact person only from explicit self-identification, or direct address plus a nearby response; require repeated contextual confirmation when one exchange is ambiguous
6. Preserve spelling from the transcript metadata, meeting notes, or exact dialogue; never infer identity from voice, accent, gender, job stereotypes, or external knowledge
7. If an exact name is not supported but a role or organisation is explicit, a role label such as "OT" is allowed; otherwise omit the inference and keep Speaker N
8. Every proposed inference must be high confidence and include exact transcript quotes with their original speaker labels; omit weak or ambiguous proposals
9. In summary_markdown, continue using the original labels such as "Speaker 1" for inferred speakers. Koe will replace only mappings that pass deterministic validation
10. Trust the meeting notes - the notes section (Agenda/Notes/Action Items) is ground truth written by the meeting host
11. Don't infer unspoken intent - if something wasn't explicitly said, don't add it
12. Preserve uncertainty - if speakers were uncertain or debating, reflect that
13. Don't add generic business advice - no "best practices" or recommendations beyond what was discussed

**RESPONSE ENVELOPE:**
Return one valid JSON object and nothing else. Do not use Markdown fences around the JSON.

{{
  "speaker_inferences": [
    {{
      "source_label": "Speaker 1",
      "proposed_label": "Exact Name (Organisation)",
      "confidence": "high",
      "evidence": [
        {{
          "kind": "direct_address",
          "speaker_label": "Trusted Name",
          "exact_quote": "An exact excerpt from that speaker's transcript line"
        }},
        {{
          "kind": "response",
          "speaker_label": "Speaker 1",
          "exact_quote": "An exact nearby response from Speaker 1"
        }}
      ]
    }}
  ],
  "summary_markdown": "The complete Markdown document described below"
}}

Use an empty speaker_inferences list when no mapping meets the evidence rules. A single source-labelled self-identification quote is sufficient only when it explicitly says "I am NAME", "I'm NAME", or "my name is NAME". Otherwise provide at least two exact, nearby evidence quotes, including one utterance by the generic source label and one quote containing the proposed name or role. If multiple generic speakers share one explicit role, the same role label may be proposed for each; Koe will preserve distinct numbering.

**SUMMARY_MARKDOWN FORMAT (markdown hierarchy: H1 title, H2 sections, H5 subtopics/owners):**

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

##### Person A
- Task one
- Task two

##### Person B
- Task three

##### Person A & Person B
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

Generate the JSON response now. The summary_markdown value must follow the format above exactly and start with its H1 title line."""
