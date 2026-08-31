import json

from meeting import summarizer


SUMMARY_MARKDOWN = (
    "# Meeting\n\n"
    "## Summary\nDone.\n\n"
    "## Key Decisions\nNone.\n\n"
    "## Topics Discussed\nDone.\n\n"
    "## Action Items\nNone.\n\n"
    "## Open Questions\nNone."
)


def _payload(*, summary=SUMMARY_MARKDOWN, inferences=None):
    return json.dumps(
        {
            "speaker_inferences": list(inferences or []),
            "summary_markdown": summary,
        }
    )


def test_summarizer_uses_gemini_37_structured_output_and_openrouter_routing(
    monkeypatch,
):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": _payload()}}]}

    def fake_post(url, *, headers, json, timeout):
        captured.update(url=url, headers=headers, body=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(summarizer.requests, "post", fake_post)

    result = summarizer.SummarizerClient(api_key="test-key").analyze("# Meeting")

    assert "## Open Questions" in result.summary
    assert result.speaker_mapping == {}
    assert captured["body"]["model"] == "google/gemini-3.7-flash"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["provider"] == {"zdr": True}


def test_summary_metadata_prefers_meeting_title_after_host_notes():
    content = """# Notes — Jordan

- Confirm the invoice date.

# Jordan - Invoice Workflow

**Date**: 2026-07-20 11:30
**Duration**: 42 minutes
**Participants**: Alex, Jordan
"""

    metadata = summarizer.SummarizerClient(api_key="test-key")._extract_metadata(content)

    assert metadata == {
        "title": "Jordan - Invoice Workflow",
        "date": "2026-07-20 11:30",
        "duration": "42 minutes",
        "participants": "Alex, Jordan",
    }


def test_summary_prompt_requests_conservative_contextual_identity_and_short_overview():
    prompt = summarizer.SummarizerClient(api_key="test-key")._build_prompt(
        "# Weekly Sync\n\nTranscript.",
        speaker_labels=["Shaun", "Speaker 1"],
    )

    assert "Only these generic labels are eligible for inference: Speaker 1" in prompt
    assert "These labels are already trusted and must never be renamed: Shaun" in prompt
    assert "explicit self-identification" in prompt
    assert '"speaker_inferences"' in prompt
    assert '"summary_markdown"' in prompt
    assert "continue using the original labels" in prompt
    assert "don't guess real names" not in prompt
    assert "No more than 2 concise paragraphs" in prompt
    assert "roughly 120 words total" in prompt
    assert "Move supporting detail" in prompt
    assert "into Topics Discussed" in prompt
    assert "2-4 short, high-signal phrases" in prompt


def test_verified_direct_address_mapping_is_applied_to_summary(monkeypatch):
    transcript = """# Planning

**Participants**: Shaun, Speaker 1

## Transcript

**[00:00] Shaun**: Jessica, could you explain the next step?

**[00:03] Speaker 1**: Yes, I can explain the process.

**[00:10] Shaun**: Thanks, Jessica.

**[00:12] Speaker 1**: You're welcome.
"""
    inference = {
        "source_label": "Speaker 1",
        "proposed_label": "Jessica (OT)",
        "confidence": "high",
        "evidence": [
            {
                "kind": "direct_address",
                "speaker_label": "Shaun",
                "exact_quote": "Jessica, could you explain the next step?",
            },
            {
                "kind": "response",
                "speaker_label": "Speaker 1",
                "exact_quote": "Yes, I can explain the process.",
            },
        ],
    }

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "message": {
                            "content": _payload(
                                summary=SUMMARY_MARKDOWN.replace(
                                    "Done.", "Speaker 1 explained the process.", 1
                                ),
                                inferences=[inference],
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(summarizer.requests, "post", lambda *_args, **_kwargs: Response())

    result = summarizer.SummarizerClient(api_key="test-key").analyze(
        transcript,
        speaker_labels=["Shaun", "Speaker 1"],
    )

    assert result.speaker_mapping == {"Speaker 1": "Jessica (OT)"}
    assert "Jessica (OT) explained the process." in result.summary
    assert "Speaker 1" not in result.summary


def test_unverified_or_low_confidence_inference_keeps_generic_label(monkeypatch):
    transcript = """# Planning

**[00:00] Shaun**: Could you explain the next step?

**[00:03] Speaker 1**: Yes, I can explain the process.
"""
    inference = {
        "source_label": "Speaker 1",
        "proposed_label": "Miranda",
        "confidence": "low",
        "evidence": [
            {
                "kind": "response",
                "speaker_label": "Speaker 1",
                "exact_quote": "Yes, I can explain the process.",
            }
        ],
    }

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "message": {
                            "content": _payload(
                                summary=SUMMARY_MARKDOWN.replace(
                                    "Done.", "Speaker 1 explained the process.", 1
                                ),
                                inferences=[inference],
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(summarizer.requests, "post", lambda *_args, **_kwargs: Response())

    result = summarizer.SummarizerClient(api_key="test-key").analyze(
        transcript,
        speaker_labels=["Shaun", "Speaker 1"],
    )

    assert result.speaker_mapping == {}
    assert "Speaker 1 explained the process." in result.summary


def test_single_explicit_self_identification_is_sufficient():
    transcript = """# Introductions

**[00:00] Speaker 1**: Hello, my name is Jessica.
"""
    proposed = [
        {
            "source_label": "Speaker 1",
            "proposed_label": "Jessica",
            "confidence": "high",
            "evidence": [
                {
                    "kind": "self_identification",
                    "speaker_label": "Speaker 1",
                    "exact_quote": "Hello, my name is Jessica.",
                }
            ],
        }
    ]

    mapping = summarizer.SummarizerClient._validate_speaker_inferences(
        proposed,
        transcript,
        ["Speaker 1"],
    )

    assert mapping == {"Speaker 1": "Jessica"}


def test_fabricated_evidence_and_trusted_label_remapping_are_rejected():
    transcript = """# Planning

**[00:00] Shaun**: Jessica, could you explain the next step?

**[00:03] Speaker 1**: Yes, I can explain the process.
"""
    proposed = [
        {
            "source_label": "Speaker 1",
            "proposed_label": "Jessica",
            "confidence": "high",
            "evidence": [
                {
                    "kind": "direct_address",
                    "speaker_label": "Shaun",
                    "exact_quote": "Jessica introduced herself here.",
                },
                {
                    "kind": "response",
                    "speaker_label": "Speaker 1",
                    "exact_quote": "Yes, I can explain the process.",
                },
            ],
        },
        {
            "source_label": "Shaun",
            "proposed_label": "Sean",
            "confidence": "high",
            "evidence": [],
        },
    ]

    mapping = summarizer.SummarizerClient._validate_speaker_inferences(
        proposed,
        transcript,
        ["Shaun", "Speaker 1"],
    )

    assert mapping == {}


def test_exact_label_rewrite_does_not_confuse_speaker_1_and_speaker_10():
    rewritten = summarizer.replace_speaker_labels(
        "Speaker 1 asked Speaker 10. Speaker 10 answered Speaker 1.",
        {"Speaker 1": "Jessica", "Speaker 10": "Tana"},
    )

    assert rewritten == "Jessica asked Tana. Tana answered Jessica."


def test_shared_role_labels_are_numbered_instead_of_merging_speakers():
    transcript = """# Planning

**[00:00] Shaun**: Could someone from OT explain the plan?

**[00:02] Speaker 1**: I can explain the living arrangements.

**[00:10] Shaun**: Could the other OT worker cover travel?

**[00:12] Speaker 2**: Yes, I can cover the travel booking.
"""
    proposed = [
        {
            "source_label": "Speaker 1",
            "proposed_label": "OT",
            "confidence": "high",
            "evidence": [
                {
                    "kind": "affiliation",
                    "speaker_label": "Shaun",
                    "exact_quote": "Could someone from OT explain the plan?",
                },
                {
                    "kind": "response",
                    "speaker_label": "Speaker 1",
                    "exact_quote": "I can explain the living arrangements.",
                },
            ],
        },
        {
            "source_label": "Speaker 2",
            "proposed_label": "OT",
            "confidence": "high",
            "evidence": [
                {
                    "kind": "affiliation",
                    "speaker_label": "Shaun",
                    "exact_quote": "Could the other OT worker cover travel?",
                },
                {
                    "kind": "response",
                    "speaker_label": "Speaker 2",
                    "exact_quote": "Yes, I can cover the travel booking.",
                },
            ],
        },
    ]

    mapping = summarizer.SummarizerClient._validate_speaker_inferences(
        proposed,
        transcript,
        ["Shaun", "Speaker 1", "Speaker 2"],
    )

    assert mapping == {"Speaker 1": "OT 1", "Speaker 2": "OT 2"}


def test_incomplete_summary_response_is_retried(monkeypatch):
    responses = iter(
        [
            _payload(summary="# Meeting\n\n## Summary\nStopped early."),
            _payload(),
        ]
    )

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": next(responses)}}]}

    calls = []
    monkeypatch.setattr(
        summarizer.requests,
        "post",
        lambda *_args, **_kwargs: calls.append(True) or Response(),
    )
    monkeypatch.setattr(summarizer.time, "sleep", lambda _seconds: None)

    result = summarizer.SummarizerClient(api_key="test-key").summarize("# Meeting")

    assert len(calls) == 2
    assert "## Open Questions" in result
