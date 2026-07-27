from meeting import summarizer


def test_summarizer_leaves_provider_selection_to_openrouter(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "# Meeting\n\n"
                                "## Summary\nDone.\n\n"
                                "## Key Decisions\nNone.\n\n"
                                "## Topics Discussed\nDone.\n\n"
                                "## Action Items\nNone.\n\n"
                                "## Open Questions\nNone."
                            )
                        }
                    }
                ]
            }

    def fake_post(url, *, headers, json, timeout):
        captured.update(
            url=url,
            headers=headers,
            body=json,
            timeout=timeout,
        )
        return Response()

    monkeypatch.setattr(summarizer.requests, "post", fake_post)

    result = summarizer.SummarizerClient(api_key="test-key").summarize("# Meeting")

    assert "## Open Questions" in result
    assert captured["body"]["model"] == "google/gemini-3.6-flash"
    assert "provider" not in captured["body"]


def test_summary_metadata_prefers_meeting_title_after_host_notes():
    content = """# Notes — Casey

- Confirm the invoice date.

# Casey - Invoice Workflow

**Date**: 2026-07-20 11:30
**Duration**: 42 minutes
**Participants**: Alex, Casey
"""

    metadata = summarizer.SummarizerClient(api_key="test-key")._extract_metadata(content)

    assert metadata == {
        "title": "Casey - Invoice Workflow",
        "date": "2026-07-20 11:30",
        "duration": "42 minutes",
        "participants": "Alex, Casey",
    }


def test_summary_prompt_keeps_overview_short_and_high_signal():
    prompt = summarizer.SummarizerClient(api_key="test-key")._build_prompt(
        "# Weekly Sync\n\nTranscript."
    )

    assert "No more than 2 concise paragraphs" in prompt
    assert "roughly 120 words total" in prompt
    assert "Move supporting detail" in prompt
    assert "into Topics Discussed" in prompt
    assert "2-4 short, high-signal phrases" in prompt


def test_incomplete_summary_response_is_retried(monkeypatch):
    responses = iter(
        [
            "# Meeting\n\n## Summary\nStopped early.",
            (
                "# Meeting\n\n"
                "## Summary\nDone.\n\n"
                "## Key Decisions\nNone.\n\n"
                "## Topics Discussed\nDone.\n\n"
                "## Action Items\nNone.\n\n"
                "## Open Questions\nNone."
            ),
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
