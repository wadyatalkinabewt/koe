from meeting import summarizer


def test_summarizer_leaves_provider_selection_to_openrouter(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "# Summary\n\nDone."}}]}

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

    assert result == "# Summary\n\nDone."
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
