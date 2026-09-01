"""Deepgram Nova-3 adapter for snippets and diarized Scribe recordings."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import requests

API_URL = "https://api.deepgram.com/v1/listen"
MODEL = "nova-3"


def _params(*, language: str | None, diarize: bool) -> dict[str, str]:
    params = {
        "model": MODEL,
        "smart_format": "true",
        "punctuate": "true",
        "filler_words": "false",
    }
    if language:
        params["language"] = language
    if diarize:
        # Deepgram deprecated diarize=true in favour of an explicitly selected
        # diarizer. "latest" is the documented choice for new batch clients.
        params["diarize_model"] = "latest"
    return params


def _normalize(payload: object, *, diarize: bool) -> tuple[dict | None, str | None]:
    if not isinstance(payload, dict):
        return None, "Deepgram returned an invalid JSON response"
    try:
        alternative = payload["results"]["channels"][0]["alternatives"][0]
    except (KeyError, IndexError, TypeError):
        return None, "Deepgram response did not contain a transcript"
    if not isinstance(alternative, dict):
        return None, "Deepgram response did not contain a transcript"

    words = alternative.get("words") or []
    if not isinstance(words, list):
        return None, "Deepgram response contained invalid word timestamps"
    if diarize and words:
        diarize_info = payload.get("metadata", {}).get("diarize_info")
        if not isinstance(diarize_info, dict):
            return None, "Deepgram did not run the requested diarizer"

    normalized_words = []
    for word in words:
        if not isinstance(word, dict):
            continue
        text = str(word.get("punctuated_word") or word.get("word") or "").strip()
        if not text:
            continue
        item = {
            "type": "word",
            "text": text,
            "start": word.get("start", 0.0),
            "end": word.get("end", word.get("start", 0.0)),
        }
        if diarize:
            if "speaker" not in word:
                return None, "Deepgram diarization returned an unlabelled word"
            item["speaker_id"] = f"speaker_{int(word['speaker'])}"
        normalized_words.append(item)

    return {
        "text": str(alternative.get("transcript") or "").strip(),
        "words": normalized_words,
    }, None


def transcribe_stream(
    stream: BinaryIO,
    api_key: str,
    *,
    language: str | None,
    diarize: bool,
    timeout: float,
) -> tuple[dict | None, str | None]:
    """Submit WAV bytes to Deepgram and return Koe's normalized result shape."""
    try:
        stream.seek(0)
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "audio/wav",
            },
            params=_params(language=language, diarize=diarize),
            data=stream,
            timeout=timeout,
        )
    except requests.Timeout:
        return None, "Deepgram timeout"
    except requests.RequestException as exc:
        return None, f"Deepgram request error: {exc}"

    if response.status_code != 200:
        return None, f"Deepgram HTTP {response.status_code}: {response.text[:300]}"
    try:
        payload = response.json()
    except ValueError:
        return None, "Deepgram returned an invalid JSON response"
    return _normalize(payload, diarize=diarize)


def transcribe_file(
    file_path: Path,
    api_key: str,
    *,
    language: str | None,
    diarize: bool,
    timeout: float,
) -> tuple[dict | None, str | None]:
    try:
        with Path(file_path).open("rb") as stream:
            return transcribe_stream(
                stream,
                api_key,
                language=language,
                diarize=diarize,
                timeout=timeout,
            )
    except OSError as exc:
        return None, f"Deepgram request error: {exc}"

