"""Mistral Voxtral adapter for snippets and diarized Scribe recordings."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import requests

API_URL = "https://api.mistral.ai/v1/audio/transcriptions"
MODEL = "voxtral-mini-latest"


def _normalize(payload: object, *, diarize: bool) -> tuple[dict | None, str | None]:
    if not isinstance(payload, dict):
        return None, "Mistral returned an invalid JSON response"
    text = str(payload.get("text") or "").strip()
    segments = payload.get("segments") or []
    if not isinstance(segments, list):
        return None, "Mistral response contained invalid segments"
    if diarize and text and not segments:
        return None, "Mistral diarization returned no speaker-labelled segments"

    normalized_words = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_text = str(segment.get("text") or "").strip()
        if not segment_text:
            continue
        item = {
            "type": "word",
            "text": segment_text,
            "start": segment.get("start", 0.0),
            "end": segment.get("end", segment.get("start", 0.0)),
        }
        if diarize:
            speaker_id = segment.get("speaker_id")
            if speaker_id is None or str(speaker_id).strip() == "":
                return None, "Mistral diarization returned an unlabelled segment"
            item["speaker_id"] = speaker_id
        normalized_words.append(item)

    return {"text": text, "words": normalized_words}, None


def transcribe_stream(
    stream: BinaryIO,
    filename: str,
    api_key: str,
    *,
    language: str | None,
    diarize: bool,
    timeout: float,
) -> tuple[dict | None, str | None]:
    """Upload audio to Voxtral and return Koe's normalized result shape."""
    data: list[tuple[str, str]] = [("model", MODEL)]
    # Mistral documents ``language`` and ``timestamp_granularities`` as
    # mutually incompatible. Scribe needs timed diarized segments, so let
    # Voxtral detect the language for that path. Snippets may still use the
    # configured language because they do not request timestamps.
    if language and not diarize:
        data.append(("language", language))
    if diarize:
        data.extend(
            [
                ("diarize", "true"),
                ("timestamp_granularities", "segment"),
            ]
        )
    try:
        stream.seek(0)
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, stream, "audio/wav")},
            data=data,
            timeout=timeout,
        )
    except requests.Timeout:
        return None, "Mistral timeout"
    except requests.RequestException as exc:
        return None, f"Mistral request error: {exc}"

    if response.status_code != 200:
        return None, f"Mistral HTTP {response.status_code}: {response.text[:300]}"
    try:
        payload = response.json()
    except ValueError:
        return None, "Mistral returned an invalid JSON response"
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
                Path(file_path).name,
                api_key,
                language=language,
                diarize=diarize,
                timeout=timeout,
            )
    except OSError as exc:
        return None, f"Mistral request error: {exc}"
