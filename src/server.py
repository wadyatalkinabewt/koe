"""
Koe Transcription Server

Provides a shared transcription endpoint for:
- Koe (hotkey-triggered transcription)
- Meeting Transcription mode (continuous meeting recording)
- Remote clients over Tailscale

Run with: python -m src.server
Or: python src/server.py
"""

import os
import sys
import io
import base64
import threading
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup CUDA DLLs before any CUDA imports (PATH modification more reliable than os.add_dll_directory)
def _setup_cuda_dlls():
    try:
        import site
        paths_to_add = []
        for sp in [site.getusersitepackages()] + site.getsitepackages():
            cudnn_bin = os.path.join(sp, "nvidia", "cudnn", "bin")
            cublas_bin = os.path.join(sp, "nvidia", "cublas", "bin")
            if os.path.exists(cudnn_bin):
                paths_to_add.append(cudnn_bin)
            if os.path.exists(cublas_bin):
                paths_to_add.append(cublas_bin)
        if paths_to_add:
            os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass

_setup_cuda_dlls()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import uvicorn
import json

# Import after CUDA setup
from faster_whisper import WhisperModel


class TranscribeRequest(BaseModel):
    """Request body for transcription."""
    audio_base64: str  # Base64-encoded int16 PCM audio
    sample_rate: int = 16000
    language: Optional[str] = None
    initial_prompt: Optional[str] = None
    vad_filter: bool = True
    filter_to_speaker: Optional[str] = None  # If set, only transcribe audio matching this enrolled speaker


class TranscribeResponse(BaseModel):
    """Response body for transcription."""
    text: str
    duration_seconds: float


class MeetingTranscribeRequest(BaseModel):
    """Request body for meeting transcription with diarization."""
    audio_base64: str  # Base64-encoded int16 PCM audio
    sample_rate: int = 16000
    language: Optional[str] = None
    initial_prompt: Optional[str] = None
    vad_filter: bool = True
    min_speakers: int = 1
    max_speakers: int = 6
    user_name: Optional[str] = None  # Name for mic audio (if known to be single speaker)


class MeetingSegment(BaseModel):
    """A transcribed segment with speaker identification."""
    speaker: str
    text: str
    start: float
    end: float


class MeetingTranscribeResponse(BaseModel):
    """Response body for meeting transcription."""
    segments: List[MeetingSegment]
    duration_seconds: float


class ServerStatus(BaseModel):
    """Server status response."""
    status: str
    model: str
    device: str
    ready: bool
    diarization_available: bool = False


# Global model instance
_model: Optional[WhisperModel] = None
_model_lock = threading.Lock()
_model_info = {"model": "", "device": ""}

# Global diarizer instance (lazy loaded)
_diarizer = None
_diarizer_lock = threading.Lock()
_diarizer_available = False


def get_diarizer():
    """Get the diarizer, loading if necessary. Returns None if unavailable."""
    global _diarizer, _diarizer_available

    with _diarizer_lock:
        if _diarizer is not None:
            return _diarizer

        try:
            from meeting.diarization import SpeakerDiarizer
            print("[Server] Loading diarization model...")
            _diarizer = SpeakerDiarizer()
            if _diarizer.load():
                _diarizer_available = True
                print("[Server] Diarization model loaded successfully")
            else:
                print("[Server] Diarization model failed to load")
                _diarizer = None
        except ImportError as e:
            print(f"[Server] Diarization not available: {e}")
            _diarizer = None
        except Exception as e:
            print(f"[Server] Diarization error: {e}")
            _diarizer = None

        return _diarizer


def load_model(model_name: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
    """Load the Whisper model."""
    global _model, _model_info

    with _model_lock:
        if _model is not None:
            return _model

        print(f"[Server] Loading Whisper model: {model_name} on {device}...")

        try:
            _model = WhisperModel(model_name, device=device, compute_type=compute_type)
            _model_info = {"model": model_name, "device": device}
            print(f"[Server] Model loaded successfully")
        except Exception as e:
            print(f"[Server] GPU load failed ({e}), falling back to CPU...")
            _model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
            _model_info = {"model": model_name, "device": "cpu"}
            print(f"[Server] Model loaded on CPU")

        return _model


def get_model() -> WhisperModel:
    """Get the loaded model, loading if necessary."""
    global _model
    if _model is None:
        load_model()
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    # Get config from environment or use defaults
    model = os.environ.get("WHISPER_MODEL", "large-v3")
    device = os.environ.get("WHISPER_DEVICE", "cuda")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")

    load_model(model, device, compute_type)
    yield
    # Cleanup if needed
    print("[Server] Shutting down...")


app = FastAPI(
    title="Koe Transcription Server",
    description="Shared Whisper transcription for Koe and Meeting Transcription",
    version="3.0.0",
    lifespan=lifespan
)


@app.get("/status", response_model=ServerStatus)
async def status():
    """Check server status and model readiness."""
    return ServerStatus(
        status="running",
        model=_model_info.get("model", "not loaded"),
        device=_model_info.get("device", "unknown"),
        ready=_model is not None,
        diarization_available=_diarizer_available
    )


@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/shutdown")
async def shutdown():
    """Shutdown the server."""
    import asyncio

    async def do_shutdown():
        await asyncio.sleep(0.5)
        os._exit(0)

    asyncio.create_task(do_shutdown())
    return {"status": "shutting_down"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """
    Transcribe audio data.

    Audio should be base64-encoded int16 PCM data.
    If filter_to_speaker is set, only audio matching that enrolled speaker is transcribed.
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert to float32 for Whisper
        audio_float = audio_int16.astype(np.float32) / 32768.0

        duration = len(audio_float) / request.sample_rate

        # Voice filtering: only transcribe audio matching the specified speaker
        if request.filter_to_speaker:
            diarizer = get_diarizer()
            if diarizer is None:
                raise HTTPException(status_code=503, detail="Diarization not available for voice filtering")

            # Run diarization to identify speakers
            speaker_segments = diarizer.diarize(
                audio_int16,
                sample_rate=request.sample_rate,
                min_speakers=1,
                max_speakers=4  # Reasonable default for snippet filtering
            )

            if not speaker_segments:
                # No speakers detected, return empty
                return TranscribeResponse(text="", duration_seconds=duration)

            # Filter to only segments matching the target speaker
            target_speaker = request.filter_to_speaker
            matching_segments = [seg for seg in speaker_segments if seg.speaker == target_speaker]

            if not matching_segments:
                # Target speaker not found in audio
                return TranscribeResponse(text="", duration_seconds=duration)

            # Extract and concatenate audio from matching segments
            filtered_audio_parts = []
            for seg in matching_segments:
                start_sample = int(seg.start * request.sample_rate)
                end_sample = int(seg.end * request.sample_rate)
                if end_sample <= start_sample or end_sample > len(audio_float):
                    continue
                filtered_audio_parts.append(audio_float[start_sample:end_sample])

            if not filtered_audio_parts:
                return TranscribeResponse(text="", duration_seconds=duration)

            audio_to_transcribe = np.concatenate(filtered_audio_parts)
        else:
            audio_to_transcribe = audio_float

        # Transcribe with anti-hallucination settings
        with _model_lock:
            segments, info = model.transcribe(
                audio=audio_to_transcribe,
                language=request.language,
                initial_prompt=request.initial_prompt,
                vad_filter=request.vad_filter,
                condition_on_previous_text=False,  # Prevents hallucination bleeding
                hallucination_silence_threshold=0.5,  # Skip silent sections (reduces trailing hallucinations)
            )
            text = "".join([segment.text for segment in segments])

        return TranscribeResponse(
            text=text.strip(),
            duration_seconds=duration
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/stream")
async def transcribe_stream(request: TranscribeRequest):
    """
    Stream transcription results segment by segment.

    Returns Server-Sent Events (SSE) with each segment as it's transcribed.
    Each event contains JSON: {"text": "...", "start": 0.0, "end": 1.5}
    Final event: {"done": true, "full_text": "..."}
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        async def generate_segments() -> AsyncGenerator[str, None]:
            full_text = []
            with _model_lock:
                segments, info = model.transcribe(
                    audio=audio_float,
                    language=request.language,
                    initial_prompt=request.initial_prompt,
                    vad_filter=request.vad_filter,
                    condition_on_previous_text=False,
                    hallucination_silence_threshold=0.5,
                )
                for segment in segments:
                    segment_data = {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end
                    }
                    full_text.append(segment.text)
                    yield f"data: {json.dumps(segment_data)}\n\n"

            # Send completion event
            done_data = {"done": True, "full_text": "".join(full_text).strip()}
            yield f"data: {json.dumps(done_data)}\n\n"

        return StreamingResponse(
            generate_segments(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe_meeting", response_model=MeetingTranscribeResponse)
async def transcribe_meeting(request: MeetingTranscribeRequest):
    """
    Transcribe audio with speaker diarization.

    Performs both diarization (who spoke when) and transcription (what they said).
    Returns segments with speaker labels and transcribed text.
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded")

    diarizer = get_diarizer()
    if diarizer is None:
        raise HTTPException(status_code=503, detail="Diarization model not available. Check HF_TOKEN and pyannote installation.")

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        duration = len(audio_float) / request.sample_rate

        # Run diarization to get speaker segments
        speaker_segments = diarizer.diarize(
            audio_int16,
            sample_rate=request.sample_rate,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers
        )

        if not speaker_segments:
            # No speakers detected, transcribe as single segment
            with _model_lock:
                segments, info = model.transcribe(
                    audio=audio_float,
                    language=request.language,
                    initial_prompt=request.initial_prompt,
                    vad_filter=request.vad_filter,
                    condition_on_previous_text=False,
                    hallucination_silence_threshold=0.5,
                )
                text = "".join([segment.text for segment in segments])

            return MeetingTranscribeResponse(
                segments=[MeetingSegment(
                    speaker=request.user_name or "Speaker",
                    text=text.strip(),
                    start=0.0,
                    end=duration
                )],
                duration_seconds=duration
            )

        # Transcribe each speaker segment
        result_segments = []
        for seg in speaker_segments:
            # Extract audio for this segment
            start_sample = int(seg.start * request.sample_rate)
            end_sample = int(seg.end * request.sample_rate)

            if end_sample <= start_sample or end_sample > len(audio_float):
                continue

            segment_audio = audio_float[start_sample:end_sample]

            # Skip very short segments (< 0.3 seconds)
            if len(segment_audio) < request.sample_rate * 0.3:
                continue

            # Transcribe this segment
            with _model_lock:
                segments, info = model.transcribe(
                    audio=segment_audio,
                    language=request.language,
                    initial_prompt=request.initial_prompt,
                    vad_filter=False,  # Already segmented by diarizer
                    condition_on_previous_text=False,
                    hallucination_silence_threshold=0.5,
                )
                text = "".join([s.text for s in segments]).strip()

            if text:  # Only include non-empty segments
                result_segments.append(MeetingSegment(
                    speaker=seg.speaker,
                    text=text,
                    start=seg.start,
                    end=seg.end
                ))

        return MeetingTranscribeResponse(
            segments=result_segments,
            duration_seconds=duration
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/diarization/reset")
async def reset_diarization():
    """Reset diarization session state (call at start of new meeting)."""
    diarizer = get_diarizer()
    if diarizer is None:
        raise HTTPException(status_code=503, detail="Diarization not available")

    diarizer.reset_session()
    return {"status": "ok", "message": "Diarization session reset"}


@app.get("/speakers")
async def list_speakers():
    """List enrolled speakers available for matching."""
    diarizer = get_diarizer()
    if diarizer is None:
        return {"speakers": [], "available": False}

    speakers = diarizer.list_enrolled_speakers()
    return {"speakers": speakers, "available": True}


def run_server(host: str = "0.0.0.0", port: int = 9876):
    """Run the server."""
    print(f"[Server] Starting on http://{host}:{port}")
    print(f"[Server] Local access: http://localhost:{port}")
    print(f"[Server] Remote access: http://<your-tailscale-ip>:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Whisper Transcription Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9876, help="Port to listen on")
    parser.add_argument("--model", default="large-v3", help="Whisper model to use")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    os.environ["WHISPER_MODEL"] = args.model
    os.environ["WHISPER_DEVICE"] = args.device

    run_server(args.host, args.port)
