#!/usr/bin/env python3
"""
Koe Audio File Transcriber
Transcribes audio files using the local Koe server.

Usage:
    python transcribe_file.py <audio_file>
    python transcribe_file.py voice_note.ogg
    python transcribe_file.py recording.mp3

Requires:
    - Koe server running (localhost:9876 or set WHISPER_SERVER_URL)
    - ffmpeg in PATH or installed via winget
"""

import sys
import base64
import subprocess
import tempfile
import os

# Try common ffmpeg locations
FFMPEG_PATHS = [
    "ffmpeg",  # In PATH
    r"C:\Users\Galbraith\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe",
]

def find_ffmpeg():
    """Find ffmpeg executable."""
    for path in FFMPEG_PATHS:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=True)
            return path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError("ffmpeg not found. Install via: winget install ffmpeg")


def transcribe_file(audio_path: str, server_url: str = None) -> str:
    """Transcribe an audio file using the Koe server."""
    import requests
    import numpy as np
    
    if server_url is None:
        server_url = os.environ.get("WHISPER_SERVER_URL", "http://localhost:9876")
    
    ffmpeg = find_ffmpeg()
    
    # Convert to 16kHz mono PCM using ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Use ffmpeg to convert to 16kHz mono 16-bit PCM
        cmd = [
            ffmpeg, '-y', '-i', audio_path,
            '-ar', '16000',     # 16kHz sample rate
            '-ac', '1',         # Mono
            '-f', 's16le',      # 16-bit little-endian PCM
            '-acodec', 'pcm_s16le',
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        # Read the raw PCM data
        with open(tmp_path, 'rb') as f:
            pcm_data = f.read()
        
        # Convert to numpy array
        audio_data = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
        
        # Send to Koe server
        response = requests.post(
            f"{server_url}/transcribe",
            json={
                "audio_base64": audio_base64,
                "sample_rate": 16000,
                "language": None,
                "initial_prompt": "Use proper punctuation including periods, commas, and question marks.",
                "vad_filter": True
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("text", "").strip()
        else:
            raise RuntimeError(f"Server error: {response.status_code} - {response.text}")
    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe_file.py <audio_file>")
        print("\nTranscribes audio files using the local Koe server.")
        print("Supports: .ogg, .oga, .mp3, .wav, .m4a, and most audio formats")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    try:
        text = transcribe_file(audio_file)
        print(text)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
