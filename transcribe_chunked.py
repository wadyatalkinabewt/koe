"""
Transcribe a large raw PCM file by sending it in chunks to the Koe server.
Usage: python transcribe_chunked.py <raw_pcm_file> [chunk_seconds]
"""
import sys
import os
import base64
import requests
import numpy as np

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit PCM
SERVER_URL = os.environ.get("WHISPER_SERVER_URL", "http://localhost:9876")


def transcribe_chunk(pcm_bytes, chunk_num):
    audio_base64 = base64.b64encode(pcm_bytes).decode('utf-8')
    response = requests.post(
        f"{SERVER_URL}/transcribe",
        json={
            "audio_base64": audio_base64,
            "sample_rate": SAMPLE_RATE,
            "language": None,
            "initial_prompt": "Use proper punctuation including periods, commas, and question marks.",
            "vad_filter": True
        },
        timeout=300
    )
    if response.status_code == 200:
        text = response.json().get("text", "").strip()
        return text
    else:
        print(f"  ERROR chunk {chunk_num}: {response.status_code} - {response.text}", file=sys.stderr)
        return ""


def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_chunked.py <raw_pcm_file> [chunk_seconds]")
        sys.exit(1)

    raw_file = sys.argv[1]
    chunk_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 120  # 2 min default

    chunk_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE * chunk_seconds

    file_size = os.path.getsize(raw_file)
    total_chunks = (file_size + chunk_bytes - 1) // chunk_bytes
    total_duration = file_size / (SAMPLE_RATE * BYTES_PER_SAMPLE)

    print(f"File: {raw_file}")
    print(f"Duration: {total_duration/60:.1f} minutes")
    print(f"Chunks: {total_chunks} x {chunk_seconds}s")
    print()

    all_text = []

    with open(raw_file, 'rb') as f:
        chunk_num = 0
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            chunk_num += 1
            start_time = (chunk_num - 1) * chunk_seconds
            mins, secs = divmod(start_time, 60)
            print(f"  [{chunk_num}/{total_chunks}] {int(mins):02d}:{int(secs):02d} ...", end=" ", flush=True)

            text = transcribe_chunk(data, chunk_num)
            if text:
                print(f"OK ({len(text)} chars)")
                all_text.append(text)
            else:
                print("(empty)")

    full_transcript = "\n\n".join(all_text)

    # Write output
    output_file = os.path.splitext(raw_file)[0] + "_transcript.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_transcript)

    print(f"\nDone! Transcript saved to: {output_file}")
    print(f"Total length: {len(full_transcript)} chars")


if __name__ == "__main__":
    main()
