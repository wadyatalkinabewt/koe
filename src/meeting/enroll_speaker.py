"""
Speaker enrollment utility.
Records audio from microphone and creates a voice fingerprint.

Usage:
    python -m src.meeting.enroll_speaker "Callum"
    python -m src.meeting.enroll_speaker "Sash" --file "sash_sample.wav"
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def record_sample(duration: float = 10.0, sample_rate: int = 16000, flexible: bool = False) -> np.ndarray:
    """Record audio from default microphone.

    Args:
        duration: Fixed recording duration (ignored if flexible=True)
        sample_rate: Audio sample rate
        flexible: If True, record until Enter is pressed (no time limit)
    """
    import sounddevice as sd
    import threading
    import msvcrt

    if flexible:
        print("\n" + "=" * 50)
        print("FLEXIBLE RECORDING MODE")
        print("Press ENTER to stop recording when done.")
        print("=" * 50)
    else:
        print(f"\nRecording for {duration} seconds...")

    print("Please speak clearly into your microphone.")
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Recording!")

    if flexible:
        # Flexible duration - record until Enter pressed
        frames = []
        stop_flag = threading.Event()

        def callback(indata, frame_count, time_info, status):
            if not stop_flag.is_set():
                frames.append(indata.copy())

        def wait_for_enter():
            while not stop_flag.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b'\r', b'\n', b' '):  # Enter or Space
                        stop_flag.set()
                        break
                time.sleep(0.05)

        # Start keyboard listener in background
        key_thread = threading.Thread(target=wait_for_enter, daemon=True)
        key_thread.start()

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', callback=callback):
            start_time = time.time()
            while not stop_flag.is_set():
                elapsed = time.time() - start_time
                print(f"\rRecording... {elapsed:.1f}s (Press ENTER to stop)", end="", flush=True)
                time.sleep(0.1)

        print("\n\nRecording complete.")

        if frames:
            audio = np.concatenate(frames)
            return audio.flatten()
        return np.array([], dtype=np.int16)
    else:
        # Fixed duration
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                       channels=1, dtype='int16')
        sd.wait()
        print("Recording complete.")
        return audio.flatten()


def load_audio_file(filepath: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio from a WAV file."""
    import wave

    with wave.open(filepath, 'rb') as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Only 16-bit audio supported")

        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)

        # Convert stereo to mono
        if wf.getnchannels() == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

        # Resample if needed
        file_rate = wf.getframerate()
        if file_rate != sample_rate:
            original_length = len(audio)
            target_length = int(original_length * sample_rate / file_rate)
            indices = np.linspace(0, original_length - 1, target_length)
            audio = np.interp(indices, np.arange(original_length),
                            audio.astype(np.float32))
            audio = audio.astype(np.int16)

    return audio


def main():
    parser = argparse.ArgumentParser(description="Enroll a speaker for voice recognition")
    parser.add_argument("name", help="Speaker name (e.g., 'Callum')")
    parser.add_argument("--file", "-f", help="Audio file to use instead of recording")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                       help="Recording duration in seconds (default: 10)")
    parser.add_argument("--flexible", action="store_true",
                       help="Record until Enter is pressed (no time limit)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List enrolled speakers")
    parser.add_argument("--remove", "-r", action="store_true",
                       help="Remove the specified speaker")

    args = parser.parse_args()

    # Import diarizer
    from .diarization import get_diarizer

    diarizer = get_diarizer()

    if args.list:
        if not diarizer.load():
            print("Failed to load diarization model")
            return 1
        speakers = diarizer.list_enrolled_speakers()
        if speakers:
            print("Enrolled speakers:")
            for name in speakers:
                print(f"  - {name}")
        else:
            print("No speakers enrolled yet.")
        return 0

    if args.remove:
        diarizer._load_known_speakers()  # Load without full model
        if diarizer.remove_speaker(args.name):
            return 0
        return 1

    # Load model
    print("Loading diarization model...")
    if not diarizer.load():
        print("Failed to load diarization model")
        return 1

    # Get audio
    if args.file:
        print(f"Loading audio from: {args.file}")
        audio = load_audio_file(args.file)
    else:
        audio = record_sample(duration=args.duration, flexible=args.flexible)

    duration = len(audio) / 16000
    print(f"Audio duration: {duration:.1f} seconds")

    # Enroll
    if diarizer.enroll_speaker(args.name, audio, sample_rate=16000):
        print(f"\nSuccessfully enrolled '{args.name}'!")
        print(f"Embedding saved to: speaker_embeddings/{args.name}.npy")
        return 0
    else:
        print(f"\nFailed to enroll '{args.name}'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
