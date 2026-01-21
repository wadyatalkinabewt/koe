"""
Record system audio (loopback) for speaker enrollment.
Play a YouTube video of someone speaking, then run this to capture it.

Usage:
    python -m src.meeting.record_loopback "Calum" --duration 30
    python -m src.meeting.record_loopback "Sash" --duration 20
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Try to import pyaudiowpatch for WASAPI loopback
try:
    import pyaudiowpatch as pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("pyaudiowpatch not installed. Run: pip install pyaudiowpatch")


def get_loopback_device():
    """Find the default loopback device."""
    p = pyaudio.PyAudio()

    try:
        # Get default WASAPI output device
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        # Find the loopback device for this output
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                if default_output["name"] in dev["name"]:
                    return dev

        # Fallback: any loopback device
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                return dev

    finally:
        p.terminate()

    return None


def record_loopback(duration: float, output_dir: Path, flexible: bool = False) -> tuple:
    """
    Record system audio for the specified duration.

    Args:
        duration: Fixed recording duration (ignored if flexible=True)
        output_dir: Directory for output files
        flexible: If True, record until Enter is pressed (no time limit)

    Returns:
        Tuple of (audio_data, sample_rate, channels)
    """
    import threading
    import msvcrt

    if not PYAUDIO_AVAILABLE:
        print("Error: pyaudiowpatch not available")
        return None, 0, 0

    device = get_loopback_device()
    if not device:
        print("Error: No loopback device found")
        return None, 0, 0

    print(f"Recording from: {device['name']}")
    print(f"Sample rate: {int(device['defaultSampleRate'])} Hz")
    print(f"Channels: {device['maxInputChannels']}")

    p = pyaudio.PyAudio()

    sample_rate = int(device['defaultSampleRate'])
    channels = device['maxInputChannels']
    chunk_size = 1024

    frames = []
    stop_flag = threading.Event()

    def wait_for_enter():
        while not stop_flag.is_set():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b'\r', b'\n', b' '):  # Enter or Space
                    stop_flag.set()
                    break
            time.sleep(0.05)

    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device['index'],
            frames_per_buffer=chunk_size
        )

        print()
        print("=" * 50)
        if flexible:
            print("FLEXIBLE RECORDING MODE")
            print("Play the YouTube video NOW!")
            print("Press ENTER to stop recording when done.")
        else:
            print("RECORDING STARTED")
            print("Play the YouTube video NOW!")
        print("=" * 50)
        print()

        # Countdown
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)

        print("Recording!")
        print()

        if flexible:
            # Start keyboard listener in background
            key_thread = threading.Thread(target=wait_for_enter, daemon=True)
            key_thread.start()

            # Record until Enter pressed
            start_time = time.time()
            while not stop_flag.is_set():
                elapsed = time.time() - start_time
                print(f"\rRecording... {elapsed:.1f}s (Press ENTER to stop)", end="", flush=True)
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
        else:
            # Record for fixed duration
            start_time = time.time()
            while time.time() - start_time < duration:
                remaining = duration - (time.time() - start_time)
                print(f"\rRecording... {remaining:.1f}s remaining", end="", flush=True)
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))

        print("\n\nRecording complete!")

        stream.stop_stream()
        stream.close()

    finally:
        p.terminate()

    if frames:
        audio = np.concatenate(frames)
        return audio, sample_rate, channels

    return None, 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Record system audio for speaker enrollment"
    )
    parser.add_argument("name", help="Speaker name (e.g., 'Calum')")
    parser.add_argument(
        "--duration", "-d", type=float, default=30.0,
        help="Recording duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--flexible", action="store_true",
        help="Record until Enter is pressed (no time limit)"
    )
    parser.add_argument(
        "--enroll", "-e", action="store_true",
        help="Automatically enroll after recording"
    )

    args = parser.parse_args()

    if not PYAUDIO_AVAILABLE:
        return 1

    output_dir = Path(__file__).parent.parent.parent / "enrollment_recordings"
    output_dir.mkdir(exist_ok=True)

    if args.flexible:
        print("Will record system audio until you press ENTER")
    else:
        print(f"Will record {args.duration} seconds of system audio")
    print(f"Speaker: {args.name}")
    print()
    print("Starting in 5 seconds - START PLAYING THE VIDEO NOW!")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    audio, sample_rate, channels = record_loopback(args.duration, output_dir, flexible=args.flexible)

    if audio is None:
        print("Recording failed")
        return 1

    # Convert stereo to mono if needed
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        original_length = len(audio)
        target_length = int(original_length * 16000 / sample_rate)
        indices = np.linspace(0, original_length - 1, target_length)
        audio = np.interp(indices, np.arange(original_length), audio.astype(np.float32))
        audio = audio.astype(np.int16)
        sample_rate = 16000

    # Save the recording
    import wave
    wav_path = output_dir / f"{args.name}_recording.wav"
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        wf.writeframes(audio.tobytes())

    print(f"\nSaved recording to: {wav_path}")

    # Check audio level
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    print(f"Audio RMS level: {rms:.0f} (should be > 500 for good audio)")

    if rms < 200:
        print("WARNING: Audio level is very low. Make sure the video was playing!")
        return 1

    # Enroll if requested
    if args.enroll:
        print("\nEnrolling speaker...")

        # Load environment
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent.parent / ".env")

        from .diarization import get_diarizer

        diarizer = get_diarizer()
        if not diarizer.load():
            print("Failed to load diarization model")
            return 1

        if diarizer.enroll_speaker(args.name, audio, sample_rate=16000):
            print(f"\nSuccessfully enrolled '{args.name}'!")
        else:
            print(f"\nFailed to enroll '{args.name}'")
            return 1
    else:
        print(f"\nTo enroll this recording, run:")
        print(f'  python -m src.meeting.enroll_speaker "{args.name}" --file "{wav_path}"')

    return 0


if __name__ == "__main__":
    sys.exit(main())
