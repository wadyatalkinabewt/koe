"""
Koe Setup CLI - Terminal-based first-time setup.

Simple, reliable setup that works on any display.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Models with their approximate sizes and recommendations
MODELS = [
    ("tiny", "~75MB", "Fast, lower accuracy. Good for testing."),
    ("base", "~150MB", "Good balance for CPU-only systems."),
    ("small", "~500MB", "Better accuracy, still CPU-friendly."),
    ("medium", "~1.5GB", "High accuracy. Needs ~2GB VRAM or good CPU."),
    ("large-v3", "~3GB", "Best accuracy. Needs ~4GB VRAM (GPU recommended)."),
]


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if sys.platform == 'win32' else 'clear')


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print()


def print_box(lines: list[str], color: str = None):
    """Print text in a simple box."""
    width = max(len(line) for line in lines) + 4
    print("+" + "-" * width + "+")
    for line in lines:
        print(f"|  {line.ljust(width - 2)}|")
    print("+" + "-" * width + "+")


def get_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    value = input(prompt).strip()
    return value if value else default


def get_choice(prompt: str, options: list[str], default: int = None) -> int:
    """Get numbered choice from user."""
    print(prompt)
    print()
    for i, option in enumerate(options, 1):
        print(f"  {i}) {option}")
    print()

    while True:
        if default:
            choice = input(f"Enter choice [1-{len(options)}] (default: {default}): ").strip()
        else:
            choice = input(f"Enter choice [1-{len(options)}]: ").strip()

        if not choice and default:
            return default

        try:
            num = int(choice)
            if 1 <= num <= len(options):
                return num
        except ValueError:
            pass

        print(f"Please enter a number between 1 and {len(options)}")


def check_gpu() -> tuple[bool, str]:
    """Check for NVIDIA GPU."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            gpu_name = parts[0].strip()
            gpu_mem = parts[1].strip() if len(parts) > 1 else "Unknown"
            return True, f"{gpu_name} ({gpu_mem})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None


def download_whisper_model(model_name: str) -> tuple[bool, str, str]:
    """Download Whisper model with progress.

    Returns: (success, device, compute_type)
    """
    print(f"\nDownloading Whisper {model_name}...")
    print("(This may take a few minutes on first run)\n")

    # Determine device
    has_gpu, _ = check_gpu()
    if has_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                device = "cpu"
                compute_type = "int8"
        except ImportError:
            device = "cpu"
            compute_type = "int8"
    else:
        device = "cpu"
        compute_type = "int8"

    try:
        from faster_whisper import WhisperModel

        print(f"Loading on {device.upper()}...")

        # This downloads if not cached
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        del model

        print("[OK] Whisper model ready!")
        return True, device, compute_type

    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        return False, device, compute_type


def download_diarization_models(hf_token: str) -> bool:
    """Download pyannote diarization models."""
    print("\nDownloading speaker diarization models...")
    print("(This may take a few minutes)\n")

    try:
        os.environ['HF_TOKEN'] = hf_token

        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        del pipeline

        print("[OK] Diarization models ready!")
        return True

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            print("[ERROR] Invalid HuggingFace token.")
        elif "403" in error_msg or "access" in error_msg.lower():
            print("[ERROR] Access denied. Accept the model license at:")
            print("        https://huggingface.co/pyannote/speaker-diarization-3.1")
        else:
            print(f"[ERROR] Failed to download: {error_msg[:100]}")
        return False


def save_config(model_name: str, user_name: str, hf_token: str, anthropic_key: str,
                meetings_folder: str, snippets_folder: str, skip_diarization: bool,
                device: str, compute_type: str):
    """Save configuration files."""
    import yaml

    koe_dir = Path(__file__).parent.parent

    # Save .env
    env_lines = []
    if hf_token:
        env_lines.append(f"HF_TOKEN={hf_token}")
    if anthropic_key:
        env_lines.append(f"ANTHROPIC_API_KEY={anthropic_key}")
    env_lines.append("WHISPER_SERVER_URL=http://localhost:9876")
    env_lines.append(f"WHISPER_MODEL={model_name}")
    env_lines.append(f"WHISPER_DEVICE={device}")
    env_lines.append(f"WHISPER_COMPUTE_TYPE={compute_type}")

    env_path = koe_dir / ".env"
    with open(env_path, 'w') as f:
        f.write("\n".join(env_lines) + "\n")

    # Save config.yaml
    config = {
        'profile': {
            'user_name': user_name,
            'my_voice_embedding': None
        },
        'meeting_options': {
            'root_folder': meetings_folder if meetings_folder else None
        },
        'recording_options': {
            'activation_key': 'ctrl+shift+space',
            'recording_mode': 'press_to_toggle',
            'sample_rate': 16000,
            'silence_duration': 900,
            'filter_snippets_to_my_voice': False
        },
        'model_options': {
            'local': {
                'model': model_name
            },
            'common': {
                'initial_prompt': "Use proper punctuation including periods, commas, and question marks."
            }
        },
        'misc': {
            'noise_on_completion': True,
            'snippets_folder': snippets_folder if snippets_folder else None,
            'print_to_terminal': True
        }
    }

    # Only include diarization skip flag if skipped
    if skip_diarization:
        config['meeting_options']['diarization_enabled'] = False

    config_path = koe_dir / "src" / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Create folders
    if meetings_folder:
        Path(meetings_folder).mkdir(parents=True, exist_ok=True)
        (Path(meetings_folder) / "Transcripts").mkdir(exist_ok=True)
        (Path(meetings_folder) / "Summaries").mkdir(exist_ok=True)

    if snippets_folder:
        Path(snippets_folder).mkdir(parents=True, exist_ok=True)

    # Create setup complete marker
    (koe_dir / ".setup_complete").touch()

    print("\n[OK] Configuration saved!")


def run_setup():
    """Run the terminal setup."""
    clear_screen()

    print("""
    ██╗  ██╗ ██████╗ ███████╗
    ██║ ██╔╝██╔═══██╗██╔════╝
    █████╔╝ ██║   ██║█████╗
    ██╔═██╗ ██║   ██║██╔══╝
    ██║  ██╗╚██████╔╝███████╗
    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝

    Local Speech-to-Text Setup
    """)

    print("This will set up Koe on your system.\n")

    # Check for GPU
    has_gpu, gpu_info = check_gpu()
    if has_gpu:
        print(f"[OK] GPU detected: {gpu_info}")
        recommended_model = 5  # large-v3
    else:
        print("[--] No NVIDIA GPU detected. CPU mode will be used.")
        print("     Smaller models recommended for better performance.")
        recommended_model = 2  # base

    # =========================================================================
    # MODEL SELECTION
    # =========================================================================
    print_header("1. Model Selection")

    print("Choose a Whisper model based on your hardware:\n")

    options = []
    for name, size, desc in MODELS:
        rec = " (Recommended)" if MODELS.index((name, size, desc)) + 1 == recommended_model else ""
        options.append(f"{name.ljust(10)} {size.ljust(10)} - {desc}{rec}")

    model_choice = get_choice("Available models:", options, default=recommended_model)
    selected_model = MODELS[model_choice - 1][0]
    print(f"\nSelected: {selected_model}")

    # =========================================================================
    # HUGGINGFACE TOKEN
    # =========================================================================
    print_header("2. Speaker Diarization (Optional)")

    print("""Speaker diarization identifies WHO is speaking in meetings.
It requires a free HuggingFace account and token.

What it enables:
  - Speaker labels in Scribe transcripts ("Bryce: ...", "Calum: ...")
  - Voice enrollment (recognize specific people by voice)
  - Post-meeting speaker identification

Without it:
  - Koe hotkey transcription works normally
  - Scribe works but without speaker identification
""")

    print("To get a token:")
    print("  1. Create account at https://huggingface.co")
    print("  2. Go to Settings -> Access Tokens")
    print("  3. Create a token (read access is sufficient)")
    print("  4. Accept the license at https://huggingface.co/pyannote/speaker-diarization-3.1")
    print()

    hf_token = get_input("Enter HuggingFace token (or press Enter to skip)")
    skip_diarization = False

    if not hf_token:
        skip_diarization = True
        print()
        print_box([
            "Skipping diarization setup.",
            "",
            "To enable later:",
            "  1. Get token from https://huggingface.co/settings/tokens",
            "  2. Add to .env file: HF_TOKEN=hf_your_token_here",
            "  3. Run: python run.py --setup"
        ])
        print()

    # =========================================================================
    # ANTHROPIC KEY (OPTIONAL)
    # =========================================================================
    print_header("3. AI Summaries (Optional)")

    print("""AI summaries automatically generate meeting notes after each Scribe session.
Uses Claude to extract key decisions, action items, and discussion topics.

What it enables:
  - Auto-generated summary when you stop recording
  - Key decisions and action items extracted
  - Topics discussed with brief descriptions
  - Saves to Meetings/Summaries/ folder

Cost: ~$0.04 per 60-minute meeting
""")

    print("To get a key:")
    print("  1. Create account at https://console.anthropic.com")
    print("  2. Go to API Keys")
    print("  3. Create a new key")
    print()

    anthropic_key = get_input("Enter Anthropic API key (or press Enter to skip)")

    if not anthropic_key:
        print()
        print_box([
            "Skipping AI summaries.",
            "",
            "To enable later:",
            "  1. Get key from https://console.anthropic.com",
            "  2. Add to .env file: ANTHROPIC_API_KEY=sk-ant-...",
        ])
        print()

    # =========================================================================
    # USER NAME
    # =========================================================================
    print_header("4. Your Name")

    print("Your name is used to label your voice in meeting transcripts.\n")

    while True:
        user_name = get_input("Enter your first name")
        if user_name:
            break
        print("Name is required.")

    # =========================================================================
    # OUTPUT FOLDERS
    # =========================================================================
    print_header("5. Output Folders")

    koe_dir = Path(__file__).parent.parent
    default_meetings = str(koe_dir / "Meetings")
    default_snippets = str(koe_dir / "Snippets")

    print("Where should Koe save files?\n")

    meetings_folder = get_input("Meetings folder", default_meetings)
    snippets_folder = get_input("Snippets folder", default_snippets)

    # =========================================================================
    # DOWNLOAD MODELS
    # =========================================================================
    print_header("6. Downloading Models")

    # Download Whisper
    success, device, compute_type = download_whisper_model(selected_model)
    if not success:
        print("\nSetup incomplete. Please fix the error and run again.")
        print("Run: python run.py --setup")
        sys.exit(1)

    # Download diarization (if token provided)
    if hf_token and not skip_diarization:
        if not download_diarization_models(hf_token):
            print("\nDiarization setup failed. Continuing without it.")
            print("You can set it up later by running: python run.py --setup")
            skip_diarization = True

    # =========================================================================
    # SAVE CONFIG
    # =========================================================================
    print_header("7. Saving Configuration")

    save_config(
        model_name=selected_model,
        user_name=user_name,
        hf_token=hf_token,
        anthropic_key=anthropic_key,
        meetings_folder=meetings_folder,
        snippets_folder=snippets_folder,
        skip_diarization=skip_diarization,
        device=device,
        compute_type=compute_type
    )

    # =========================================================================
    # DONE
    # =========================================================================
    print_header("Setup Complete!")

    print(f"""
Koe is ready to use!

Quick start:
  - Press {chr(0x2318) if sys.platform == 'darwin' else 'Ctrl'}+Shift+Space to transcribe speech
  - Right-click tray icon -> Start Scribe for meetings
  - Right-click tray icon -> Settings to customize

Model: {selected_model}
Diarization: {'Enabled' if not skip_diarization else 'Disabled (can enable later)'}
AI Summaries: {'Enabled' if anthropic_key else 'Disabled (can enable later)'}
""")

    input("Press Enter to launch Koe...")

    # Launch Koe
    print("\nStarting Koe...")


def main():
    """Entry point."""
    try:
        run_setup()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)


if __name__ == '__main__':
    main()
