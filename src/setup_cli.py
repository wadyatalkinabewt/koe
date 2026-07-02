"""
Koe first-time setup — minimal terminal flow.

Asks for:
  1. ElevenLabs API key (required, used for transcription)
  2. OpenRouter key (optional, used for meeting summaries and cleanup benchmarks)
  3. User name (labels your voice in Scribe transcripts)

Writes .env and src/config.yaml, then exits. No model downloads, no GPU
checks — transcription is cloud-only.
"""

import os
import sys
from pathlib import Path


KOE_DIR = Path(__file__).parent.parent


def _print_header(title: str):
    print()
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)
    print()


def _input(prompt: str, default: str = "") -> str:
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    val = input(prompt).strip()
    return val if val else default


def _save_env(elevenlabs_key: str, openrouter_key: str, groq_key: str = ""):
    env_lines = []
    if elevenlabs_key:
        env_lines.append(f"ELEVENLABS_API_KEY={elevenlabs_key}")
    if groq_key:
        env_lines.append(f"GROQ_API_KEY={groq_key}")
    if openrouter_key:
        env_lines.append(f"OPENROUTER_API_KEY={openrouter_key}")
    (KOE_DIR / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")


def _save_config(user_name: str):
    import yaml

    config = {
        "profile": {
            "user_name": user_name,
        },
        "recording_options": {
            "activation_key": "ctrl+shift+space",
            "recording_mode": "press_to_toggle",
            "sample_rate": 16000,
            "silence_duration": 900,
        },
        "model_options": {
            "transcription_provider": "elevenlabs",
            "common": {
                "language": None,
                "initial_prompt": (
                    "Use proper punctuation including periods, commas, and question marks."
                ),
            },
            "elevenlabs": {
                "model_id": "scribe_v2",
                "keyterms_enabled": True,
                "temperature": 0.0,
            },
        },
        "post_processing": {
            "ai_cleanup_enabled": False,
            "ai_cleanup_threshold": 10,
            "ai_cleanup_model": "google/gemini-3.5-flash",
        },
        "misc": {
            "noise_on_completion": True,
            "snippets_folder": None,
            "print_to_terminal": True,
        },
        "meeting_options": {
            "root_folder": None,
        },
    }

    config_path = KOE_DIR / "src" / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)


def run_setup():
    os.system("cls" if sys.platform == "win32" else "clear")
    print("""
    ██╗  ██╗ ██████╗ ███████╗
    ██║ ██╔╝██╔═══██╗██╔════╝
    █████╔╝ ██║   ██║█████╗
    ██╔═██╗ ██║   ██║██╔══╝
    ██║  ██╗╚██████╔╝███████╗
    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝

    Cloud speech-to-text setup
    """)

    _print_header("1. ElevenLabs API key (required)")
    print("ElevenLabs runs Scribe v2 on their servers — no local GPU needed.")
    print("Get a key at: https://elevenlabs.io/app/settings/api-keys")
    print()
    elevenlabs_key = ""
    while not elevenlabs_key:
        elevenlabs_key = _input("Enter ElevenLabs API key")
        if not elevenlabs_key:
            print("ElevenLabs key is required for transcription.")

    _print_header("2. OpenRouter API key (optional)")
    print("OpenRouter powers meeting summaries and optional cleanup benchmarks.")
    print("Skip if you don't want either feature — transcription works without it.")
    print("Get a key at: https://openrouter.ai/keys")
    print()
    openrouter_key = _input("Enter OpenRouter API key (or press Enter to skip)")

    _print_header("3. Your name")
    print("Used to label your voice in Scribe meeting transcripts.")
    print()
    user_name = ""
    while not user_name:
        user_name = _input("Enter your first name")
        if not user_name:
            print("Name is required.")

    _print_header("Saving config")
    _save_env(elevenlabs_key, openrouter_key)
    _save_config(user_name)
    (KOE_DIR / ".setup_complete").touch()
    print("Done.")
    print()
    print("Press Ctrl+Shift+Space to transcribe.")
    print("Right-click the tray icon → Start Scribe for meetings.")
    print()


def main():
    try:
        run_setup()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
