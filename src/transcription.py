import io
import os
import re
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
import requests
import soundfile as sf

from utils import ConfigManager
from transcription_client import TranscriptionClient, is_server_running

# Debug logging to file
_DEBUG_LOG = Path(__file__).parent.parent / "logs" / "debug.log"
_DEBUG_LOG.parent.mkdir(exist_ok=True)

def _debug(msg: str):
    """Write debug message to file with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [transcription] {msg}\n")
    except:
        pass

# Server client (lazy initialized)
_server_client = None
_server_mode = None  # None = not checked, True = use server, False = use local
_server_lock = threading.Lock()

# Rolling snippet storage
MAX_SNIPPETS = 5

def _get_snippets_dir() -> Path:
    """Get the snippets directory (configurable or default to Koe/Snippets)."""
    snippets_folder = ConfigManager.get_config_value('misc', 'snippets_folder')
    if snippets_folder:
        snippets_dir = Path(snippets_folder)
    else:
        # Default to <repo_root>/Snippets (relative to this file's location)
        snippets_dir = Path(__file__).parent.parent / "Snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    return snippets_dir

def save_rolling_transcription(text):
    """Save snippet to rolling markdown files (keeps last 5). Newest is 1, oldest is 5."""
    _debug("save_rolling_transcription() STARTED")
    if not text or not text.strip():
        _debug("  Empty text, skipping")
        return

    try:
        snippets_dir = _get_snippets_dir()
        _debug(f"  snippets_dir: {snippets_dir}")

        # Delete oldest (5) if it exists
        oldest = snippets_dir / f"snippet_{MAX_SNIPPETS}.md"
        if oldest.exists():
            _debug(f"  Deleting oldest: {oldest}")
            oldest.unlink()

        # Shift existing files up (4→5, 3→4, 2→3, 1→2)
        for i in range(MAX_SNIPPETS - 1, 0, -1):
            old_file = snippets_dir / f"snippet_{i}.md"
            new_file = snippets_dir / f"snippet_{i+1}.md"
            if old_file.exists():
                _debug(f"  Renaming {old_file.name} -> {new_file.name}")
                old_file.rename(new_file)

        # Save new snippet as 1 (newest)
        new_file = snippets_dir / "snippet_1.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"# Snippet\n\n**Time:** {timestamp}\n\n---\n\n{text.strip()}\n"
        _debug(f"  Writing to {new_file}")
        new_file.write_text(content, encoding='utf-8')
        _debug("save_rolling_transcription() FINISHED")

    except Exception as e:
        _debug(f"  EXCEPTION: {e}")
        ConfigManager.console_print(f"Failed to save snippet: {e}")

def create_local_engine():
    """Create a local transcription engine using the engine factory."""
    try:
        from engines import create_engine, is_engine_available, get_default_engine
    except ImportError:
        # Fallback for direct imports
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from engines import create_engine, is_engine_available, get_default_engine

    ConfigManager.console_print('Creating local transcription engine...')

    model_options = ConfigManager.get_config_section('model_options')

    # Get engine ID from config or default to whisper
    engine_id = model_options.get('engine', 'whisper')
    if not is_engine_available(engine_id):
        ConfigManager.console_print(f'Engine {engine_id} not available, using default')
        engine_id = get_default_engine()

    # Get engine-specific config
    if engine_id == 'whisper':
        engine_config = model_options.get('local', {})
    else:
        engine_config = model_options.get(engine_id, {})

    model_name = engine_config.get('model', 'large-v3')
    device = engine_config.get('device', 'auto')
    compute_type = engine_config.get('compute_type', 'float16')

    # Handle model_path for whisper
    model_path = engine_config.get('model_path')
    if model_path:
        model_name = model_path

    try:
        engine = create_engine(engine_id)
        success = engine.load(model_name, device, compute_type)
        if success:
            ConfigManager.console_print(f'Local engine ({engine_id}) created.')
            return engine
        else:
            ConfigManager.console_print(f'Failed to load engine {engine_id}')
            return None
    except Exception as e:
        ConfigManager.console_print(f'Error creating engine: {e}')
        return None


def create_local_model():
    """Create a local model (backward compatibility wrapper)."""
    return create_local_engine()

def transcribe_local(audio_data, local_engine=None):
    """Transcribe audio using a local engine."""
    if not local_engine:
        local_engine = create_local_engine()

    if local_engine is None:
        ConfigManager.console_print('No local engine available')
        return ''

    model_options = ConfigManager.get_config_section('model_options')

    # Convert audio to float32 if needed
    if audio_data.dtype != np.float32:
        audio_data_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_data_float = audio_data

    # Get common options
    language = model_options.get('common', {}).get('language')
    initial_prompt = model_options.get('common', {}).get('initial_prompt')
    vad_filter = model_options.get('local', {}).get('vad_filter', False)
    condition_on_previous = model_options.get('local', {}).get('condition_on_previous_text', False)

    result = local_engine.transcribe(
        audio=audio_data_float,
        sample_rate=16000,
        language=language,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous,
    )

    return result.text

def _groq_transcribe_chunk(buf, data, api_key, timeout=30):
    """Send a single WAV buffer to Groq API. Returns transcribed text or empty string."""
    import requests as req

    for attempt in range(2):
        try:
            response = req.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", buf, "audio/wav")},
                data=data,
                timeout=timeout,
            )

            if response.status_code == 200:
                text = response.json().get("text", "")
                _debug(f"  Groq response: {len(text)} chars")
                return text

            if response.status_code >= 500 and attempt == 0:
                _debug(f"  Groq server error {response.status_code}, retrying...")
                buf.seek(0)
                continue

            _debug(f"  Groq API error: {response.status_code} {response.text[:200]}")
            ConfigManager.console_print(f"Groq API error: {response.status_code}")
            return ''

        except req.Timeout:
            _debug(f"  Groq API timeout (attempt {attempt + 1})")
            if attempt == 0:
                buf.seek(0)
                continue
            ConfigManager.console_print("Groq API timeout")
            return ''
        except req.RequestException as e:
            _debug(f"  Groq API request error: {e}")
            ConfigManager.console_print(f"Groq API error: {e}")
            return ''

    return ''


# Groq file upload limit is 25MB. 16kHz mono int16 WAV = ~32KB/sec.
# 10 minutes = ~19.2MB, safely under the limit.
GROQ_CHUNK_MAX_SAMPLES = 10 * 60 * 16000  # 10 minutes at 16kHz


def transcribe_groq(audio_data):
    """Transcribe audio using Groq cloud API (whisper-large-v3)."""
    import wave

    _debug("transcribe_groq() STARTED")

    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        _debug("  ERROR: No GROQ_API_KEY in environment")
        ConfigManager.console_print("Error: GROQ_API_KEY not set in .env file")
        return ''

    # Convert int16 PCM numpy array to WAV in-memory
    if audio_data.dtype == np.float32:
        audio_int16 = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
    else:
        audio_int16 = audio_data.astype(np.int16)

    # Build request parameters
    model_options = ConfigManager.get_config_section('model_options')
    language = model_options.get('common', {}).get('language') or 'en'
    initial_prompt = model_options.get('common', {}).get('initial_prompt')

    data = {
        "model": "whisper-large-v3",
        "language": language,
    }
    if initial_prompt:
        data["prompt"] = initial_prompt

    # Split into chunks if audio exceeds Groq's 25MB file upload limit
    total_samples = len(audio_int16)
    if total_samples <= GROQ_CHUNK_MAX_SAMPLES:
        # Short audio — single request
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())
        buf.seek(0)

        _debug(f"  Sending to Groq API (language={language})")
        return _groq_transcribe_chunk(buf, data, api_key, timeout=60)
    else:
        # Long audio — chunk and concatenate
        num_chunks = (total_samples + GROQ_CHUNK_MAX_SAMPLES - 1) // GROQ_CHUNK_MAX_SAMPLES
        _debug(f"  Audio too long for single request ({total_samples / 16000:.0f}s), splitting into {num_chunks} chunks")

        all_text = []
        for i in range(num_chunks):
            start = i * GROQ_CHUNK_MAX_SAMPLES
            end = min(start + GROQ_CHUNK_MAX_SAMPLES, total_samples)
            chunk = audio_int16[start:end]
            chunk_dur = len(chunk) / 16000

            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(chunk.tobytes())
            buf.seek(0)

            _debug(f"  Chunk {i + 1}/{num_chunks} ({chunk_dur:.0f}s) sending to Groq API")
            text = _groq_transcribe_chunk(buf, data, api_key, timeout=120)
            if text:
                all_text.append(text)

        return ' '.join(all_text)


def post_process_transcription(transcription):
    """Apply post-processing to the transcription using the centralized TextProcessor."""
    from utils import TextProcessor
    return TextProcessor.process(transcription, add_trailing_space=True)



def ai_cleanup_transcription(text):
    """Clean up grammar, punctuation, and filler words via OpenRouter.

    Model and prompt configurable via post_processing.ai_cleanup_model and
    post_processing.ai_cleanup_prompt in config.yaml. Default model is
    google/gemini-3-flash-preview (winner of the cleanup model benchmark —
    see benchmarks/cleanup_bench.py).
    Returns the cleaned text, or the original text if cleanup fails.
    """
    _debug("ai_cleanup_transcription() STARTED")

    if not text or not text.strip():
        _debug("  Empty text, skipping")
        return text

    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            _debug("  No OPENROUTER_API_KEY, skipping AI cleanup")
            return text

        model = ConfigManager.get_config_value('post_processing', 'ai_cleanup_model') or 'google/gemini-3-flash-preview'
        prompt_prefix = ConfigManager.get_config_value('post_processing', 'ai_cleanup_prompt') or (
            "Clean up this transcription. Fix grammar, add proper punctuation, and remove filler words (um, uh, like, you know, etc.).\n\n"
            "IMPORTANT:\n- Do NOT summarize or change the meaning\n- Do NOT add any new information\n"
            "- Keep the same speaking style and tone\n- Output ONLY the cleaned text, nothing else (no quotes, no explanation)\n\nTranscription:\n"
        )
        prompt = prompt_prefix + text.strip()

        # Pin provider to avoid quantisation routing variance between runs.
        provider_pin = _provider_pin_for_model(model)

        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            # Generous output cap. 16384 comfortably exceeds any realistic snippet
            # (~75 min at 150 wpm). Cost is based on actual tokens generated.
            "max_tokens": 16384,
        }
        if provider_pin:
            body["provider"] = provider_pin

        _debug(f"  Calling OpenRouter ({model}, provider={provider_pin})...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )

        if response.status_code != 200:
            _debug(f"  OpenRouter HTTP {response.status_code}: {response.text[:300]}")
            return text

        data = response.json()
        if data.get("error"):
            _debug(f"  OpenRouter API error: {data['error']}")
            return text

        cleaned = data["choices"][0]["message"]["content"].strip()
        _debug(f"  AI cleanup complete: {len(text)} -> {len(cleaned)} chars")

        # Add trailing space back (was in original post-processing)
        if not cleaned.endswith(' '):
            cleaned += ' '

        return cleaned

    except Exception as e:
        _debug(f"  AI cleanup error: {e}")
        return text


def _provider_pin_for_model(model_id):
    """Return OpenRouter provider routing config for a given model slug.

    Pinning prevents the same model from routing to FP4/FP8 quantised hosts
    on some calls and full-precision on others, which shows up as quality
    variance. See benchmarks/cleanup_bench.py for the providers verified on.
    Returns None for unknown models (lets OpenRouter pick).
    """
    pins = {
        "google/gemini-3-flash-preview":         ["Google AI Studio"],
        "google/gemini-3.1-flash-lite-preview":  ["Google AI Studio"],
        "anthropic/claude-haiku-4-5":            ["Anthropic"],
        "anthropic/claude-sonnet-4-6":           ["Anthropic"],
        "openai/gpt-5.4-mini":                   ["OpenAI"],
        "deepseek/deepseek-v3.2":                ["Friendli"],
    }
    order = pins.get(model_id)
    if not order:
        return None
    return {"order": order, "allow_fallbacks": False}

def check_server_available():
    """Check if the transcription server is running."""
    global _server_client, _server_mode

    with _server_lock:
        if _server_mode is not None:
            return _server_mode

        _server_client = TranscriptionClient()
        _server_mode = _server_client.is_server_available(force_check=True)

        if _server_mode:
            ConfigManager.console_print('Transcription server detected - using shared model')
        else:
            ConfigManager.console_print('No transcription server - using local model')

        return _server_mode


def transcribe_server(audio_data, retry_count=0):
    """Transcribe using the shared server with retry on failure."""
    global _server_client, _server_mode

    with _server_lock:
        if _server_client is None:
            _server_client = TranscriptionClient()
        client = _server_client

    model_options = ConfigManager.get_config_section('model_options')
    language = model_options.get('common', {}).get('language')

    # Check if voice filtering is enabled
    filter_to_speaker = None
    if ConfigManager.get_config_value('recording_options', 'filter_snippets_to_my_voice'):
        my_voice = ConfigManager.get_config_value('profile', 'my_voice_embedding')
        if my_voice:
            filter_to_speaker = my_voice
            ConfigManager.console_print(f'Voice filtering enabled: {my_voice}')

    text, success = client.transcribe(
        audio_data,
        sample_rate=16000,
        language=language,
        vad_filter=model_options.get('local', {}).get('vad_filter', True),
        filter_to_speaker=filter_to_speaker
    )

    if success:
        return text
    else:
        ConfigManager.console_print(f'Server transcription failed: {text}')

        # Retry once with fresh connection check
        if retry_count < 1:
            ConfigManager.console_print('Retrying with fresh server connection...')
            # Reset cached state and recreate client
            with _server_lock:
                _server_mode = None
                _server_client = TranscriptionClient()
                new_client = _server_client
            if new_client.is_server_available(force_check=True):
                return transcribe_server(audio_data, retry_count=1)
            else:
                ConfigManager.console_print('Server no longer available after retry')

        return ''


def transcribe(audio_data, local_model=None):
    """Transcribe audio using server or local model."""
    _debug("transcribe() STARTED")
    if audio_data is None:
        _debug("  audio_data is None, returning empty")
        return ''

    # Calculate audio duration for AI cleanup threshold check
    sample_rate = 16000
    audio_duration_sec = len(audio_data) / sample_rate
    _debug(f"  Audio duration: {audio_duration_sec:.1f}s")

    # Get configured engine
    engine = ConfigManager.get_config_value('model_options', 'engine') or 'whisper'
    _debug(f"  Engine: {engine}")

    if engine == 'groq':
        _debug("  Using Groq cloud transcription")
        transcription = transcribe_groq(audio_data)
    else:
        # Check if server is available
        server_available = check_server_available()
        _debug(f"  Server available: {server_available}")

        # Parakeet requires server (can't run locally on Windows)
        if engine == 'parakeet' and not server_available:
            _debug("  ERROR: Parakeet requires server but server not available")
            raise RuntimeError("Parakeet is still loading - please wait.")

        # Priority: 1) Server if running, 2) Local model
        if server_available:
            _debug("  Using server transcription")
            transcription = transcribe_server(audio_data)
        else:
            _debug("  Using local transcription")
            transcription = transcribe_local(audio_data, local_model)

    _debug(f"  Raw transcription length: {len(transcription)}")
    result = post_process_transcription(transcription)
    _debug(f"  Post-processed result length: {len(result)}")

    # Check if AI cleanup is enabled and duration meets threshold
    ai_cleanup_enabled = ConfigManager.get_config_value('post_processing', 'ai_cleanup_enabled')
    ai_cleanup_threshold = ConfigManager.get_config_value('post_processing', 'ai_cleanup_threshold') or 30

    if ai_cleanup_enabled and audio_duration_sec >= ai_cleanup_threshold:
        _debug(f"  AI cleanup enabled and duration ({audio_duration_sec:.1f}s) >= threshold ({ai_cleanup_threshold}s)")
        result = ai_cleanup_transcription(result)
    else:
        _debug(f"  AI cleanup skipped (enabled={ai_cleanup_enabled}, duration={audio_duration_sec:.1f}s, threshold={ai_cleanup_threshold}s)")

    save_rolling_transcription(result)
    _debug("transcribe() FINISHED")
    return result
