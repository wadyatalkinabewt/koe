import io
import os
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf

from utils import ConfigManager
from transcription_client import TranscriptionClient, is_server_running

# Server client (lazy initialized)
_server_client = None
_server_mode = None  # None = not checked, True = use server, False = use local

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
    if not text or not text.strip():
        return

    try:
        snippets_dir = _get_snippets_dir()

        # Delete oldest (5) if it exists
        oldest = snippets_dir / f"snippet_{MAX_SNIPPETS}.md"
        if oldest.exists():
            oldest.unlink()

        # Shift existing files up (4→5, 3→4, 2→3, 1→2)
        for i in range(MAX_SNIPPETS - 1, 0, -1):
            old_file = snippets_dir / f"snippet_{i}.md"
            new_file = snippets_dir / f"snippet_{i+1}.md"
            if old_file.exists():
                old_file.rename(new_file)

        # Save new snippet as 1 (newest)
        new_file = snippets_dir / "snippet_1.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"# Snippet\n\n**Time:** {timestamp}\n\n---\n\n{text.strip()}\n"
        new_file.write_text(content, encoding='utf-8')

    except Exception as e:
        ConfigManager.console_print(f"Failed to save snippet: {e}")

def create_local_model():
    """Create a local model using the faster-whisper library."""
    from faster_whisper import WhisperModel

    ConfigManager.console_print('Creating local model...')
    local_model_options = ConfigManager.get_config_section('model_options')['local']
    compute_type = local_model_options['compute_type']
    model_path = local_model_options.get('model_path')

    if compute_type == 'int8':
        device = 'cpu'
        ConfigManager.console_print('Using int8 quantization, forcing CPU usage.')
    else:
        device = local_model_options['device']

    try:
        if model_path:
            ConfigManager.console_print(f'Loading model from: {model_path}')
            model = WhisperModel(model_path,
                                 device=device,
                                 compute_type=compute_type,
                                 download_root=None)
        else:
            model = WhisperModel(local_model_options['model'],
                                 device=device,
                                 compute_type=compute_type)
    except Exception as e:
        ConfigManager.console_print(f'Error initializing WhisperModel: {e}')
        ConfigManager.console_print('Falling back to CPU.')
        model = WhisperModel(model_path or local_model_options['model'],
                             device='cpu',
                             compute_type=compute_type,
                             download_root=None if model_path else None)

    ConfigManager.console_print('Local model created.')
    return model

def transcribe_local(audio_data, local_model=None):
    """Transcribe an audio file using a local model."""
    if not local_model:
        local_model = create_local_model()
    model_options = ConfigManager.get_config_section('model_options')

    audio_data_float = audio_data.astype(np.float32) / 32768.0

    response = local_model.transcribe(audio=audio_data_float,
                                      language=model_options['common']['language'],
                                      initial_prompt=model_options['common']['initial_prompt'],
                                      condition_on_previous_text=model_options['local']['condition_on_previous_text'],
                                      temperature=model_options['common']['temperature'],
                                      vad_filter=model_options['local']['vad_filter'],
                                      hallucination_silence_threshold=0.5,)
    return ''.join([segment.text for segment in list(response[0])])

def transcribe_api(audio_data):
    """Transcribe an audio file using the OpenAI API."""
    from openai import OpenAI

    model_options = ConfigManager.get_config_section('model_options')
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY') or None,
        base_url=model_options['api']['base_url'] or 'https://api.openai.com/v1'
    )

    byte_io = io.BytesIO()
    sample_rate = ConfigManager.get_config_section('recording_options').get('sample_rate') or 16000
    sf.write(byte_io, audio_data, sample_rate, format='wav')
    byte_io.seek(0)

    response = client.audio.transcriptions.create(
        model=model_options['api']['model'],
        file=('audio.wav', byte_io, 'audio/wav'),
        language=model_options['common']['language'],
        prompt=model_options['common']['initial_prompt'],
        temperature=model_options['common']['temperature'],
    )
    return response.text

def remove_filler_words(text):
    """Remove common filler words and clean up the result."""
    # Filler words to remove (case insensitive)
    fillers = [
        r'\bum+\b', r'\buh+\b', r'\bah+\b', r'\beh+\b',
        r'\bhmm+\b', r'\bmm+\b', r'\bhm+\b',
        r'\byou know,?\s*', r'\bI mean,?\s*',
    ]

    # Whisper hallucinations - ONLY remove at end of transcription
    # (model hallucinates YouTube outros when audio trails off)
    trailing_hallucinations = [
        # YouTube outros
        r"\s*we'?ll be right back\.?\s*$",
        r"\s*thank(s| you) for watching\.?\s*$",
        r"\s*subscribe to (my|the|our) channel\.?\s*$",
        r"\s*please (like and )?subscribe\.?\s*$",
        r"\s*see you (in the )?next (one|video|time)\.?\s*$",
        r"\s*don'?t forget to (like and )?subscribe\.?\s*$",
        r"\s*like (and )?subscribe\.?\s*$",
        r"\s*hit the (like|bell|subscribe)( button)?\.?\s*$",
        # Common trailing phrases
        r"\s*(so,?\s*)?that'?s (it|all)( for (today|now))?\.?\s*$",
        r"\s*bye( bye)?\.?\s*$",
        r"\s*goodbye\.?\s*$",
        r"\s*take care\.?\s*$",
        # Incomplete trailing sentences (hallucinated continuations)
        r",?\s*and I'?ll\.?\s*$",
        r",?\s*and we'?ll\.?\s*$",
        r",?\s*and I'?m\.?\s*$",
        r",?\s*so I'?ll\.?\s*$",
        r",?\s*I'?ll see\.?\s*$",
        # Music/sound descriptions
        r"\s*\[music\]\s*$",
        r"\s*\[applause\]\s*$",
        r"\s*♪.*$",
    ]
    for pattern in trailing_hallucinations:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    for filler in fillers:
        text = re.sub(filler, '', text, flags=re.IGNORECASE)

    # Clean up resulting issues
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\s+([,.?!])', r'\1', text)  # Space before punctuation
    text = re.sub(r'([,.?!])\s*\1+', r'\1', text)  # Duplicate punctuation
    text = re.sub(r',\s*\.', '.', text)  # Comma followed by period
    text = re.sub(r'^\s*,\s*', '', text)  # Leading comma
    return text.strip()

def ensure_ending_punctuation(text):
    """Ensure text ends with proper punctuation."""
    text = text.strip()
    if text and text[-1] not in '.?!':
        text += '.'
    return text

def post_process_transcription(transcription):
    """Apply post-processing to the transcription."""
    transcription = transcription.strip()
    transcription = remove_filler_words(transcription)
    transcription = ensure_ending_punctuation(transcription)
    transcription += ' '  # Trailing space for easy pasting
    return transcription

def check_server_available():
    """Check if the transcription server is running."""
    global _server_client, _server_mode

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

    if _server_client is None:
        _server_client = TranscriptionClient()

    model_options = ConfigManager.get_config_section('model_options')
    language = model_options['common'].get('language')

    # Check if voice filtering is enabled
    filter_to_speaker = None
    if ConfigManager.get_config_value('recording_options', 'filter_snippets_to_my_voice'):
        my_voice = ConfigManager.get_config_value('profile', 'my_voice_embedding')
        if my_voice:
            filter_to_speaker = my_voice
            ConfigManager.console_print(f'Voice filtering enabled: {my_voice}')

    text, success = _server_client.transcribe(
        audio_data,
        sample_rate=16000,
        language=language,
        vad_filter=model_options['local'].get('vad_filter', True),
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
            _server_mode = None
            _server_client = TranscriptionClient()
            if _server_client.is_server_available(force_check=True):
                return transcribe_server(audio_data, retry_count=1)
            else:
                ConfigManager.console_print('Server no longer available after retry')

        return ''


def transcribe(audio_data, local_model=None):
    """Transcribe audio using server, API, or local model."""
    if audio_data is None:
        return ''

    # Priority: 1) API if configured, 2) Server if running, 3) Local model
    if ConfigManager.get_config_value('model_options', 'use_api'):
        transcription = transcribe_api(audio_data)
    elif check_server_available():
        transcription = transcribe_server(audio_data)
    else:
        transcription = transcribe_local(audio_data, local_model)

    result = post_process_transcription(transcription)
    save_rolling_transcription(result)
    return result
