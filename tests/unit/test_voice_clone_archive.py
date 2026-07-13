import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_save_voice_clone_audio_writes_lossless_wav_to_snippets_folder(monkeypatch, tmp_path):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager,
        "get_config_value",
        lambda section, key: str(tmp_path) if (section, key) == ("misc", "snippets_folder") else None,
    )
    audio = np.array([-32768, -100, 0, 100, 32767], dtype=np.int16)

    saved_path = transcription.save_voice_clone_audio(audio, sample_rate=16000)

    assert saved_path is not None
    assert saved_path.parent == tmp_path / "Eleven Labs voice clone"
    assert saved_path.name.startswith("snippet_")
    assert saved_path.suffix == ".wav"
    with wave.open(str(saved_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == len(audio)
        assert np.array_equal(np.frombuffer(wf.readframes(len(audio)), dtype=np.int16), audio)


def test_save_voice_clone_audio_ignores_empty_audio(monkeypatch, tmp_path):
    import transcription

    monkeypatch.setattr(
        transcription.ConfigManager,
        "get_config_value",
        lambda section, key: str(tmp_path) if (section, key) == ("misc", "snippets_folder") else None,
    )

    assert transcription.save_voice_clone_audio(np.array([], dtype=np.int16)) is None
    assert not (tmp_path / "Eleven Labs voice clone").exists()
