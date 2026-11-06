import pathlib
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))

NEW_NAME = "20261106_141027.552+1300_out_Amy_+64210385860.ogg"


def test_parses_the_recorder_filename_convention():
    import recording_transcript as rt

    parsed = rt.parse_recording_filename(NEW_NAME)

    assert parsed.recorded_at == datetime(
        2026, 11, 6, 14, 10, 27, 552000, tzinfo=timezone(timedelta(hours=13))
    )
    assert parsed.direction == "out"
    assert parsed.other_name == "Amy"
    assert parsed.phone_number == "+64210385860"


@pytest.mark.parametrize(
    "stem, direction, name",
    [
        ("20261106_141027.552+1300_in_Amy_+64210385860.ogg", "in", "Amy"),
        # Unresolved caller: the recorder writes a doubled separator.
        ("20261106_141027.552+1300_out__+64210385860.ogg", "out", ""),
        # Older three-field template, no name at all.
        ("20261106_141027.552+1300_out_+64210385860.ogg", "out", ""),
        # Contact names may contain underscores and spaces.
        ("20261106_141027.552+1300_out_Mary_Jane_+64210385860.ogg", "out", "Mary_Jane"),
        ("20261106_141027.552+1300_out_Amy Smith_+64210385860.ogg", "out", "Amy Smith"),
    ],
)
def test_parses_every_filename_shape(stem, direction, name):
    import recording_transcript as rt

    parsed = rt.parse_recording_filename(stem)

    assert parsed.direction == direction
    assert parsed.other_name == name
    assert parsed.phone_number == "+64210385860"


def test_unknown_filenames_are_not_forced_into_the_convention():
    import recording_transcript as rt

    parsed = rt.parse_recording_filename("holiday clip.m4a")

    assert parsed.recorded_at is None
    assert parsed.direction is None
    assert parsed.other_name == ""
    assert parsed.phone_number == ""
    assert parsed.recognised is False


def test_channels_are_split_only_when_the_signals_differ():
    import recording_transcript as rt

    left = np.random.default_rng(1).normal(0, 500, 4000).astype(np.float32)
    unrelated = np.random.default_rng(2).normal(0, 400, 4000).astype(np.float32)
    silence = np.zeros(4000, dtype=np.float32)

    assert rt.channels_are_split(left, unrelated) is True
    assert rt.channels_are_split(left, left) is False
    assert rt.channels_are_split(silence, silence) is False
    assert rt.channels_are_split(left, left[:10]) is False


def test_resolver_attributes_to_the_louder_channel():
    import recording_transcript as rt

    rate = 16000
    left = np.full(rate * 4, 800, dtype=np.float32)
    right = np.full(rate * 4, 100, dtype=np.float32)
    right[rate * 2 : rate * 3] = 4000  # only the remote party speaks here

    resolve = rt.build_channel_resolver(left, right, rate, "Bryce", "Amy")

    assert resolve(0.0, 1.0) == "Bryce"
    assert resolve(2.2, 2.8) == "Amy"


def test_resolver_holds_the_previous_speaker_when_channels_tie():
    import recording_transcript as rt

    rate = 16000
    left = np.full(rate * 3, 900, dtype=np.float32)
    right = np.full(rate * 3, 900, dtype=np.float32)

    resolve = rt.build_channel_resolver(left, right, rate, "Bryce", "Amy")

    first = resolve(0.0, 1.0)
    # Both channels are active with no meaningful margin: keep the speaker.
    assert resolve(1.0, 2.0) == first


def test_resolver_with_a_clear_margin_still_switches():
    import recording_transcript as rt

    rate = 16000
    left = np.full(rate * 3, 100, dtype=np.float32)
    right = np.full(rate * 3, 100, dtype=np.float32)
    right[rate : rate * 2] = 6000

    resolve = rt.build_channel_resolver(left, right, rate, "Bryce", "Amy")

    assert resolve(0.0, 0.9) == "Bryce"
    assert resolve(1.0, 2.0) == "Amy"


def test_silent_segments_fall_back_to_the_owner():
    import recording_transcript as rt

    resolve = rt.build_channel_resolver(
        np.zeros(16000, dtype=np.float32),
        np.zeros(16000, dtype=np.float32),
        16000,
        "Bryce",
        "Amy",
    )

    assert resolve(0.0, 1.0) == "Bryce"


def test_mono_mix_averages_without_clipping():
    import recording_transcript as rt

    left = np.full(100, 32767, dtype=np.float32)
    right = np.full(100, 32767, dtype=np.float32)
    mixed = rt._mix_to_mono(left, right)

    assert mixed.dtype == np.int16
    assert np.max(np.abs(mixed)) <= 32767
    assert np.all(mixed == 32767)


def test_output_folder_uses_the_recording_time_direction_and_name():
    import recording_transcript as rt

    parsed = rt.parse_recording_filename(NEW_NAME)
    folder = rt._output_dir(
        pathlib.Path("root"), parsed, pathlib.Path(NEW_NAME)
    )

    assert folder.name == "2026-11-06_141027_out_Amy"


def test_two_calls_with_the_same_person_on_one_day_do_not_collide():
    import recording_transcript as rt

    first = rt.parse_recording_filename(NEW_NAME)
    later = rt.parse_recording_filename(
        "20261106_172000.100+1300_out_Amy_+64210385860.ogg"
    )

    first_dir = rt._output_dir(pathlib.Path("root"), first, pathlib.Path(NEW_NAME))
    later_dir = rt._output_dir(
        pathlib.Path("root"), later, pathlib.Path("20261106_172000.100+1300_out_Amy_+64210385860.ogg")
    )

    assert first_dir != later_dir


def test_repeat_runs_of_one_recording_stay_deterministic():
    import recording_transcript as rt

    parsed = rt.parse_recording_filename(NEW_NAME)

    assert rt._output_dir(pathlib.Path("root"), parsed, pathlib.Path(NEW_NAME)) == (
        rt._output_dir(pathlib.Path("root"), parsed, pathlib.Path(NEW_NAME))
    )


def _stub(monkeypatch, captured, labels=("Bryce", "Amy")):
    import recording_transcript as rt

    def fake_segments(path, label="Speaker", *, diarize=False, **kwargs):
        captured["diarize"] = diarize
        captured["has_resolver"] = kwargs.get("label_resolver") is not None
        return [
            {"start": 1.0, "end": 2.0, "text": "First.", "label": labels[0]},
            {"start": 3.0, "end": 4.0, "text": "Second.", "label": labels[1]},
        ]

    monkeypatch.setattr(rt, "transcribe_file_segments", fake_segments)
    monkeypatch.setattr(rt, "_decode_stereo", lambda source, dest: dest.write_bytes(b""))
    monkeypatch.setattr(
        rt,
        "_read_stereo",
        lambda path: (
            np.random.default_rng(3).normal(0, 500, 16000).astype(np.float32),
            np.random.default_rng(4).normal(0, 400, 16000).astype(np.float32),
        ),
    )
    monkeypatch.setattr(rt, "probe_audio", lambda path: {"duration": 5.0})
    monkeypatch.setattr(rt, "write_mono_wav", lambda path, audio, rate: path)


def test_channel_split_uses_a_resolver_instead_of_diarization(
    tmp_path, monkeypatch
):
    import recording_transcript as rt

    captured: dict = {}
    _stub(monkeypatch, captured)
    source = tmp_path / NEW_NAME
    source.write_bytes(b"x")

    result = rt.transcribe_recording(
        source, user_name="Bryce", output_root=tmp_path / "out"
    )

    assert captured == {"diarize": False, "has_resolver": True}
    assert result.attribution == "channel"
    assert result.participants == ["Bryce", "Amy"]
    assert result.output_dir.name == "2026-11-06_141027_out_Amy"
    assert result.transcript_md.exists()
    assert result.reused is False


def test_identical_channels_fall_back_to_diarization(tmp_path, monkeypatch):
    import recording_transcript as rt

    captured: dict = {}
    _stub(monkeypatch, captured)
    same = np.random.default_rng(5).normal(0, 500, 16000).astype(np.float32)
    monkeypatch.setattr(rt, "_read_stereo", lambda path: (same, same))
    source = tmp_path / NEW_NAME
    source.write_bytes(b"x")

    result = rt.transcribe_recording(source, output_root=tmp_path / "out")

    assert captured["diarize"] is True
    assert result.attribution == "diarization"
    assert any("diarization" in warning for warning in result.warnings)


def test_repeating_a_transcription_reuses_the_existing_document(
    tmp_path, monkeypatch
):
    import recording_transcript as rt

    captured: dict = {}
    _stub(monkeypatch, captured)
    source = tmp_path / NEW_NAME
    source.write_bytes(b"x")
    root = tmp_path / "out"

    first = rt.transcribe_recording(source, output_root=root)
    calls = []
    monkeypatch.setattr(
        rt, "transcribe_file_segments", lambda *a, **k: calls.append(1) or []
    )
    second = rt.transcribe_recording(source, output_root=root)

    assert first.reused is False
    assert second.reused is True
    assert calls == []
    assert second.output_dir == first.output_dir
