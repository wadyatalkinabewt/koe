"""
Tests for transcription post-processing.
"""

import pytest
import sys
import wave
from pathlib import Path
import numpy as np

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from transcription import (
    _audio_to_wav_bytes,
    _chunk_max_samples,
    _cleanup_preserves_tail,
    _merge_tail_retry,
    ensure_ending_punctuation,
    post_process_transcription,
    remove_filler_words,
)


class TestRemoveFillerWords:
    """Tests for filler word removal."""

    def test_removes_um(self):
        """Should remove 'um' filler words."""
        assert "Hello world" in remove_filler_words("Hello um world")
        assert "Hello world" in remove_filler_words("Hello umm world")
        assert "Hello world" in remove_filler_words("Hello ummm world")

    def test_removes_uh(self):
        """Should remove 'uh' filler words."""
        assert "Hello world" in remove_filler_words("Hello uh world")
        assert "Hello world" in remove_filler_words("Hello uhh world")

    def test_removes_hmm(self):
        """Should remove 'hmm' filler words."""
        assert "Hello world" in remove_filler_words("Hello hmm world")
        assert "Hello world" in remove_filler_words("Hello hmmm world")

    def test_preserves_real_words(self):
        """Should not remove words that contain filler patterns."""
        result = remove_filler_words("The umbrella is here")
        assert "umbrella" in result

    def test_removes_trailing_hallucinations(self):
        """Should remove common Whisper hallucinations at end of text."""
        test_cases = [
            ("Hello world. Thank you for watching.", "Hello world."),
            ("Hello world. Subscribe to my channel.", "Hello world."),
            ("Hello world. Please like and subscribe.", "Hello world."),
            ("Hello world. We'll be right back.", "Hello world."),
            ("Hello world. See you in the next video.", "Hello world."),
        ]
        for input_text, expected in test_cases:
            result = remove_filler_words(input_text)
            assert result.strip() == expected.strip(), f"Failed for: {input_text}"

    def test_cleans_multiple_spaces(self):
        """Should collapse multiple spaces to single space."""
        result = remove_filler_words("Hello    world")
        assert "  " not in result

    def test_removes_space_before_punctuation(self):
        """Should remove space before punctuation."""
        result = remove_filler_words("Hello , world .")
        assert result.strip() == "Hello, world."

    def test_handles_empty_string(self):
        """Should handle empty string gracefully."""
        assert remove_filler_words("") == ""

    def test_handles_only_fillers(self):
        """Should handle text that's only filler words."""
        result = remove_filler_words("um uh hmm")
        # Should return empty or very short string
        assert len(result.strip()) < 3


class TestEnsureEndingPunctuation:
    """Tests for ensuring proper ending punctuation."""

    def test_adds_period_if_missing(self):
        """Should add period if no ending punctuation."""
        assert ensure_ending_punctuation("Hello world") == "Hello world."

    def test_preserves_existing_period(self):
        """Should not add period if already ends with period."""
        assert ensure_ending_punctuation("Hello world.") == "Hello world."

    def test_preserves_question_mark(self):
        """Should not add period if ends with question mark."""
        assert ensure_ending_punctuation("How are you?") == "How are you?"

    def test_preserves_exclamation(self):
        """Should not add period if ends with exclamation."""
        assert ensure_ending_punctuation("Hello world!") == "Hello world!"

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        assert ensure_ending_punctuation("  Hello world  ") == "Hello world."

    def test_handles_empty_string(self):
        """Should handle empty string gracefully."""
        result = ensure_ending_punctuation("")
        assert result == ""


class TestPostProcessTranscription:
    """Tests for full post-processing pipeline."""

    def test_full_pipeline(self):
        """Should apply all post-processing steps."""
        input_text = "  um Hello , world uh "
        result = post_process_transcription(input_text)
        # Should remove fillers, fix punctuation, add trailing period and space
        assert "um" not in result.lower()
        assert "uh" not in result.lower()
        assert result.endswith(". ")  # Trailing space for easy pasting

    def test_preserves_content(self):
        """Should preserve the actual content."""
        result = post_process_transcription("Hello world")
        assert "Hello" in result
        assert "world" in result

    def test_handles_empty_input(self):
        """Should handle empty input gracefully."""
        result = post_process_transcription("")
        assert result == ""

    def test_handles_whitespace_only(self):
        """Should handle whitespace-only input."""
        result = post_process_transcription("   ")
        assert result == "   "


class TestAudioToWavBytes:
    def test_chunk_limit_stays_under_groq_upload_cap_for_high_sample_rates(self):
        assert _chunk_max_samples(48000) == 10 * 60 * 16000

    def test_appends_trailing_silence_without_changing_sample_rate(self):
        audio = np.array([1000, -1000, 500], dtype=np.int16)

        buf = _audio_to_wav_bytes(audio, sample_rate=8000, trailing_silence_sec=1.0)

        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == 8000
            assert wf.getnframes() == len(audio) + 8000
            raw = wf.readframes(wf.getnframes())

        samples = np.frombuffer(raw, dtype=np.int16)
        assert np.array_equal(samples[:len(audio)], audio)
        assert np.all(samples[len(audio):] == 0)


class TestCleanupTailGuard:
    def test_accepts_cleanup_that_preserves_tail(self):
        original = "This is a transcription. Could you look into this and fix it if you can."
        cleaned = "This is a transcription. Could you look into this, and fix it if you can."

        assert _cleanup_preserves_tail(original, cleaned)

    def test_rejects_cleanup_that_drops_tail(self):
        original = "This is a transcription. Could you look into this and fix it if you can."
        cleaned = "This is a transcription."

        assert not _cleanup_preserves_tail(original, cleaned)

    def test_rejects_cleanup_when_tail_words_only_appear_earlier(self):
        original = "Check the docs first. Then update the current setup with the final fix."
        cleaned = "Check the docs first. Then update the current setup."

        assert not _cleanup_preserves_tail(original, cleaned)


class TestTailRetryMerge:
    def test_keeps_full_text_when_tail_is_already_present(self):
        full = "Please check the docs and then update the final section."
        tail = "Then update the final section."

        assert _merge_tail_retry(full, tail) == full

    def test_appends_missing_tail_suffix(self):
        full = "Please check the docs and then update"
        tail = "and then update the final section before you finish."

        assert _merge_tail_retry(full, tail) == (
            "Please check the docs and then update the final section before you finish."
        )
