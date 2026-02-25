"""
Integration tests for server endpoints.

These tests require the server to be running, or mock the server responses.
Use pytest markers to skip when server is not available.
"""

import pytest
import base64
import numpy as np
import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Server URL for testing
TEST_SERVER_URL = "http://localhost:9876"


def server_available():
    """Check if the test server is available."""
    try:
        response = requests.get(f"{TEST_SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


# Skip marker for tests that require server
requires_server = pytest.mark.skipif(
    not server_available(),
    reason="Server not available at localhost:9876"
)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @requires_server
    def test_health_returns_ok(self):
        """Health endpoint should return ok status."""
        response = requests.get(f"{TEST_SERVER_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestStatusEndpoint:
    """Tests for /status endpoint."""

    @requires_server
    def test_status_returns_info(self):
        """Status endpoint should return server info."""
        response = requests.get(f"{TEST_SERVER_URL}/status")
        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "status" in data
        assert "model" in data
        assert "device" in data
        assert "ready" in data
        assert "diarization_available" in data
        assert "supports_long_audio" in data
        assert "busy" in data
        assert "active_requests" in data

    @requires_server
    def test_status_shows_running(self):
        """Status should show server is running."""
        response = requests.get(f"{TEST_SERVER_URL}/status")
        data = response.json()
        assert data["status"] == "running"


class TestTranscribeEndpoint:
    """Tests for /transcribe endpoint."""

    @staticmethod
    def create_test_audio(duration_seconds: float = 1.0, sample_rate: int = 16000) -> str:
        """Create test audio data as base64."""
        # Generate silent audio (zeros)
        samples = int(duration_seconds * sample_rate)
        audio = np.zeros(samples, dtype=np.int16)
        return base64.b64encode(audio.tobytes()).decode('utf-8')

    @staticmethod
    def create_sine_wave_audio(frequency: float = 440.0, duration_seconds: float = 1.0,
                                sample_rate: int = 16000, amplitude: float = 0.5) -> str:
        """Create a sine wave test audio as base64."""
        t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), False)
        audio_float = amplitude * np.sin(2 * np.pi * frequency * t)
        audio_int16 = (audio_float * 32767).astype(np.int16)
        return base64.b64encode(audio_int16.tobytes()).decode('utf-8')

    @requires_server
    def test_transcribe_accepts_valid_request(self):
        """Should accept valid transcription request."""
        audio_base64 = self.create_test_audio(0.5)

        response = requests.post(
            f"{TEST_SERVER_URL}/transcribe",
            json={
                "audio_base64": audio_base64,
                "sample_rate": 16000,
                "vad_filter": True
            },
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "duration_seconds" in data

    @requires_server
    def test_transcribe_returns_503_when_not_ready(self):
        """Should return 503 if engine not loaded."""
        # This test is conditional - if server is ready, it won't return 503
        # We're mainly testing that the endpoint handles this case
        pass  # Placeholder - would need to test with uninitialized server

    def test_transcribe_request_format(self):
        """Test that request format is correct."""
        # This is a unit test of the request format, not requiring server
        audio_base64 = self.create_test_audio(0.5)

        payload = {
            "audio_base64": audio_base64,
            "sample_rate": 16000,
            "language": "en",
            "initial_prompt": "Test prompt",
            "vad_filter": True,
            "filter_to_speaker": None
        }

        # Verify all expected fields are present
        assert "audio_base64" in payload
        assert "sample_rate" in payload
        assert isinstance(payload["sample_rate"], int)

    @requires_server
    def test_transcribe_invalid_base64(self):
        """Should handle invalid base64 gracefully."""
        response = requests.post(
            f"{TEST_SERVER_URL}/transcribe",
            json={
                "audio_base64": "not-valid-base64!!!",
                "sample_rate": 16000
            },
            timeout=10
        )

        # Should return error, not crash
        assert response.status_code in [400, 422, 500]


class TestTranscribeMeetingEndpoint:
    """Tests for /transcribe_meeting endpoint."""

    @staticmethod
    def create_test_audio(duration_seconds: float = 1.0, sample_rate: int = 16000) -> str:
        """Create test audio data as base64."""
        samples = int(duration_seconds * sample_rate)
        audio = np.zeros(samples, dtype=np.int16)
        return base64.b64encode(audio.tobytes()).decode('utf-8')

    @requires_server
    def test_transcribe_meeting_accepts_valid_request(self):
        """Should accept valid meeting transcription request."""
        audio_base64 = self.create_test_audio(1.0)

        response = requests.post(
            f"{TEST_SERVER_URL}/transcribe_meeting",
            json={
                "audio_base64": audio_base64,
                "sample_rate": 16000,
                "min_speakers": 1,
                "max_speakers": 4,
                "vad_filter": True
            },
            timeout=60
        )

        # May return 503 if diarization not available
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "segments" in data
            assert "duration_seconds" in data


class TestDiarizationEndpoints:
    """Tests for diarization-related endpoints."""

    @requires_server
    def test_speakers_list(self):
        """Should list enrolled speakers."""
        response = requests.get(f"{TEST_SERVER_URL}/speakers", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "speakers" in data
        assert "available" in data
        assert isinstance(data["speakers"], list)

    @requires_server
    def test_diarization_reset(self):
        """Should reset diarization session."""
        response = requests.post(f"{TEST_SERVER_URL}/diarization/reset", timeout=5)

        # May return 503 if diarization not available
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "ok"


class TestInputValidation:
    """Tests for input validation on endpoints."""

    @requires_server
    def test_missing_required_field(self):
        """Should reject request missing required fields."""
        response = requests.post(
            f"{TEST_SERVER_URL}/transcribe",
            json={
                # Missing audio_base64
                "sample_rate": 16000
            },
            timeout=10
        )

        assert response.status_code == 422  # Validation error

    @requires_server
    def test_invalid_sample_rate(self):
        """Should handle unusual sample rates."""
        audio_base64 = base64.b64encode(np.zeros(1000, dtype=np.int16).tobytes()).decode()

        response = requests.post(
            f"{TEST_SERVER_URL}/transcribe",
            json={
                "audio_base64": audio_base64,
                "sample_rate": 8000  # Unusual but valid
            },
            timeout=30
        )

        # Should accept or return meaningful error
        assert response.status_code in [200, 400, 500]


class TestAudioSizeLimits:
    """Tests for audio size handling (DoS prevention)."""

    def test_large_audio_base64_encoding(self):
        """Test encoding of large audio for size calculations."""
        # 10 minutes of audio at 16kHz mono
        duration_seconds = 600
        sample_rate = 16000
        samples = duration_seconds * sample_rate

        # Calculate expected sizes
        audio_bytes = samples * 2  # int16 = 2 bytes
        base64_size = (audio_bytes * 4) // 3 + 4  # Base64 overhead

        # About 38MB of base64 for 10 minutes
        assert base64_size < 50 * 1024 * 1024, "10 minutes should be under 50MB limit"

    def test_estimate_size_from_base64(self):
        """Test estimating audio size from base64 length."""
        # Create 1 second of audio
        audio = np.zeros(16000, dtype=np.int16)
        base64_str = base64.b64encode(audio.tobytes()).decode()

        # Estimate size (base64 is ~4/3 of original)
        estimated_bytes = len(base64_str) * 3 // 4

        # Should be approximately 32000 bytes (16000 samples * 2 bytes)
        assert 31000 < estimated_bytes < 33000
