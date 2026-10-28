from unittest.mock import MagicMock
import pytest

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.services.audio_capture import (
    AudioCapture,
    pcm_to_wav_bytes,
)


class MockAudioStream:
    """Mock audio stream that simulates PortAudio/sounddevice without physical hardware."""

    def __init__(
        self,
        samplerate=16000,
        channels=1,
        dtype="int16",
        device=None,
        callback=None,
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.callback = callback
        self.is_started = False
        self.is_stopped = False
        self.is_closed = False

    def start(self):
        self.is_started = True
        if self.callback:
            # Simulate 1600 bytes of 16-bit PCM audio frames
            sample_frames = b"\x05\x00\x08\x00" * 400
            self.callback(sample_frames, len(sample_frames) // 2, None, None)

    def stop(self):
        self.is_stopped = True

    def close(self):
        self.is_closed = True


def test_pcm_to_wav_bytes():
    """Verify in-memory conversion of PCM frames to a standard WAV container."""
    dummy_pcm = b"\x00\x00" * 160
    wav_bytes = pcm_to_wav_bytes(dummy_pcm, sample_rate=16000, channels=1)

    assert isinstance(wav_bytes, bytes)
    assert wav_bytes.startswith(b"RIFF")
    assert b"WAVE" in wav_bytes
    assert len(wav_bytes) == len(dummy_pcm) + 44  # Standard 44-byte WAV header


def test_audio_capture_initialization():
    """Validate initialization and default speech parameters."""
    capture = AudioCapture(
        sample_rate=16000,
        channels=1,
        dtype="int16",
        stream_factory=MockAudioStream,
    )

    assert capture.sample_rate == 16000
    assert capture.channels == 1
    assert capture.dtype == "int16"
    assert not capture.is_recording


def test_invalid_configuration():
    """Validate that invalid sample rates or channels raise CustomException."""
    with pytest.raises(CustomException):
        AudioCapture(sample_rate=0)

    with pytest.raises(CustomException):
        AudioCapture(sample_rate=-16000)

    with pytest.raises(CustomException):
        AudioCapture(channels=0)


def test_start_and_stop_recording():
    """Validate start/stop recording lifecycle and stream resource disposal."""
    mock_stream_instance = None

    def stream_factory(**kwargs):
        nonlocal mock_stream_instance
        mock_stream_instance = MockAudioStream(**kwargs)
        return mock_stream_instance

    capture = AudioCapture(stream_factory=stream_factory)

    capture.start_recording()
    assert capture.is_recording
    assert mock_stream_instance.is_started

    # Stop recording
    audio_bytes = capture.stop_recording(as_wav=True)
    assert not capture.is_recording
    assert mock_stream_instance.is_stopped
    assert mock_stream_instance.is_closed
    assert capture.last_capture_time >= 0
    assert len(audio_bytes) > 0
    assert audio_bytes.startswith(b"RIFF")


def test_audio_bytes_returned_raw_pcm():
    """Validate returning raw PCM bytes when as_wav=False."""
    capture = AudioCapture(stream_factory=MockAudioStream)
    capture.start_recording()
    raw_bytes = capture.stop_recording(as_wav=False)

    assert isinstance(raw_bytes, bytes)
    assert len(raw_bytes) > 0
    assert not raw_bytes.startswith(b"RIFF")


def test_microphone_failure_on_start():
    """Validate graceful error handling when microphone device is unavailable."""
    def failing_stream_factory(**kwargs):
        raise RuntimeError("PortAudio device unavailable or permission denied")

    capture = AudioCapture(stream_factory=failing_stream_factory)

    with pytest.raises(CustomException) as exc_info:
        capture.start_recording()

    assert "Microphone recording failure" in str(exc_info.value)
    assert not capture.is_recording


def test_resource_cleanup_on_stop_error():
    """Validate that resources are cleared even if stream stop throws."""
    class ErrorStoppingStream(MockAudioStream):
        def stop(self):
            super().stop()
            raise RuntimeError("Stream buffer flush error")

    capture = AudioCapture(stream_factory=ErrorStoppingStream)
    capture.start_recording()

    with pytest.raises(CustomException) as exc_info:
        capture.stop_recording()

    assert "Recording stop failure" in str(exc_info.value)
    assert not capture.is_recording


def test_context_manager_cleanup():
    """Validate context manager starts recording on enter and stops on exit."""
    mock_instance = None

    def stream_factory(**kwargs):
        nonlocal mock_instance
        mock_instance = MockAudioStream(**kwargs)
        return mock_instance

    with AudioCapture(stream_factory=stream_factory) as capture:
        assert capture.is_recording
        assert mock_instance.is_started

    assert not capture.is_recording
    assert mock_instance.is_stopped
    assert mock_instance.is_closed
