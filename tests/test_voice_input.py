from unittest.mock import MagicMock
import pytest

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.services.voice_input import VoiceInputService


@pytest.fixture
def mock_audio_capture():
    capture = MagicMock()
    # 44-byte WAV header + 3200 bytes of PCM = 3244 bytes (~100ms of 16kHz mono audio)
    fake_wav = b"RIFF" + b"\x00" * 40 + b"\x01\x00" * 1600
    capture.stop_recording.return_value = fake_wav
    capture.sample_rate = 16000
    capture.channels = 1
    capture.is_recording = False
    return capture


@pytest.fixture
def mock_stt_service():
    stt = MagicMock()
    stt.transcribe.return_value = "Show me gaming laptops under 1000 dollars"
    return stt


def test_successful_capture_and_transcribe(mock_audio_capture, mock_stt_service):
    """Validate end-to-end voice capture and transcription flow."""
    voice_input = VoiceInputService(
        audio_capture=mock_audio_capture,
        stt_service=mock_stt_service,
    )

    transcript = voice_input.capture_and_transcribe(duration_seconds=0.01)

    assert transcript == "Show me gaming laptops under 1000 dollars"
    mock_audio_capture.start_recording.assert_called_once()
    mock_audio_capture.stop_recording.assert_called_once_with(as_wav=True)
    mock_stt_service.transcribe.assert_called_once()


def test_manual_start_stop_transcribe(mock_audio_capture, mock_stt_service):
    """Validate manual start_listening and stop_and_transcribe lifecycle."""
    voice_input = VoiceInputService(
        audio_capture=mock_audio_capture,
        stt_service=mock_stt_service,
    )

    voice_input.start_listening()
    mock_audio_capture.start_recording.assert_called_once()

    transcript = voice_input.stop_and_transcribe()
    assert transcript == "Show me gaming laptops under 1000 dollars"
    mock_audio_capture.stop_recording.assert_called_once_with(as_wav=True)
    mock_stt_service.transcribe.assert_called_once()


def test_empty_audio_handling(mock_audio_capture, mock_stt_service):
    """Validate that empty audio from capture raises CustomException and halts STT."""
    mock_audio_capture.stop_recording.return_value = b""
    voice_input = VoiceInputService(
        audio_capture=mock_audio_capture,
        stt_service=mock_stt_service,
    )

    with pytest.raises(CustomException) as exc_info:
        voice_input.stop_and_transcribe()

    assert "Audio capture returned empty audio" in str(exc_info.value)
    mock_stt_service.transcribe.assert_not_called()


def test_stt_failure_propagation(mock_audio_capture, mock_stt_service):
    """Validate that STT service errors are caught and re-raised as CustomException."""
    mock_stt_service.transcribe.side_effect = RuntimeError("ElevenLabs connection error")
    voice_input = VoiceInputService(
        audio_capture=mock_audio_capture,
        stt_service=mock_stt_service,
    )

    with pytest.raises(CustomException) as exc_info:
        voice_input.stop_and_transcribe()

    assert "Voice input transcription failed" in str(exc_info.value)


def test_microphone_start_failure(mock_audio_capture, mock_stt_service):
    """Validate error propagation when microphone fails to start."""
    mock_audio_capture.start_recording.side_effect = RuntimeError("Microphone device busy")
    voice_input = VoiceInputService(
        audio_capture=mock_audio_capture,
        stt_service=mock_stt_service,
    )

    with pytest.raises(CustomException) as exc_info:
        voice_input.start_listening()

    assert "Microphone start failure" in str(exc_info.value)


def test_invalid_duration_rejected(mock_audio_capture, mock_stt_service):
    """Validate rejection of non-positive duration values."""
    voice_input = VoiceInputService(
        audio_capture=mock_audio_capture,
        stt_service=mock_stt_service,
    )

    with pytest.raises(CustomException):
        voice_input.capture_and_transcribe(duration_seconds=0)

    with pytest.raises(CustomException):
        voice_input.capture_and_transcribe(duration_seconds=-1)


def test_timing_and_metric_tracking(mock_audio_capture, mock_stt_service):
    """Validate structured timing and audio size metric tracking."""
    voice_input = VoiceInputService(
        audio_capture=mock_audio_capture,
        stt_service=mock_stt_service,
    )

    voice_input.stop_and_transcribe()

    assert voice_input.last_audio_capture_time >= 0
    assert voice_input.last_stt_time >= 0
    assert voice_input.last_voice_input_total_time >= 0
    assert voice_input.last_audio_size_bytes > 0
    assert voice_input.last_audio_duration_ms > 0
