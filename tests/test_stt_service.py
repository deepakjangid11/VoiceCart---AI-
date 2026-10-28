import io
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.services.stt_service import (
    ElevenLabsSTTService,
    STTService,
)


@pytest.fixture
def mock_elevenlabs_client():
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Show me wireless noise cancelling headphones"
    client.speech_to_text.convert.return_value = mock_response
    return client


def test_stt_service_alias():
    """Verify STTService alias matches ElevenLabsSTTService."""
    assert STTService is ElevenLabsSTTService


def test_successful_transcription(mock_elevenlabs_client):
    """Validate successful transcription and returned text."""
    audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00fake_audio_payload"
    service = ElevenLabsSTTService(
        api_key="mock_test_key",
        client=mock_elevenlabs_client,
    )

    transcript = service.transcribe(audio_bytes)

    assert transcript == "Show me wireless noise cancelling headphones"
    mock_elevenlabs_client.speech_to_text.convert.assert_called_once()
    call_kwargs = mock_elevenlabs_client.speech_to_text.convert.call_args.kwargs
    assert call_kwargs["model_id"] == "scribe_v1"
    assert call_kwargs["file"][0] == "audio.wav"


def test_transcript_extraction_formats(mock_elevenlabs_client):
    """Validate extraction across different response shapes (object, dict, multichannel)."""
    service = ElevenLabsSTTService(
        api_key="mock_test_key",
        client=mock_elevenlabs_client,
    )

    # 1. Dict response format
    mock_elevenlabs_client.speech_to_text.convert.return_value = {
        "text": "   Laptops under 1000 dollars   "
    }
    res_dict = service.transcribe(b"audio_bytes")
    assert res_dict == "Laptops under 1000 dollars"

    # 2. Multichannel transcripts list response format
    mock_chunk1 = MagicMock(text="Find bestsellers")
    mock_chunk2 = MagicMock(text="in electronics")
    mock_resp_multi = MagicMock(spec=["transcripts"])
    mock_resp_multi.transcripts = [mock_chunk1, mock_chunk2]
    mock_elevenlabs_client.speech_to_text.convert.return_value = mock_resp_multi
    res_multi = service.transcribe(b"audio_bytes")
    assert res_multi == "Find bestsellers in electronics"


def test_empty_audio_validation(mock_elevenlabs_client):
    """Validate that empty or None audio payloads raise CustomException."""
    service = ElevenLabsSTTService(
        api_key="mock_test_key",
        client=mock_elevenlabs_client,
    )

    with pytest.raises(CustomException) as exc_empty:
        service.transcribe(b"")
    assert "Audio validation failed" in str(exc_empty.value)

    with pytest.raises(CustomException) as exc_none:
        service.transcribe(None)
    assert "Audio validation failed" in str(exc_none.value)


def test_missing_api_key_validation(monkeypatch):
    """Validate that missing or empty ELEVENLABS_API_KEY raises CustomException."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    with pytest.raises(CustomException) as exc_info:
        ElevenLabsSTTService(api_key=None)

    assert "ELEVENLABS_API_KEY is missing" in str(exc_info.value)

    with pytest.raises(CustomException) as exc_empty:
        ElevenLabsSTTService(api_key="   ")

    assert "ELEVENLABS_API_KEY is missing" in str(exc_empty.value)


def test_provider_api_failure(mock_elevenlabs_client):
    """Validate that ElevenLabs API errors are captured and raised as CustomException."""
    mock_elevenlabs_client.speech_to_text.convert.side_effect = RuntimeError(
        "ElevenLabs 429 Too Many Requests"
    )
    service = ElevenLabsSTTService(
        api_key="mock_test_key",
        client=mock_elevenlabs_client,
    )

    with pytest.raises(CustomException) as exc_info:
        service.transcribe(b"valid_audio_payload")

    assert "STT transcription failed" in str(exc_info.value)


def test_timing_and_metrics_logging(mock_elevenlabs_client):
    """Validate that timing and size metrics are recorded on the service."""
    audio_data = b"sample_audio_binary_data_123456"
    service = ElevenLabsSTTService(
        api_key="mock_test_key",
        client=mock_elevenlabs_client,
    )

    service.transcribe(audio_data)

    assert service.last_stt_time >= 0
    assert service.last_audio_size == len(audio_data)
    assert service.last_transcript_length == len(
        "Show me wireless noise cancelling headphones"
    )


def test_file_and_stream_inputs(tmp_path, mock_elevenlabs_client):
    """Validate handling of binary streams and audio file paths."""
    service = ElevenLabsSTTService(
        api_key="mock_test_key",
        client=mock_elevenlabs_client,
    )

    # 1. Binary stream (BytesIO)
    stream = io.BytesIO(b"streamed_audio_content")
    res_stream = service.transcribe(stream)
    assert res_stream == "Show me wireless noise cancelling headphones"
    assert service.last_audio_size == len(b"streamed_audio_content")

    # 2. File path
    audio_file = tmp_path / "test_recording.wav"
    audio_file.write_bytes(b"file_audio_content")
    res_file = service.transcribe(audio_file)
    assert res_file == "Show me wireless noise cancelling headphones"
    assert service.last_audio_size == len(b"file_audio_content")

    # 3. Missing file path
    non_existent = tmp_path / "missing.wav"
    with pytest.raises(CustomException) as exc_missing:
        service.transcribe(non_existent)
    assert "Audio validation failed" in str(exc_missing.value)
