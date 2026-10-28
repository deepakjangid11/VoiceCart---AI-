import io
import os
from pathlib import Path
import sys
import time
from typing import Any, BinaryIO, Optional, Union

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

DEFAULT_STT_MODEL = "scribe_v1"

AudioInputType = Union[bytes, bytearray, BinaryIO, Path, str]


class ElevenLabsSTTService:
    """Isolated Speech-to-Text service utilizing ElevenLabs Scribe API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        # Load API key securely
        self._api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self._api_key or not self._api_key.strip():
            logger.error("Configuration error: ELEVENLABS_API_KEY is missing or empty.")
            raise CustomException(
                "ELEVENLABS_API_KEY is missing. Add it to your .env file or pass it to the service.",
                sys,
            )

        # Retained for future TTS compatibility as per project requirements
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")

        # Configurable model ID (defaults to scribe_v1)
        self.model_id = model_id or os.getenv("ELEVENLABS_STT_MODEL", DEFAULT_STT_MODEL)

        # Timing metrics
        self.last_stt_time: float = 0.0
        self.last_audio_size: int = 0
        self.last_transcript_length: int = 0

        # Provider client injection or creation
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            return ElevenLabs(api_key=self._api_key)
        except Exception as e:
            logger.exception("Failed to initialize ElevenLabs client: %s", e)
            raise CustomException(f"Failed to initialize ElevenLabs client: {e}", sys) from e

    def _normalize_audio_payload(
        self,
        audio_data: AudioInputType,
    ) -> tuple[Any, int]:
        """Validate and prepare audio input into a payload suitable for ElevenLabs API."""
        if audio_data is None:
            raise ValueError("Audio data cannot be None.")

        if isinstance(audio_data, (bytes, bytearray)):
            byte_len = len(audio_data)
            if byte_len == 0:
                raise ValueError("Audio bytes payload cannot be empty.")
            return ("audio.wav", io.BytesIO(audio_data), "audio/wav"), byte_len

        if isinstance(audio_data, (str, Path)):
            audio_path = Path(audio_data)
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            file_bytes = audio_path.read_bytes()
            byte_len = len(file_bytes)
            if byte_len == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            return (audio_path.name, io.BytesIO(file_bytes), "audio/wav"), byte_len

        if hasattr(audio_data, "read"):
            content = audio_data.read()
            if hasattr(audio_data, "seek"):
                audio_data.seek(0)
            byte_len = len(content)
            if byte_len == 0:
                raise ValueError("Audio stream payload is empty.")
            return ("audio.wav", io.BytesIO(content), "audio/wav"), byte_len

        raise TypeError(f"Unsupported audio data type: {type(audio_data).__name__}")

    def transcribe(self, audio_data: AudioInputType) -> str:
        """Transcribe audio into text using ElevenLabs Speech-to-Text API.

        Args:
            audio_data: Raw audio bytes, bytearray, file path, or file-like binary stream.

        Returns:
            Clean transcript string.
        """
        t0 = time.perf_counter()

        # Step 1: Input Validation
        try:
            payload, audio_size = self._normalize_audio_payload(audio_data)
        except Exception as e:
            self.last_stt_time = time.perf_counter() - t0
            logger.error("Audio validation rejected: %s", e)
            raise CustomException(f"Audio validation failed: {e}", sys) from e

        self.last_audio_size = audio_size

        # Step 2: ElevenLabs STT API Call
        try:
            client = self._get_client()
            logger.info(
                "Starting STT transcription | model=%s | audio_size=%d bytes",
                self.model_id,
                audio_size,
            )

            response = client.speech_to_text.convert(
                file=payload,
                model_id=self.model_id,
            )

            # Extract clean transcript text from response
            transcript: str = ""
            if hasattr(response, "text") and response.text is not None:
                transcript = str(response.text)
            elif isinstance(response, dict) and "text" in response:
                transcript = str(response["text"])
            elif hasattr(response, "transcripts") and response.transcripts:
                transcript = " ".join(
                    str(getattr(t, "text", "")) for t in response.transcripts
                )
            else:
                transcript = str(response)

            clean_transcript = transcript.strip()
            self.last_stt_time = time.perf_counter() - t0
            self.last_transcript_length = len(clean_transcript)

            # Step 3: Structured Timing Logs (Never log raw audio or API keys)
            logger.info(
                "STT transcription completed | stt_time=%.4fs | audio_size=%d | transcript_length=%d",
                self.last_stt_time,
                self.last_audio_size,
                self.last_transcript_length,
            )

            return clean_transcript

        except Exception as e:
            self.last_stt_time = time.perf_counter() - t0
            logger.exception("STT transcription failed: %s", e)
            if isinstance(e, CustomException):
                raise
            raise CustomException(f"STT transcription failed: {e}", sys) from e


# Clean public interface alias
STTService = ElevenLabsSTTService
