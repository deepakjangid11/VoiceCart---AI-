import sys
import time
from typing import Optional

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger
from e_commerce_voice_bot.services.audio_capture import AudioCapture
from e_commerce_voice_bot.services.stt_service import ElevenLabsSTTService

logger = get_logger(__name__)


class VoiceInputService:
    """Voice input orchestration service combining microphone audio capture and STT."""

    def __init__(
        self,
        audio_capture: Optional[AudioCapture] = None,
        stt_service: Optional[ElevenLabsSTTService] = None,
    ) -> None:
        self.audio_capture = audio_capture or AudioCapture()
        self.stt_service = stt_service or ElevenLabsSTTService()

        # Timing metrics
        self.last_audio_capture_time: float = 0.0
        self.last_stt_time: float = 0.0
        self.last_voice_input_total_time: float = 0.0
        self.last_audio_duration_ms: float = 0.0
        self.last_audio_size_bytes: int = 0

    def start_listening(self) -> None:
        """Start microphone recording."""
        try:
            self.audio_capture.start_recording()
        except Exception as e:
            logger.exception("Voice input start recording failed: %s", e)
            if isinstance(e, CustomException):
                raise
            raise CustomException(f"Microphone start failure: {e}", sys) from e

    def stop_and_transcribe(self) -> str:
        """Stop microphone recording and transcribe captured audio using STTService."""
        t0_total = time.perf_counter()
        t0_capture = time.perf_counter()

        audio_bytes = b""
        try:
            audio_bytes = self.audio_capture.stop_recording(as_wav=True)
            self.last_audio_capture_time = time.perf_counter() - t0_capture
        except Exception as e:
            self.last_audio_capture_time = time.perf_counter() - t0_capture
            logger.exception("Voice input stop recording failed: %s", e)
            if isinstance(e, CustomException):
                raise
            raise CustomException(f"Microphone stop failure: {e}", sys) from e

        # Validate captured audio payload
        self.last_audio_size_bytes = len(audio_bytes)
        if not audio_bytes:
            logger.error("Audio capture returned empty byte stream.")
            raise CustomException("Audio capture returned empty audio.", sys)

        # Estimate duration in ms (16kHz mono 16-bit PCM = 32,000 bytes/sec + 44 byte header)
        pcm_bytes_len = max(0, self.last_audio_size_bytes - 44)
        bytes_per_sec = self.audio_capture.sample_rate * self.audio_capture.channels * 2
        self.last_audio_duration_ms = (pcm_bytes_len / bytes_per_sec) * 1000.0 if bytes_per_sec else 0.0

        # Transcribe audio with existing STTService
        t0_stt = time.perf_counter()
        try:
            transcript = self.stt_service.transcribe(audio_bytes)
            self.last_stt_time = time.perf_counter() - t0_stt
        except Exception as e:
            self.last_stt_time = time.perf_counter() - t0_stt
            logger.exception("STT transcription failure in voice input service: %s", e)
            if isinstance(e, CustomException):
                raise
            raise CustomException(f"Voice input transcription failed: {e}", sys) from e

        self.last_voice_input_total_time = time.perf_counter() - t0_total

        # Structured timing log (NEVER logs raw audio or sensitive data)
        logger.info(
            "Voice input completed | "
            "audio_duration_ms=%.1f | audio_size_bytes=%d | "
            "stt_time=%.4fs | total_time=%.4fs",
            self.last_audio_duration_ms,
            self.last_audio_size_bytes,
            self.last_stt_time,
            self.last_voice_input_total_time,
        )

        return transcript

    def capture_and_transcribe(self, duration_seconds: float = 3.0) -> str:
        """Convenience method to capture microphone audio for a set duration and transcribe.

        Args:
            duration_seconds: Number of seconds to record from microphone before transcribing.

        Returns:
            Clean transcript string.
        """
        if duration_seconds <= 0:
            raise CustomException("Recording duration must be greater than 0.", sys)

        try:
            self.start_listening()
            time.sleep(duration_seconds)
            return self.stop_and_transcribe()
        except Exception as e:
            # Ensure microphone resources are released on error
            if self.audio_capture.is_recording:
                try:
                    self.audio_capture.stop_recording()
                except Exception:
                    pass
            logger.exception("capture_and_transcribe failed: %s", e)
            if isinstance(e, CustomException):
                raise
            raise CustomException(f"Voice capture and transcription failed: {e}", sys) from e
