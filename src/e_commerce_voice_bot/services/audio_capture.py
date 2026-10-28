import io
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Optional, Union
import wave

import sounddevice as sd

from e_commerce_voice_bot.exception import CustomException
from e_commerce_voice_bot.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DTYPE = "int16"


def pcm_to_wav_bytes(
    pcm_bytes: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> bytes:
    """Convert raw 16-bit PCM bytes into in-memory WAV formatted bytes without disk I/O."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return wav_buffer.getvalue()


class AudioCapture:
    """Microphone audio capture service with explicit lifecycle and resource cleanup."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        dtype: str = DEFAULT_DTYPE,
        device: Optional[Union[int, str]] = None,
        stream_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if sample_rate <= 0:
            raise CustomException("Invalid audio configuration: sample_rate must be > 0.", sys)
        if channels <= 0:
            raise CustomException("Invalid audio configuration: channels must be > 0.", sys)

        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self._stream_factory = stream_factory or sd.RawInputStream

        self._stream: Optional[Any] = None
        self._buffer: list[bytes] = []
        self._is_recording = False
        self._lock = threading.RLock()

        self.last_capture_time: float = 0.0
        self.last_audio_size_bytes: int = 0

    @property
    def is_recording(self) -> bool:
        """Return True if microphone stream is actively recording."""
        return self._is_recording

    def _audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """Internal callback invoked by the audio stream for captured audio chunks."""
        if status:
            logger.warning("Audio capture stream status warning: %s", status)
        with self._lock:
            if self._is_recording:
                self._buffer.append(bytes(indata))

    def start_recording(self) -> None:
        """Initialize the audio input stream and begin capturing microphone frames."""
        with self._lock:
            if self._is_recording:
                logger.warning("start_recording called while already recording.")
                return

            self._buffer.clear()
            self._start_time = time.perf_counter()

            try:
                logger.info(
                    "Starting audio capture | sample_rate=%d | channels=%d | dtype=%s",
                    self.sample_rate,
                    self.channels,
                    self.dtype,
                )
                self._stream = self._stream_factory(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    device=self.device,
                    callback=self._audio_callback,
                )
                self._is_recording = True
                self._stream.start()
                logger.info("Microphone stream started successfully.")

            except Exception as e:
                self._is_recording = False
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                logger.exception("Failed to start microphone recording: %s", e)
                raise CustomException(f"Microphone recording failure: {e}", sys) from e

    def stop_recording(self, as_wav: bool = True) -> bytes:
        """Stop microphone capture, safely release stream resources, and return audio bytes."""
        with self._lock:
            if not self._is_recording and self._stream is None:
                logger.warning("stop_recording called when not recording.")
                return b""

            t_stop = time.perf_counter()
            self.last_capture_time = t_stop - getattr(self, "_start_time", t_stop)
            self._is_recording = False

            try:
                if self._stream is not None:
                    try:
                        self._stream.stop()
                    finally:
                        self._stream.close()
                        self._stream = None
                        logger.info("Microphone stream closed and released successfully.")

                raw_pcm = b"".join(self._buffer)
                self._buffer.clear()

                if as_wav and raw_pcm:
                    audio_bytes = pcm_to_wav_bytes(
                        raw_pcm,
                        sample_rate=self.sample_rate,
                        channels=self.channels,
                    )
                else:
                    audio_bytes = raw_pcm

                self.last_audio_size_bytes = len(audio_bytes)
                logger.info(
                    "Audio recording stopped | duration=%.4fs | size=%d bytes | wav=%s",
                    self.last_capture_time,
                    self.last_audio_size_bytes,
                    as_wav,
                )
                return audio_bytes

            except Exception as e:
                logger.exception("Failed during stop_recording resource cleanup: %s", e)
                raise CustomException(f"Recording stop failure: {e}", sys) from e
            finally:
                if self._stream is not None:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                self._is_recording = False

    def __enter__(self) -> "AudioCapture":
        self.start_recording()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._is_recording:
            try:
                self.stop_recording()
            except Exception:
                pass
