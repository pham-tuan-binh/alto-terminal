"""Audio input handler for microphone capture.

This module provides the AudioInputHandler class for capturing audio from
a microphone and sending it to LiveKit via an AudioSource.

Key design decisions:
- Runs in OS audio thread (sounddevice callback)
- Bridges to async event loop using asyncio.run_coroutine_threadsafe()
- Converts float32 (sounddevice) → int16 (LiveKit)
- Tracks metrics for monitoring (frames captured, errors)
"""

import asyncio
import logging
import numpy as np
import sounddevice as sd
from livekit import rtc
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class InputMetrics:
    """Metrics for audio input handler.

    Attributes:
        frames_captured: Total number of audio frames captured
        errors: Total number of errors encountered
        last_error: Description of the most recent error
    """
    frames_captured: int = 0
    errors: int = 0
    last_error: Optional[str] = None


class AudioInputHandler:
    """Handle audio input from microphone.

    This handler encapsulates microphone capture logic and bridges the
    OS audio thread (sounddevice) with the async event loop (LiveKit).

    Thread model:
    - Audio callback runs in OS audio thread (high priority, real-time)
    - Must complete FAST (no blocking operations)
    - Uses asyncio.run_coroutine_threadsafe() to bridge to event loop

    Example:
        >>> handler = AudioInputHandler(
        ...     audio_source=livekit_audio_source,
        ...     event_loop=asyncio.get_event_loop(),
        ...     sample_rate=48000,
        ...     num_channels=1,
        ...     samples_per_channel=2400,
        ...     device_index=None
        ... )
        >>> handler.start()
        >>> # ... audio is now being captured ...
        >>> handler.stop()
    """

    def __init__(
        self,
        audio_source: rtc.AudioSource,
        event_loop: asyncio.AbstractEventLoop,
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
        device_index: Optional[int] = None,
        on_audio_captured: Optional[Callable[[np.ndarray], None]] = None
    ):
        """Initialize audio input handler.

        Args:
            audio_source: LiveKit AudioSource to capture frames to
            event_loop: Async event loop for LiveKit operations
            sample_rate: Sample rate in Hz (e.g., 48000)
            num_channels: Number of audio channels (typically 1 for mono)
            samples_per_channel: Number of samples per audio frame (e.g., 2400 for 50ms)
            device_index: Optional device index (None = default device)
            on_audio_captured: Optional callback for captured audio data (for visualization)
                               Called with int16 audio data from OS audio thread
        """
        self.audio_source = audio_source
        self.event_loop = event_loop
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel
        self.device_index = device_index
        self._on_audio_captured = on_audio_captured

        # Input stream (created when started)
        self._input_stream: Optional[sd.InputStream] = None

        # Metrics
        self._metrics = InputMetrics()

        logger.info(
            f"Initialized AudioInputHandler: {sample_rate}Hz, "
            f"{num_channels}ch, {samples_per_channel} samples/frame, "
            f"device={device_index}"
        )

    def start(self):
        """Start capturing audio from microphone.

        This creates and starts a sounddevice InputStream.
        The audio callback will run in a high-priority OS audio thread.
        """
        if self._input_stream is not None:
            logger.warning("Input stream already started")
            return

        try:
            # Create input stream
            # blocksize: Number of frames per callback (50ms chunks)
            # dtype: float32 is sounddevice's native format
            self._input_stream = sd.InputStream(
                device=self.device_index,
                channels=self.num_channels,
                samplerate=self.sample_rate,
                blocksize=self.samples_per_channel,
                dtype='float32',
                callback=self._audio_callback
            )

            self._input_stream.start()
            logger.info("Audio input started")

        except Exception as e:
            logger.error(f"Failed to start audio input: {e}")
            self._metrics.errors += 1
            self._metrics.last_error = str(e)
            raise

    def stop(self):
        """Stop capturing audio from microphone.

        This stops and closes the sounddevice InputStream.
        Blocks until the stream is fully closed.
        """
        if self._input_stream is None:
            logger.warning("Input stream not started")
            return

        try:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
            logger.info("Audio input stopped")

        except Exception as e:
            logger.error(f"Failed to stop audio input: {e}")
            self._metrics.errors += 1
            self._metrics.last_error = str(e)
            raise

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback (runs in OS audio thread).

        This callback is invoked by sounddevice in a high-priority OS thread.
        It MUST complete quickly to avoid audio glitches.

        Flow:
        1. Check for errors (status parameter)
        2. Convert float32 → int16
        3. Create LiveKit AudioFrame
        4. Bridge to async event loop (non-blocking)

        Args:
            indata: Audio samples as float32 numpy array (shape: [frames, channels])
            frames: Number of frames captured
            time_info: Timing information (not used)
            status: Error status (if any)

        Note:
            This runs in OS thread, NOT the async event loop.
            Use asyncio.run_coroutine_threadsafe() to bridge to async.
        """
        # Log any input errors
        if status:
            logger.warning(f"Audio input status: {status}")
            self._metrics.errors += 1
            self._metrics.last_error = str(status)

        try:
            # Convert float32 (range -1.0 to 1.0) → int16 (range -32768 to 32767)
            # Sounddevice uses float32 with range [-1.0, 1.0]
            # LiveKit expects int16 with range [-32768, 32767]
            # Multiply by 32767 (not 32768) to avoid overflow at -1.0
            audio_data = (indata[:, 0] * 32767).astype(np.int16)

            # Notify visualization callback if provided (e.g., for TUI)
            # This runs in OS audio thread, so callback must be thread-safe
            if self._on_audio_captured:
                try:
                    self._on_audio_captured(audio_data)
                except Exception as e:
                    # Don't let visualization errors crash audio capture
                    logger.debug(f"Error in audio capture callback: {e}")

            # Create LiveKit AudioFrame
            frame = rtc.AudioFrame(
                data=audio_data.tobytes(),  # Convert to bytes
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
                samples_per_channel=len(audio_data)
            )

            # Bridge to async event loop (non-blocking)
            # This schedules the coroutine to run in the event loop thread
            # and returns immediately (does not block the audio thread)
            asyncio.run_coroutine_threadsafe(
                self.audio_source.capture_frame(frame),
                self.event_loop
            )

            # Update metrics
            self._metrics.frames_captured += 1

        except Exception as e:
            # Log error but don't raise (audio thread must continue)
            logger.error(f"Error in audio input callback: {e}")
            self._metrics.errors += 1
            self._metrics.last_error = str(e)

    def get_metrics(self) -> dict:
        """Get metrics for monitoring.

        Returns:
            Dictionary with metrics:
                - frames_captured: Total frames captured
                - errors: Total errors encountered
                - last_error: Description of most recent error (or None)
                - is_active: Whether input stream is currently active
        """
        return {
            'frames_captured': self._metrics.frames_captured,
            'errors': self._metrics.errors,
            'last_error': self._metrics.last_error,
            'is_active': self._input_stream is not None and self._input_stream.active
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
