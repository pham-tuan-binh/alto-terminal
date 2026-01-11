"""Audio output handler for speaker playback.

This module provides the AudioOutputHandler class for playing audio from
the mixer to speakers via sounddevice.

Key design decisions:
- Runs in OS audio thread (sounddevice callback)
- Pulls audio from AudioMixer in real-time
- Converts int16 (mixer) → float32 (sounddevice)
- Handles underruns gracefully (mixer returns silence)
- Tracks metrics for monitoring (frames played, underruns, errors)
"""

import logging
import numpy as np
import sounddevice as sd
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OutputMetrics:
    """Metrics for audio output handler.

    Attributes:
        frames_played: Total number of audio frames played
        underruns: Total number of buffer underruns
        errors: Total number of errors encountered
        last_error: Description of the most recent error
    """
    frames_played: int = 0
    underruns: int = 0
    errors: int = 0
    last_error: Optional[str] = None


class AudioOutputHandler:
    """Handle audio output to speakers.

    This handler encapsulates speaker playback logic, pulling audio from
    the AudioMixer and converting it to the format required by sounddevice.

    Thread model:
    - Audio callback runs in OS audio thread (high priority, real-time)
    - Must complete FAST (no blocking operations)
    - Pulls from AudioMixer (thread-safe, synchronous)

    Example:
        >>> mixer = AudioMixer(48000, 1)
        >>> handler = AudioOutputHandler(
        ...     mixer=mixer,
        ...     sample_rate=48000,
        ...     num_channels=1,
        ...     samples_per_channel=2400,
        ...     device_index=None
        ... )
        >>> handler.start()
        >>> # ... audio is now being played ...
        >>> handler.stop()
    """

    def __init__(
        self,
        mixer,  # AudioMixer instance (avoid circular import by not type hinting)
        sample_rate: int,
        num_channels: int,
        samples_per_channel: int,
        device_index: Optional[int] = None
    ):
        """Initialize audio output handler.

        Args:
            mixer: AudioMixer instance to pull audio from
            sample_rate: Sample rate in Hz (e.g., 48000)
            num_channels: Number of audio channels (typically 1 for mono)
            samples_per_channel: Number of samples per audio frame (e.g., 2400 for 50ms)
            device_index: Optional device index (None = default device)
        """
        self.mixer = mixer
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel
        self.device_index = device_index

        # Output stream (created when started)
        self._output_stream: Optional[sd.OutputStream] = None

        # Metrics
        self._metrics = OutputMetrics()

        logger.info(
            f"Initialized AudioOutputHandler: {sample_rate}Hz, "
            f"{num_channels}ch, {samples_per_channel} samples/frame, "
            f"device={device_index}"
        )

    def start(self):
        """Start playing audio to speakers.

        This creates and starts a sounddevice OutputStream.
        The audio callback will run in a high-priority OS audio thread.
        """
        if self._output_stream is not None:
            logger.warning("Output stream already started")
            return

        try:
            # Create output stream
            # blocksize: Number of frames per callback (50ms chunks)
            # dtype: float32 is sounddevice's native format
            self._output_stream = sd.OutputStream(
                device=self.device_index,
                channels=self.num_channels,
                samplerate=self.sample_rate,
                blocksize=self.samples_per_channel,
                dtype='float32',
                callback=self._audio_callback
            )

            self._output_stream.start()
            logger.info("Audio output started")

        except Exception as e:
            logger.error(f"Failed to start audio output: {e}")
            self._metrics.errors += 1
            self._metrics.last_error = str(e)
            raise

    def stop(self):
        """Stop playing audio to speakers.

        This stops and closes the sounddevice OutputStream.
        Blocks until the stream is fully closed.
        """
        if self._output_stream is None:
            logger.warning("Output stream not started")
            return

        try:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None
            logger.info("Audio output stopped")

        except Exception as e:
            logger.error(f"Failed to stop audio output: {e}")
            self._metrics.errors += 1
            self._metrics.last_error = str(e)
            raise

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback (runs in OS audio thread).

        This callback is invoked by sounddevice in a high-priority OS thread.
        It MUST complete quickly to avoid audio glitches.

        Flow:
        1. Check for errors (status parameter)
        2. Pull audio from mixer (thread-safe, synchronous)
        3. Convert int16 → float32
        4. Write to output buffer

        Args:
            outdata: Output buffer to fill (shape: [frames, channels])
            frames: Number of frames to generate
            time_info: Timing information (not used)
            status: Error status (if any)

        Note:
            This runs in OS thread, NOT the async event loop.
            Mixer.get_audio_data() is thread-safe and synchronous.
        """
        # Log any output errors
        if status:
            logger.warning(f"Audio output status: {status}")
            self._metrics.errors += 1
            self._metrics.last_error = str(status)

        try:
            # Pull audio from mixer (thread-safe)
            # Mixer handles underruns gracefully by returning silence
            audio_data = self.mixer.get_audio_data(frames)

            # Check if we got silence (underrun)
            if len(audio_data) == frames and np.all(audio_data == 0):
                self._metrics.underruns += 1

            # Convert int16 (range -32768 to 32767) → float32 (range -1.0 to 1.0)
            # Mixer uses int16 with range [-32768, 32767]
            # Sounddevice expects float32 with range [-1.0, 1.0]
            # Divide by 32767 (not 32768) to match input conversion
            outdata[:, 0] = audio_data.astype(np.float32) / 32767.0

            # Update metrics
            self._metrics.frames_played += 1

        except Exception as e:
            # Log error but don't raise (audio thread must continue)
            logger.error(f"Error in audio output callback: {e}")
            self._metrics.errors += 1
            self._metrics.last_error = str(e)

            # Fill with silence on error (prevents pops/clicks)
            outdata.fill(0)

    def get_metrics(self) -> dict:
        """Get metrics for monitoring.

        Returns:
            Dictionary with metrics:
                - frames_played: Total frames played
                - underruns: Total buffer underruns detected
                - errors: Total errors encountered
                - last_error: Description of most recent error (or None)
                - is_active: Whether output stream is currently active
        """
        return {
            'frames_played': self._metrics.frames_played,
            'underruns': self._metrics.underruns,
            'errors': self._metrics.errors,
            'last_error': self._metrics.last_error,
            'is_active': self._output_stream is not None and self._output_stream.active
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
