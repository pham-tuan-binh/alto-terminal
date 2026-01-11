"""Audio mixer with per-stream control and metrics.

This module provides the AudioMixer class for mixing multiple audio streams
in real-time. It handles buffering, volume control, and graceful underruns.

Key design decisions:
- Uses threading.Lock (not asyncio.Lock) for audio callback compatibility
- Handles buffer underruns gracefully by padding with silence
- Tracks per-stream metrics for debugging and monitoring
- Simple FIFO buffering for minimal latency
"""

import threading
import logging
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StreamMetrics:
    """Metrics for a single audio stream.

    Attributes:
        samples_added: Total number of samples added to this stream
        buffer_size: Current buffer size for this stream
        underrun_count: Number of times this stream underran
    """

    samples_added: int = 0
    buffer_size: int = 0
    underrun_count: int = 0


class AudioMixer:
    """Mix multiple audio streams with per-stream control and metrics.

    This mixer is designed to handle real-time audio with minimal latency:
    - Uses threading.Lock for audio callback compatibility (not asyncio.Lock)
    - Tracks multiple streams independently with unique stream_id
    - Provides per-stream volume control
    - Reports detailed metrics for monitoring and debugging
    - Handles buffer underruns gracefully with silence padding

    Thread-safety: All public methods are thread-safe and can be called
    from both OS audio threads and async event loop threads.

    Example:
        >>> mixer = AudioMixer(48000, 1, master_volume=0.8)
        >>> mixer.add_stream("alice", volume=1.0)
        >>> mixer.add_audio_data(audio_data, stream_id="alice")
        >>> output = mixer.get_audio_data(2400)  # Get 50ms of audio
    """

    def __init__(self, sample_rate: int, channels: int, master_volume: float = 1.0):
        """Initialize the audio mixer.

        Args:
            sample_rate: Sample rate in Hz (e.g., 48000)
            channels: Number of audio channels (typically 1 for mono)
            master_volume: Global volume multiplier (0.0 to 1.0)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.master_volume = master_volume

        # Per-stream buffers: stream_id -> numpy array
        self._streams: Dict[str, np.ndarray] = {}

        # Per-stream volume: stream_id -> float
        self._stream_volumes: Dict[str, float] = {}

        # Per-stream metrics: stream_id -> StreamMetrics
        self._metrics: Dict[str, StreamMetrics] = {}

        # Global mixed buffer (FIFO)
        self._mixed_buffer = np.array([], dtype=np.int16)

        # Thread lock for all operations
        # Use threading.Lock (not asyncio.Lock) because audio callbacks
        # run in OS threads, not in the async event loop
        self._lock = threading.Lock()

        # Global underrun tracking
        self._total_underruns = 0
        self._last_underrun_log = 0

    def add_stream(self, stream_id: str, volume: float = 1.0):
        """Register a new audio stream.

        Args:
            stream_id: Unique identifier for the stream (e.g., participant ID)
            volume: Initial volume for this stream (0.0 to 1.0)
        """
        with self._lock:
            if stream_id in self._streams:
                logger.warning(f"Stream {stream_id} already exists, resetting")

            self._streams[stream_id] = np.array([], dtype=np.int16)
            self._stream_volumes[stream_id] = volume
            self._metrics[stream_id] = StreamMetrics()

        logger.info(f"Added audio stream: {stream_id} with volume {volume}")

    def remove_stream(self, stream_id: str):
        """Remove an audio stream.

        Args:
            stream_id: Stream identifier to remove
        """
        with self._lock:
            if stream_id in self._streams:
                del self._streams[stream_id]
                del self._stream_volumes[stream_id]
                del self._metrics[stream_id]
                logger.info(f"Removed audio stream: {stream_id}")

    def set_stream_volume(self, stream_id: str, volume: float):
        """Set volume for a specific stream.

        Args:
            stream_id: Stream identifier
            volume: Volume level (0.0 to 1.0)
        """
        with self._lock:
            if stream_id in self._stream_volumes:
                self._stream_volumes[stream_id] = np.clip(volume, 0.0, 1.0)
                logger.debug(f"Set stream {stream_id} volume to {volume}")

    def add_audio_data(self, audio_data: np.ndarray, stream_id: str = "default"):
        """Add audio data to a specific stream (thread-safe).

        Args:
            audio_data: Audio samples as int16 numpy array
            stream_id: Stream identifier (auto-creates if not exists)
        """
        with self._lock:
            # Auto-create stream if needed
            if stream_id not in self._streams:
                self._streams[stream_id] = np.array([], dtype=np.int16)
                self._stream_volumes[stream_id] = 1.0
                self._metrics[stream_id] = StreamMetrics()
                logger.info(f"Auto-created stream: {stream_id}")

            # Append to stream buffer
            if len(self._streams[stream_id]) == 0:
                self._streams[stream_id] = audio_data.copy()
            else:
                self._streams[stream_id] = np.concatenate([
                    self._streams[stream_id],
                    audio_data
                ])

            # Update metrics
            self._metrics[stream_id].samples_added += len(audio_data)
            self._metrics[stream_id].buffer_size = len(self._streams[stream_id])

    def _mix_streams(self):
        """Mix all stream buffers into the global playback buffer.

        This internal method must be called with _lock held.
        Uses simple averaging to prevent clipping when mixing multiple streams.

        The mixing algorithm:
        1. Find minimum buffer size across all streams
        2. Average samples from all streams (prevents clipping)
        3. Apply per-stream volume control
        4. Apply master volume
        5. Consume mixed samples from each stream
        """
        if not self._streams:
            return

        # Find minimum buffer size across all streams (only mix what all streams have)
        min_size = min(
            (len(buf) for buf in self._streams.values() if len(buf) > 0),
            default=0
        )
        if min_size == 0:
            return

        # Mix streams by averaging (prevents clipping)
        # Use float32 for intermediate calculations to avoid overflow
        mixed = np.zeros(min_size, dtype=np.float32)
        active_streams = 0

        for stream_id, buffer in self._streams.items():
            if len(buffer) >= min_size:
                # Apply per-stream volume
                volume = self._stream_volumes[stream_id]
                mixed += buffer[:min_size].astype(np.float32) * volume
                active_streams += 1

                # Consume mixed samples from stream
                self._streams[stream_id] = buffer[min_size:]
                self._metrics[stream_id].buffer_size = len(self._streams[stream_id])

        if active_streams > 0:
            # Average to prevent clipping, then apply master volume
            mixed = (mixed / active_streams) * self.master_volume
            # Clip to valid int16 range and convert
            mixed = np.clip(mixed, -32768, 32767).astype(np.int16)

            # Append to global mixed buffer
            if len(self._mixed_buffer) == 0:
                self._mixed_buffer = mixed
            else:
                self._mixed_buffer = np.concatenate([self._mixed_buffer, mixed])

    def get_audio_data(self, num_samples: int) -> np.ndarray:
        """Get audio data from the mixed buffer (thread-safe).

        This method is called from the audio output callback (real-time thread).
        It must return immediately with either audio data or silence.

        Args:
            num_samples: Number of samples to retrieve

        Returns:
            Audio samples as int16 numpy array (padded with silence if needed)
        """
        with self._lock:
            # Try to mix more data if buffer is low
            if len(self._mixed_buffer) < num_samples:
                self._mix_streams()

            if len(self._mixed_buffer) < num_samples:
                # Buffer underrun - we don't have enough mixed audio
                self._total_underruns += 1

                # Log occasionally to avoid log spam (every 100 underruns)
                if self._total_underruns - self._last_underrun_log >= 100:
                    logger.warning(
                        f"Audio buffer underrun #{self._total_underruns}. "
                        f"Buffer has {len(self._mixed_buffer)} samples, need {num_samples}"
                    )
                    self._last_underrun_log = self._total_underruns

                # Return available data padded with silence (graceful degradation)
                # This prevents audio glitches - silence is better than pops/clicks
                result = np.zeros(num_samples, dtype=np.int16)
                if len(self._mixed_buffer) > 0:
                    result[:len(self._mixed_buffer)] = self._mixed_buffer
                    self._mixed_buffer = np.array([], dtype=np.int16)
                return result
            else:
                # Extract requested samples from buffer
                result = self._mixed_buffer[:num_samples]
                self._mixed_buffer = self._mixed_buffer[num_samples:]
                return result

    def get_metrics(self, stream_id: Optional[str] = None) -> Dict:
        """Get metrics for monitoring.

        Args:
            stream_id: If provided, return metrics for specific stream.
                      If None, return global metrics with all stream metrics.

        Returns:
            Dictionary of metrics
        """
        with self._lock:
            if stream_id:
                # Return metrics for specific stream
                if stream_id in self._metrics:
                    metrics = self._metrics[stream_id]
                    return {
                        'stream_id': stream_id,
                        'samples_added': metrics.samples_added,
                        'buffer_size': len(self._streams.get(stream_id, [])),
                        'underrun_count': metrics.underrun_count,
                        'volume': self._stream_volumes.get(stream_id, 0.0)
                    }
                return {}
            else:
                # Return global metrics with all streams
                # Build stream metrics without recursive lock acquisition
                stream_metrics_dict = {}
                for sid in self._streams.keys():
                    if sid in self._metrics:
                        metrics = self._metrics[sid]
                        stream_metrics_dict[sid] = {
                            'stream_id': sid,
                            'samples_added': metrics.samples_added,
                            'buffer_size': len(self._streams.get(sid, [])),
                            'underrun_count': metrics.underrun_count,
                            'volume': self._stream_volumes.get(sid, 0.0)
                        }

                return {
                    'mixed_buffer_size': len(self._mixed_buffer),
                    'total_underruns': self._total_underruns,
                    'active_streams': len(self._streams),
                    'master_volume': self.master_volume,
                    'stream_metrics': stream_metrics_dict
                }

    def clear(self, stream_id: Optional[str] = None):
        """Clear buffers.

        Args:
            stream_id: If provided, clear only that stream.
                      If None, clear all buffers including mixed buffer.
        """
        with self._lock:
            if stream_id:
                if stream_id in self._streams:
                    self._streams[stream_id] = np.array([], dtype=np.int16)
                    self._metrics[stream_id].buffer_size = 0
            else:
                # Clear all stream buffers
                for sid in self._streams:
                    self._streams[sid] = np.array([], dtype=np.int16)
                    self._metrics[sid].buffer_size = 0
                # Clear mixed buffer
                self._mixed_buffer = np.array([], dtype=np.int16)

    def buffer_size(self) -> int:
        """Get current mixed buffer size (in samples).

        Returns:
            Number of samples in the mixed buffer ready for playback
        """
        with self._lock:
            return len(self._mixed_buffer)
