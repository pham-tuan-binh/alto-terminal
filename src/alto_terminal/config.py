"""Audio configuration constants and dataclass.

This module defines the audio parameters used throughout the application:
- Sample rate: 48kHz (standard for real-time audio, good quality)
- Channels: Mono (sufficient for voice, reduces bandwidth)
- Block size: 50ms chunks (balance between latency and CPU usage)
"""

from dataclasses import dataclass
from typing import Optional

# Audio configuration constants
SAMPLE_RATE = 48000  # 48kHz sample rate (high quality, low latency)
NUM_CHANNELS = 1  # Mono audio (voice focus, lower bandwidth)
SAMPLES_PER_CHANNEL = SAMPLE_RATE // 20  # 50ms chunks = 2400 samples @ 48kHz


@dataclass
class AudioConfig:
    """Configuration for audio processing.

    This dataclass encapsulates all audio-related configuration options,
    making it easy to pass configuration between components.

    Attributes:
        sample_rate: Sample rate in Hz (default: 48000)
        num_channels: Number of audio channels, 1=mono, 2=stereo (default: 1)
        samples_per_channel: Samples per channel per frame (default: 2400 = 50ms)
        input_device: Audio input device index, None for default (default: None)
        output_device: Audio output device index, None for default (default: None)
        volume: Master playback volume 0.0-1.0 (default: 1.0)
        no_playback: Disable audio playback (capture only) (default: False)
    """

    sample_rate: int = SAMPLE_RATE
    num_channels: int = NUM_CHANNELS
    samples_per_channel: int = SAMPLES_PER_CHANNEL
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    volume: float = 1.0
    no_playback: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.num_channels not in (1, 2):
            raise ValueError(f"num_channels must be 1 or 2, got {self.num_channels}")
        if self.samples_per_channel <= 0:
            raise ValueError(f"samples_per_channel must be positive, got {self.samples_per_channel}")
        if not 0.0 <= self.volume <= 1.0:
            raise ValueError(f"volume must be between 0.0 and 1.0, got {self.volume}")
