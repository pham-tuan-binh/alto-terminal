"""Unit tests for AudioConfig.

These tests verify the AudioConfig dataclass:
- Initialization with default values
- Initialization with custom values
- AEC and audio processing settings
- Validation of configuration values
"""

import pytest
from src.alto_terminal.config import AudioConfig, SAMPLE_RATE, NUM_CHANNELS, SAMPLES_PER_CHANNEL


class TestAudioConfigInitialization:
    """Test AudioConfig initialization."""

    def test_default_initialization(self):
        """Test initialization with all defaults."""
        config = AudioConfig()

        assert config.sample_rate == SAMPLE_RATE
        assert config.num_channels == NUM_CHANNELS
        assert config.samples_per_channel == SAMPLES_PER_CHANNEL
        assert config.input_device is None
        assert config.output_device is None
        assert config.volume == 1.0
        assert config.no_playback is False
        assert config.enable_aec is False
        assert config.noise_suppression is False
        assert config.high_pass_filter is False
        assert config.auto_gain_control is False

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        config = AudioConfig(
            sample_rate=16000,
            num_channels=2,
            samples_per_channel=1600,
            input_device=1,
            output_device=2,
            volume=0.5,
            no_playback=True
        )

        assert config.sample_rate == 16000
        assert config.num_channels == 2
        assert config.samples_per_channel == 1600
        assert config.input_device == 1
        assert config.output_device == 2
        assert config.volume == 0.5
        assert config.no_playback is True


class TestAudioProcessingSettings:
    """Test AEC and audio processing settings."""

    def test_enable_aec_only(self):
        """Test enabling only AEC."""
        config = AudioConfig(enable_aec=True)

        assert config.enable_aec is True
        assert config.noise_suppression is False
        assert config.high_pass_filter is False
        assert config.auto_gain_control is False

    def test_enable_noise_suppression_only(self):
        """Test enabling only noise suppression."""
        config = AudioConfig(noise_suppression=True)

        assert config.enable_aec is False
        assert config.noise_suppression is True
        assert config.high_pass_filter is False
        assert config.auto_gain_control is False

    def test_enable_high_pass_filter_only(self):
        """Test enabling only high-pass filter."""
        config = AudioConfig(high_pass_filter=True)

        assert config.enable_aec is False
        assert config.noise_suppression is False
        assert config.high_pass_filter is True
        assert config.auto_gain_control is False

    def test_enable_auto_gain_control_only(self):
        """Test enabling only auto gain control."""
        config = AudioConfig(auto_gain_control=True)

        assert config.enable_aec is False
        assert config.noise_suppression is False
        assert config.high_pass_filter is False
        assert config.auto_gain_control is True

    def test_enable_all_audio_processing(self):
        """Test enabling all audio processing features."""
        config = AudioConfig(
            enable_aec=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=True
        )

        assert config.enable_aec is True
        assert config.noise_suppression is True
        assert config.high_pass_filter is True
        assert config.auto_gain_control is True

    def test_enable_common_combination(self):
        """Test common combination: AEC + noise suppression."""
        config = AudioConfig(
            enable_aec=True,
            noise_suppression=True
        )

        assert config.enable_aec is True
        assert config.noise_suppression is True
        assert config.high_pass_filter is False
        assert config.auto_gain_control is False


class TestAudioConfigValidation:
    """Test configuration validation."""

    def test_invalid_sample_rate_negative(self):
        """Test that negative sample rate raises error."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            AudioConfig(sample_rate=-1000)

    def test_invalid_sample_rate_zero(self):
        """Test that zero sample rate raises error."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            AudioConfig(sample_rate=0)

    def test_valid_sample_rates(self):
        """Test various valid sample rates."""
        for rate in [8000, 16000, 44100, 48000, 96000]:
            config = AudioConfig(sample_rate=rate)
            assert config.sample_rate == rate

    def test_invalid_num_channels(self):
        """Test that invalid number of channels raises error."""
        with pytest.raises(ValueError, match="num_channels must be 1 or 2"):
            AudioConfig(num_channels=3)

        with pytest.raises(ValueError, match="num_channels must be 1 or 2"):
            AudioConfig(num_channels=0)

    def test_valid_num_channels(self):
        """Test valid channel configurations."""
        config_mono = AudioConfig(num_channels=1)
        assert config_mono.num_channels == 1

        config_stereo = AudioConfig(num_channels=2)
        assert config_stereo.num_channels == 2

    def test_invalid_samples_per_channel_negative(self):
        """Test that negative samples_per_channel raises error."""
        with pytest.raises(ValueError, match="samples_per_channel must be positive"):
            AudioConfig(samples_per_channel=-100)

    def test_invalid_samples_per_channel_zero(self):
        """Test that zero samples_per_channel raises error."""
        with pytest.raises(ValueError, match="samples_per_channel must be positive"):
            AudioConfig(samples_per_channel=0)

    def test_valid_samples_per_channel(self):
        """Test various valid samples_per_channel values."""
        for samples in [480, 960, 1600, 2400, 4800]:
            config = AudioConfig(samples_per_channel=samples)
            assert config.samples_per_channel == samples

    def test_invalid_volume_too_low(self):
        """Test that volume below 0.0 raises error."""
        with pytest.raises(ValueError, match="volume must be between 0.0 and 1.0"):
            AudioConfig(volume=-0.1)

    def test_invalid_volume_too_high(self):
        """Test that volume above 1.0 raises error."""
        with pytest.raises(ValueError, match="volume must be between 0.0 and 1.0"):
            AudioConfig(volume=1.1)

    def test_valid_volume_range(self):
        """Test valid volume values."""
        for volume in [0.0, 0.25, 0.5, 0.75, 1.0]:
            config = AudioConfig(volume=volume)
            assert config.volume == volume

    def test_volume_boundary_values(self):
        """Test volume at exact boundaries."""
        config_min = AudioConfig(volume=0.0)
        assert config_min.volume == 0.0

        config_max = AudioConfig(volume=1.0)
        assert config_max.volume == 1.0


class TestAudioConfigUseCases:
    """Test real-world configuration use cases."""

    def test_voice_ai_configuration(self):
        """Test typical voice AI configuration with AEC and noise suppression."""
        config = AudioConfig(
            sample_rate=48000,
            num_channels=1,
            enable_aec=True,
            noise_suppression=True
        )

        assert config.sample_rate == 48000
        assert config.num_channels == 1
        assert config.enable_aec is True
        assert config.noise_suppression is True

    def test_high_quality_conference_configuration(self):
        """Test high-quality conference configuration with all processing."""
        config = AudioConfig(
            sample_rate=48000,
            num_channels=1,
            enable_aec=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=True,
            volume=0.8
        )

        assert config.enable_aec is True
        assert config.noise_suppression is True
        assert config.high_pass_filter is True
        assert config.auto_gain_control is True
        assert config.volume == 0.8

    def test_capture_only_configuration(self):
        """Test capture-only configuration (no playback)."""
        config = AudioConfig(
            no_playback=True,
            enable_aec=False  # No AEC needed without playback
        )

        assert config.no_playback is True
        assert config.enable_aec is False

    def test_custom_devices_configuration(self):
        """Test configuration with custom audio devices."""
        config = AudioConfig(
            input_device=3,
            output_device=5,
            enable_aec=True
        )

        assert config.input_device == 3
        assert config.output_device == 5
        assert config.enable_aec is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
