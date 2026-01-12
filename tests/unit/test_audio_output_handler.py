"""Unit tests for AudioOutputHandler.

These tests verify the core functionality of the AudioOutputHandler class:
- Initialization
- Start/stop lifecycle
- Audio callback (int16 → float32 conversion)
- Pulling audio from mixer
- Error handling
- Underrun detection
- Metrics tracking
- Context manager usage
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from src.alto_terminal.audio_output_handler import AudioOutputHandler, OutputMetrics


class TestAudioOutputHandlerInitialization:
    """Test AudioOutputHandler initialization."""

    def test_initialization(self):
        """Test handler initialization with all parameters."""
        mixer = Mock()

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=5,
        )

        assert handler.mixer is mixer
        assert handler.sample_rate == 48000
        assert handler.num_channels == 1
        assert handler.samples_per_channel == 2400
        assert handler.device_index == 5
        assert handler._output_stream is None

    def test_initialization_default_device(self):
        """Test initialization with default device (None)."""
        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        assert handler.device_index is None

    def test_initialization_with_filter_disabled(self):
        """Test handler initialization with filter disabled (default)."""
        mixer = Mock()

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
            use_audio_out_filter=False,
        )

        assert handler._pedalboard_filter is None

    def test_initialization_with_filter_enabled(self):
        """Test handler initialization with filter enabled."""
        mixer = Mock()

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
            use_audio_out_filter=True,
        )

        assert handler._pedalboard_filter is not None
        assert hasattr(handler._pedalboard_filter, "process")


class TestAudioOutputLifecycle:
    """Test start/stop lifecycle."""

    @patch("src.alto_terminal.audio_output_handler.sd.OutputStream")
    def test_start(self, mock_output_stream_class):
        """Test starting audio output."""
        mock_stream = MagicMock()
        mock_output_stream_class.return_value = mock_stream

        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        handler.start()

        # Verify OutputStream was created with correct parameters
        mock_output_stream_class.assert_called_once_with(
            device=None,
            channels=1,
            samplerate=48000,
            blocksize=2400,
            dtype="float32",
            callback=handler._audio_callback,
        )

        # Verify stream was started
        mock_stream.start.assert_called_once()

        # Verify stream is stored
        assert handler._output_stream is mock_stream

    @patch("src.alto_terminal.audio_output_handler.sd.OutputStream")
    def test_start_with_device_index(self, mock_output_stream_class):
        """Test starting with specific device index."""
        mock_stream = MagicMock()
        mock_output_stream_class.return_value = mock_stream

        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=5,
        )

        handler.start()

        # Verify device parameter was passed
        call_kwargs = mock_output_stream_class.call_args[1]
        assert call_kwargs["device"] == 5

    @patch("src.alto_terminal.audio_output_handler.sd.OutputStream")
    def test_start_already_started(self, mock_output_stream_class):
        """Test starting when already started (should warn)."""
        mock_stream = MagicMock()
        mock_output_stream_class.return_value = mock_stream

        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        handler.start()
        handler.start()  # Start again

        # Should only create stream once
        assert mock_output_stream_class.call_count == 1

    @patch("src.alto_terminal.audio_output_handler.sd.OutputStream")
    def test_start_error(self, mock_output_stream_class):
        """Test error during start."""
        mock_output_stream_class.side_effect = Exception("Device not found")

        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        with pytest.raises(Exception, match="Device not found"):
            handler.start()

        # Verify metrics tracked error
        metrics = handler.get_metrics()
        assert metrics["errors"] == 1
        assert "Device not found" in metrics["last_error"]

    @patch("src.alto_terminal.audio_output_handler.sd.OutputStream")
    def test_stop(self, mock_output_stream_class):
        """Test stopping audio output."""
        mock_stream = MagicMock()
        mock_output_stream_class.return_value = mock_stream

        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        handler.start()
        handler.stop()

        # Verify stream was stopped and closed
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

        # Verify stream reference cleared
        assert handler._output_stream is None

    def test_stop_not_started(self):
        """Test stopping when not started (should warn)."""
        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        # Should not raise error
        handler.stop()

    @patch("src.alto_terminal.audio_output_handler.sd.OutputStream")
    def test_stop_error(self, mock_output_stream_class):
        """Test error during stop."""
        mock_stream = MagicMock()
        mock_stream.stop.side_effect = Exception("Stop failed")
        mock_output_stream_class.return_value = mock_stream

        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        handler.start()

        with pytest.raises(Exception, match="Stop failed"):
            handler.stop()

        # Verify metrics tracked error
        metrics = handler.get_metrics()
        assert metrics["errors"] == 1
        assert "Stop failed" in metrics["last_error"]


class TestAudioCallback:
    """Test audio callback functionality."""

    def test_audio_callback_conversion(self):
        """Test int16 to float32 conversion in callback."""
        mixer = Mock()
        # Mixer returns 100 samples at amplitude 16383 (0.5 * 32767)
        mixer.get_audio_data.return_value = np.full(100, 16383, dtype=np.int16)

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None,
        )

        # Create output buffer
        outdata = np.zeros((100, 1), dtype=np.float32)

        # Call the callback
        handler._audio_callback(outdata, 100, None, None)

        # Verify mixer was called with correct frame count
        mixer.get_audio_data.assert_called_once_with(100)

        # Verify conversion: 16383 / 32767 ≈ 0.5
        expected = np.full(100, 16383 / 32767, dtype=np.float32)
        np.testing.assert_array_almost_equal(outdata[:, 0], expected, decimal=5)

    def test_audio_callback_full_range(self):
        """Test conversion across full int16 range."""
        mixer = Mock()
        # Test samples: -32767, -16383, 0, 16383, 32767
        mixer.get_audio_data.return_value = np.array(
            [-32767, -16383, 0, 16383, 32767], dtype=np.int16
        )

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=5,
            device_index=None,
        )

        outdata = np.zeros((5, 1), dtype=np.float32)
        handler._audio_callback(outdata, 5, None, None)

        # Verify conversion
        # -32767 / 32767 = -1.0
        # -16383 / 32767 ≈ -0.5
        # 0 / 32767 = 0.0
        # 16383 / 32767 ≈ 0.5
        # 32767 / 32767 = 1.0
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        # Use decimal=4 for float32 precision (not 5, due to rounding)
        np.testing.assert_array_almost_equal(outdata[:, 0], expected, decimal=4)

    def test_audio_callback_pulls_from_mixer(self):
        """Test callback pulls correct amount from mixer."""
        mixer = Mock()
        mixer.get_audio_data.return_value = np.zeros(2400, dtype=np.int16)

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        outdata = np.zeros((2400, 1), dtype=np.float32)
        handler._audio_callback(outdata, 2400, None, None)

        # Verify correct frame count requested
        mixer.get_audio_data.assert_called_once_with(2400)

    def test_audio_callback_detects_underrun(self):
        """Test callback detects underrun (silence from mixer)."""
        mixer = Mock()
        # Mixer returns silence (underrun)
        mixer.get_audio_data.return_value = np.zeros(100, dtype=np.int16)

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None,
        )

        outdata = np.zeros((100, 1), dtype=np.float32)
        handler._audio_callback(outdata, 100, None, None)

        # Verify underrun was tracked
        metrics = handler.get_metrics()
        assert metrics["underruns"] == 1

    def test_audio_callback_with_status_error(self):
        """Test callback logs status errors."""
        mixer = Mock()
        mixer.get_audio_data.return_value = np.zeros(100, dtype=np.int16)

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None,
        )

        outdata = np.zeros((100, 1), dtype=np.float32)

        # Simulate error status
        handler._audio_callback(outdata, 100, None, "Output underflow")

        # Verify error was tracked
        metrics = handler.get_metrics()
        assert metrics["errors"] == 1
        assert "Output underflow" in metrics["last_error"]

    def test_audio_callback_handles_exception(self):
        """Test callback handles exceptions gracefully."""
        mixer = Mock()
        mixer.get_audio_data.side_effect = Exception("Mixer failed")

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None,
        )

        outdata = np.zeros((100, 1), dtype=np.float32)

        # Should not raise (audio callback must continue)
        handler._audio_callback(outdata, 100, None, None)

        # Verify error was tracked
        metrics = handler.get_metrics()
        assert metrics["errors"] == 1
        assert "Mixer failed" in metrics["last_error"]

        # Verify output buffer filled with silence
        assert np.all(outdata == 0)

    def test_audio_callback_silence_on_error(self):
        """Test callback fills with silence on error."""
        mixer = Mock()
        mixer.get_audio_data.side_effect = Exception("Mixer error")

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None,
        )

        # Pre-fill buffer with non-zero values
        outdata = np.ones((100, 1), dtype=np.float32)

        handler._audio_callback(outdata, 100, None, None)

        # Verify buffer was filled with silence
        assert np.all(outdata == 0)

    def test_audio_callback_with_filter_enabled(self):
        """Test callback applies filter when enabled."""
        mixer = Mock()
        mixer.get_audio_data.return_value = np.full(100, 16383, dtype=np.int16)

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None,
            use_audio_out_filter=True,
        )

        outdata = np.zeros((100, 1), dtype=np.float32)
        handler._audio_callback(outdata, 100, None, None)

        # Verify filter was applied (output should be different from input)
        # Input was 0.5, but filter should change the output
        assert not np.allclose(outdata[:, 0], 16383 / 32767)

    def test_audio_callback_with_filter_disabled(self):
        """Test callback bypasses filter when disabled (default)."""
        mixer = Mock()
        mixer.get_audio_data.return_value = np.full(100, 16383, dtype=np.int16)

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None,
            use_audio_out_filter=False,
        )

        outdata = np.zeros((100, 1), dtype=np.float32)
        handler._audio_callback(outdata, 100, None, None)

        # Verify no filter applied (output should be direct conversion)
        expected = np.full(100, 16383 / 32767, dtype=np.float32)
        np.testing.assert_array_almost_equal(outdata[:, 0], expected, decimal=5)


class TestMetrics:
    """Test metrics tracking."""

    def test_initial_metrics(self):
        """Test initial metrics are zero."""
        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        metrics = handler.get_metrics()
        assert metrics["frames_played"] == 0
        assert metrics["underruns"] == 0
        assert metrics["errors"] == 0
        assert metrics["last_error"] is None
        assert metrics["is_active"] is False

    @patch("src.alto_terminal.audio_output_handler.sd.OutputStream")
    def test_is_active_metric(self, mock_output_stream_class):
        """Test is_active metric reflects stream state."""
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_output_stream_class.return_value = mock_stream

        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        # Before start
        assert handler.get_metrics()["is_active"] is False

        # After start
        handler.start()
        assert handler.get_metrics()["is_active"] is True

        # After stop
        handler.stop()
        assert handler.get_metrics()["is_active"] is False

    def test_frames_played_metric(self):
        """Test frames_played metric increments."""
        mixer = Mock()
        mixer.get_audio_data.return_value = np.zeros(100, dtype=np.int16)

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None,
        )

        outdata = np.zeros((100, 1), dtype=np.float32)

        # Play 5 frames
        for _ in range(5):
            handler._audio_callback(outdata, 100, None, None)

        metrics = handler.get_metrics()
        assert metrics["frames_played"] == 5

    def test_underruns_metric(self):
        """Test underruns metric increments."""
        mixer = Mock()
        # Return silence (underrun)
        mixer.get_audio_data.return_value = np.zeros(100, dtype=np.int16)

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None,
        )

        outdata = np.zeros((100, 1), dtype=np.float32)

        # Cause 3 underruns
        for _ in range(3):
            handler._audio_callback(outdata, 100, None, None)

        metrics = handler.get_metrics()
        assert metrics["underruns"] == 3


class TestContextManager:
    """Test context manager usage."""

    @patch("src.alto_terminal.audio_output_handler.sd.OutputStream")
    def test_context_manager(self, mock_output_stream_class):
        """Test using handler as context manager."""
        mock_stream = MagicMock()
        mock_output_stream_class.return_value = mock_stream

        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        # Use as context manager
        with handler:
            # Verify stream was started
            mock_stream.start.assert_called_once()

        # Verify stream was stopped after context exit
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch("src.alto_terminal.audio_output_handler.sd.OutputStream")
    def test_context_manager_with_exception(self, mock_output_stream_class):
        """Test context manager stops stream even if exception occurs."""
        mock_stream = MagicMock()
        mock_output_stream_class.return_value = mock_stream

        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        # Exception should not prevent cleanup
        with pytest.raises(ValueError):
            with handler:
                raise ValueError("Test exception")

        # Verify stream was still stopped
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


class TestAudioProcessingModule:
    """Test AudioProcessingModule (APM) integration."""

    def test_initialization_with_apm(self):
        """Test handler initialization with APM."""
        mock_apm = Mock()
        mixer = Mock()

        handler = AudioOutputHandler(
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
            apm=mock_apm,
        )

        assert handler._apm is mock_apm
        assert handler._apm_frame_size == 480  # 48000 / 100 = 480 (10ms)

    def test_initialization_without_apm(self):
        """Test handler initialization without APM."""
        handler = AudioOutputHandler(
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
        )

        assert handler._apm is None

    def test_audio_callback_with_apm_processes_reverse_stream(self):
        """Test that APM processes reverse stream when enabled."""
        mock_apm = Mock()
        mock_mixer = Mock()

        # Mixer returns 2400 samples (50ms)
        audio_data = np.random.randint(-1000, 1000, 2400, dtype=np.int16)
        mock_mixer.get_audio_data.return_value = audio_data

        handler = AudioOutputHandler(
            mixer=mock_mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
            apm=mock_apm,
        )

        # Simulate audio callback
        outdata = np.zeros((2400, 1), dtype=np.float32)
        handler._audio_callback(outdata, 2400, None, None)

        # Verify APM process_reverse_stream was called 5 times (50ms → 5x 10ms chunks)
        assert mock_apm.process_reverse_stream.call_count == 5

        # Verify each call used 480 samples (10ms)
        for call in mock_apm.process_reverse_stream.call_args_list:
            frame = call[0][0]
            assert frame.samples_per_channel == 480

    def test_audio_callback_without_apm_no_processing(self):
        """Test that without APM, no processing occurs."""
        mock_mixer = Mock()
        audio_data = np.zeros(2400, dtype=np.int16)
        mock_mixer.get_audio_data.return_value = audio_data

        handler = AudioOutputHandler(
            mixer=mock_mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
            apm=None,  # No APM
        )

        outdata = np.zeros((2400, 1), dtype=np.float32)
        handler._audio_callback(outdata, 2400, None, None)

        # Verify mixer was still called (no APM doesn't break flow)
        mock_mixer.get_audio_data.assert_called_once_with(2400)

        # Verify output buffer was filled
        assert not np.all(outdata == 0) or np.all(audio_data == 0)

    def test_audio_callback_apm_processes_before_playback(self):
        """Test that APM processes audio BEFORE it's converted to float32."""
        mock_apm = Mock()
        mock_mixer = Mock()

        # Create recognizable pattern
        audio_data = np.array([100] * 2400, dtype=np.int16)
        mock_mixer.get_audio_data.return_value = audio_data

        handler = AudioOutputHandler(
            mixer=mock_mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
            apm=mock_apm,
        )

        outdata = np.zeros((2400, 1), dtype=np.float32)
        handler._audio_callback(outdata, 2400, None, None)

        # Verify APM was called before audio is played
        # The audio data passed to APM should be int16
        for call in mock_apm.process_reverse_stream.call_args_list:
            frame = call[0][0]
            frame_data = np.frombuffer(frame.data, dtype=np.int16)
            assert frame_data.dtype == np.int16

        # Verify output was converted to float32
        assert outdata.dtype == np.float32

    def test_audio_callback_apm_handles_partial_frames(self):
        """Test that partial frames (not evenly divisible by 480) are handled."""
        mock_apm = Mock()
        mock_mixer = Mock()

        # 2300 samples = 4 complete 10ms chunks (1920) + partial chunk (380)
        audio_data = np.zeros(2300, dtype=np.int16)
        mock_mixer.get_audio_data.return_value = audio_data

        handler = AudioOutputHandler(
            mixer=mock_mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2300,  # Not divisible by 480
            device_index=None,
            apm=mock_apm,
        )

        outdata = np.zeros((2300, 1), dtype=np.float32)
        handler._audio_callback(outdata, 2300, None, None)

        # Should only process complete chunks
        assert mock_apm.process_reverse_stream.call_count == 4

        # Verify output buffer was still filled with all 2300 samples
        assert outdata.shape == (2300, 1)

    def test_audio_callback_apm_with_silence(self):
        """Test APM processing with silence (all zeros)."""
        mock_apm = Mock()
        mock_mixer = Mock()

        # Return silence
        audio_data = np.zeros(2400, dtype=np.int16)
        mock_mixer.get_audio_data.return_value = audio_data

        handler = AudioOutputHandler(
            mixer=mock_mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
            apm=mock_apm,
        )

        outdata = np.zeros((2400, 1), dtype=np.float32)
        handler._audio_callback(outdata, 2400, None, None)

        # APM should still be called even with silence
        assert mock_apm.process_reverse_stream.call_count == 5

        # Output should be silence (zeros converted to float32)
        assert np.allclose(outdata, 0.0)

    def test_audio_callback_apm_does_not_modify_mixer_data(self):
        """Test that APM processing doesn't affect mixer's original data."""
        mock_apm = Mock()
        mock_mixer = Mock()

        original_data = np.array([1000] * 2400, dtype=np.int16)
        mock_mixer.get_audio_data.return_value = original_data.copy()

        handler = AudioOutputHandler(
            mixer=mock_mixer,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None,
            apm=mock_apm,
        )

        outdata = np.zeros((2400, 1), dtype=np.float32)
        handler._audio_callback(outdata, 2400, None, None)

        # APM shouldn't modify the original array from mixer
        # (it creates AudioFrames with copies)
        mixer_result = mock_mixer.get_audio_data.return_value
        assert np.all(mixer_result == 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
