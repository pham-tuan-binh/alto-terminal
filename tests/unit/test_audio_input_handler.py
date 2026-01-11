"""Unit tests for AudioInputHandler.

These tests verify the core functionality of the AudioInputHandler class:
- Initialization
- Start/stop lifecycle
- Audio callback (float32 → int16 conversion)
- Error handling
- Metrics tracking
- Context manager usage
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, MagicMock, patch, call
from src.alto_terminal.audio_input_handler import AudioInputHandler, InputMetrics


class TestAudioInputHandlerInitialization:
    """Test AudioInputHandler initialization."""

    def test_initialization(self):
        """Test handler initialization with all parameters."""
        audio_source = Mock()
        event_loop = Mock()

        handler = AudioInputHandler(
            audio_source=audio_source,
            event_loop=event_loop,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=5
        )

        assert handler.audio_source is audio_source
        assert handler.event_loop is event_loop
        assert handler.sample_rate == 48000
        assert handler.num_channels == 1
        assert handler.samples_per_channel == 2400
        assert handler.device_index == 5
        assert handler._input_stream is None

    def test_initialization_default_device(self):
        """Test initialization with default device (None)."""
        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        assert handler.device_index is None


class TestAudioInputLifecycle:
    """Test start/stop lifecycle."""

    @patch('src.alto_terminal.audio_input_handler.sd.InputStream')
    def test_start(self, mock_input_stream_class):
        """Test starting audio input."""
        mock_stream = MagicMock()
        mock_input_stream_class.return_value = mock_stream

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        handler.start()

        # Verify InputStream was created with correct parameters
        mock_input_stream_class.assert_called_once_with(
            device=None,
            channels=1,
            samplerate=48000,
            blocksize=2400,
            dtype='float32',
            callback=handler._audio_callback
        )

        # Verify stream was started
        mock_stream.start.assert_called_once()

        # Verify stream is stored
        assert handler._input_stream is mock_stream

    @patch('src.alto_terminal.audio_input_handler.sd.InputStream')
    def test_start_with_device_index(self, mock_input_stream_class):
        """Test starting with specific device index."""
        mock_stream = MagicMock()
        mock_input_stream_class.return_value = mock_stream

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=5
        )

        handler.start()

        # Verify device parameter was passed
        call_kwargs = mock_input_stream_class.call_args[1]
        assert call_kwargs['device'] == 5

    @patch('src.alto_terminal.audio_input_handler.sd.InputStream')
    def test_start_already_started(self, mock_input_stream_class):
        """Test starting when already started (should warn)."""
        mock_stream = MagicMock()
        mock_input_stream_class.return_value = mock_stream

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        handler.start()
        handler.start()  # Start again

        # Should only create stream once
        assert mock_input_stream_class.call_count == 1

    @patch('src.alto_terminal.audio_input_handler.sd.InputStream')
    def test_start_error(self, mock_input_stream_class):
        """Test error during start."""
        mock_input_stream_class.side_effect = Exception("Device not found")

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        with pytest.raises(Exception, match="Device not found"):
            handler.start()

        # Verify metrics tracked error
        metrics = handler.get_metrics()
        assert metrics['errors'] == 1
        assert "Device not found" in metrics['last_error']

    @patch('src.alto_terminal.audio_input_handler.sd.InputStream')
    def test_stop(self, mock_input_stream_class):
        """Test stopping audio input."""
        mock_stream = MagicMock()
        mock_input_stream_class.return_value = mock_stream

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        handler.start()
        handler.stop()

        # Verify stream was stopped and closed
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

        # Verify stream reference cleared
        assert handler._input_stream is None

    def test_stop_not_started(self):
        """Test stopping when not started (should warn)."""
        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        # Should not raise error
        handler.stop()

    @patch('src.alto_terminal.audio_input_handler.sd.InputStream')
    def test_stop_error(self, mock_input_stream_class):
        """Test error during stop."""
        mock_stream = MagicMock()
        mock_stream.stop.side_effect = Exception("Stop failed")
        mock_input_stream_class.return_value = mock_stream

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        handler.start()

        with pytest.raises(Exception, match="Stop failed"):
            handler.stop()

        # Verify metrics tracked error
        metrics = handler.get_metrics()
        assert metrics['errors'] == 1
        assert "Stop failed" in metrics['last_error']


class TestAudioCallback:
    """Test audio callback functionality."""

    @patch('src.alto_terminal.audio_input_handler.asyncio.run_coroutine_threadsafe')
    @patch('src.alto_terminal.audio_input_handler.rtc.AudioFrame')
    def test_audio_callback_conversion(self, mock_audio_frame, mock_run_coro):
        """Test float32 to int16 conversion in callback."""
        audio_source = Mock()
        event_loop = Mock()

        handler = AudioInputHandler(
            audio_source=audio_source,
            event_loop=event_loop,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None
        )

        # Simulate audio input: 100 samples at 0.5 amplitude
        indata = np.full((100, 1), 0.5, dtype=np.float32)

        # Call the callback
        handler._audio_callback(indata, 100, None, None)

        # Verify AudioFrame was created
        mock_audio_frame.assert_called_once()
        call_kwargs = mock_audio_frame.call_args[1]

        # Verify parameters
        assert call_kwargs['sample_rate'] == 48000
        assert call_kwargs['num_channels'] == 1
        assert call_kwargs['samples_per_channel'] == 100

        # Verify conversion: 0.5 * 32767 = 16383.5 → 16383
        audio_bytes = call_kwargs['data']
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        assert len(audio_data) == 100
        assert np.all(audio_data == 16383)

    @patch('src.alto_terminal.audio_input_handler.asyncio.run_coroutine_threadsafe')
    @patch('src.alto_terminal.audio_input_handler.rtc.AudioFrame')
    def test_audio_callback_full_range(self, mock_audio_frame, mock_run_coro):
        """Test conversion across full float32 range."""
        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=5,
            device_index=None
        )

        # Test samples: -1.0, -0.5, 0.0, 0.5, 1.0
        indata = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]], dtype=np.float32)

        handler._audio_callback(indata, 5, None, None)

        # Extract converted data
        call_kwargs = mock_audio_frame.call_args[1]
        audio_data = np.frombuffer(call_kwargs['data'], dtype=np.int16)

        # Verify conversion
        # -1.0 * 32767 = -32767
        # -0.5 * 32767 = -16383.5 → -16383
        # 0.0 * 32767 = 0
        # 0.5 * 32767 = 16383.5 → 16383
        # 1.0 * 32767 = 32767
        expected = np.array([-32767, -16383, 0, 16383, 32767], dtype=np.int16)
        np.testing.assert_array_equal(audio_data, expected)

    @patch('src.alto_terminal.audio_input_handler.asyncio.run_coroutine_threadsafe')
    @patch('src.alto_terminal.audio_input_handler.rtc.AudioFrame')
    def test_audio_callback_bridges_to_event_loop(self, mock_audio_frame, mock_run_coro):
        """Test callback bridges to async event loop."""
        audio_source = Mock()
        event_loop = Mock()
        mock_frame = Mock()
        mock_audio_frame.return_value = mock_frame

        handler = AudioInputHandler(
            audio_source=audio_source,
            event_loop=event_loop,
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None
        )

        indata = np.zeros((100, 1), dtype=np.float32)
        handler._audio_callback(indata, 100, None, None)

        # Verify run_coroutine_threadsafe was called
        mock_run_coro.assert_called_once()

        # Verify it was called with audio_source.capture_frame and event_loop
        call_args = mock_run_coro.call_args[0]
        assert call_args[1] is event_loop

    def test_audio_callback_with_status_error(self):
        """Test callback logs status errors."""
        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None
        )

        indata = np.zeros((100, 1), dtype=np.float32)

        # Simulate error status
        with patch('src.alto_terminal.audio_input_handler.asyncio.run_coroutine_threadsafe'), \
             patch('src.alto_terminal.audio_input_handler.rtc.AudioFrame'):
            handler._audio_callback(indata, 100, None, "Input overflow")

        # Verify error was tracked
        metrics = handler.get_metrics()
        assert metrics['errors'] == 1
        assert "Input overflow" in metrics['last_error']

    @patch('src.alto_terminal.audio_input_handler.rtc.AudioFrame')
    def test_audio_callback_handles_exception(self, mock_audio_frame):
        """Test callback handles exceptions gracefully."""
        mock_audio_frame.side_effect = Exception("Frame creation failed")

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None
        )

        indata = np.zeros((100, 1), dtype=np.float32)

        # Should not raise (audio callback must continue)
        handler._audio_callback(indata, 100, None, None)

        # Verify error was tracked
        metrics = handler.get_metrics()
        assert metrics['errors'] == 1
        assert "Frame creation failed" in metrics['last_error']


class TestMetrics:
    """Test metrics tracking."""

    def test_initial_metrics(self):
        """Test initial metrics are zero."""
        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        metrics = handler.get_metrics()
        assert metrics['frames_captured'] == 0
        assert metrics['errors'] == 0
        assert metrics['last_error'] is None
        assert metrics['is_active'] is False

    @patch('src.alto_terminal.audio_input_handler.sd.InputStream')
    def test_is_active_metric(self, mock_input_stream_class):
        """Test is_active metric reflects stream state."""
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_input_stream_class.return_value = mock_stream

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        # Before start
        assert handler.get_metrics()['is_active'] is False

        # After start
        handler.start()
        assert handler.get_metrics()['is_active'] is True

        # After stop
        handler.stop()
        assert handler.get_metrics()['is_active'] is False

    @patch('src.alto_terminal.audio_input_handler.asyncio.run_coroutine_threadsafe')
    @patch('src.alto_terminal.audio_input_handler.rtc.AudioFrame')
    def test_frames_captured_metric(self, mock_audio_frame, mock_run_coro):
        """Test frames_captured metric increments."""
        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=100,
            device_index=None
        )

        indata = np.zeros((100, 1), dtype=np.float32)

        # Capture 5 frames
        for _ in range(5):
            handler._audio_callback(indata, 100, None, None)

        metrics = handler.get_metrics()
        assert metrics['frames_captured'] == 5


class TestContextManager:
    """Test context manager usage."""

    @patch('src.alto_terminal.audio_input_handler.sd.InputStream')
    def test_context_manager(self, mock_input_stream_class):
        """Test using handler as context manager."""
        mock_stream = MagicMock()
        mock_input_stream_class.return_value = mock_stream

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        # Use as context manager
        with handler:
            # Verify stream was started
            mock_stream.start.assert_called_once()

        # Verify stream was stopped after context exit
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch('src.alto_terminal.audio_input_handler.sd.InputStream')
    def test_context_manager_with_exception(self, mock_input_stream_class):
        """Test context manager stops stream even if exception occurs."""
        mock_stream = MagicMock()
        mock_input_stream_class.return_value = mock_stream

        handler = AudioInputHandler(
            audio_source=Mock(),
            event_loop=Mock(),
            sample_rate=48000,
            num_channels=1,
            samples_per_channel=2400,
            device_index=None
        )

        # Exception should not prevent cleanup
        with pytest.raises(ValueError):
            with handler:
                raise ValueError("Test exception")

        # Verify stream was still stopped
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
