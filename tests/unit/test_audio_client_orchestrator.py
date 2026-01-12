"""Unit tests for AudioClientOrchestrator.

These tests verify the core functionality of the AudioClientOrchestrator class:
- Initialization
- Component creation in correct order
- Connection lifecycle
- Disconnection and cleanup
- Error handling with partial cleanup
- Volume control
- Metrics aggregation
- Context manager usage
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch, call
from src.alto_terminal.audio_client_orchestrator import AudioClientOrchestrator
from src.alto_terminal.config import AudioConfig


class TestOrchestratorInitialization:
    """Test orchestrator initialization."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        config = AudioConfig(
            sample_rate=48000,
            num_channels=1,
            input_device=5,
            output_device=6,
            volume=0.8
        )

        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )

        assert orchestrator.url == "wss://livekit.example.com"
        assert orchestrator.token == "test-token"
        assert orchestrator.config is config
        assert orchestrator.mixer is None
        assert orchestrator.network_manager is None
        assert orchestrator.input_handler is None
        assert orchestrator.output_handler is None


class TestConnection:
    """Test connection lifecycle."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_connect(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test connecting creates all components in correct order."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        # Create orchestrator
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )

        # Connect
        await orchestrator.connect()

        # Verify components were created in order
        mock_mixer_class.assert_called_once_with(
            sample_rate=48000,
            channels=1,
            master_volume=1.0
        )

        mock_network_manager_class.assert_called_once_with(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mock_mixer,
            sample_rate=48000,
            num_channels=1,
            no_playback=False,
            on_transcription=None,
            on_audio_data=None
        )

        # Verify network manager methods called
        mock_network_manager.connect.assert_called_once()
        mock_network_manager.publish_audio_track.assert_called_once()

        # Verify input handler created
        mock_input_handler_class.assert_called_once()
        call_kwargs = mock_input_handler_class.call_args[1]
        assert call_kwargs['audio_source'] is mock_network_manager.audio_source
        assert call_kwargs['sample_rate'] == 48000

        # Verify output handler created
        mock_output_handler_class.assert_called_once()
        call_kwargs = mock_output_handler_class.call_args[1]
        assert call_kwargs['mixer'] is mock_mixer

        # Verify audio I/O started
        mock_input_handler.start.assert_called_once()
        mock_output_handler.start.assert_called_once()

        # Verify components stored
        assert orchestrator.mixer is mock_mixer
        assert orchestrator.network_manager is mock_network_manager
        assert orchestrator.input_handler is mock_input_handler
        assert orchestrator.output_handler is mock_output_handler

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_connect_no_playback(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test connecting with no_playback=True."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        # Create orchestrator with no_playback
        config = AudioConfig(sample_rate=48000, num_channels=1, no_playback=True)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )

        await orchestrator.connect()

        # Verify output handler was NOT created
        mock_output_handler_class.assert_not_called()
        assert orchestrator.output_handler is None

        # Verify input handler was still created
        mock_input_handler_class.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    async def test_connect_error_cleanup(
        self,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test error during connect triggers cleanup."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        # Make connect fail
        mock_network_manager.connect.side_effect = Exception("Connection failed")
        mock_network_manager_class.return_value = mock_network_manager

        # Create orchestrator
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )

        # Connect should fail
        with pytest.raises(Exception, match="Connection failed"):
            await orchestrator.connect()

        # Verify cleanup was called (network_manager.disconnect)
        mock_network_manager.disconnect.assert_called_once()

        # Verify components were cleared
        assert orchestrator.mixer is None
        assert orchestrator.network_manager is None


class TestDisconnection:
    """Test disconnection and cleanup."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_disconnect(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test disconnecting stops all components."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        # Create and connect
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )
        await orchestrator.connect()

        # Disconnect
        await orchestrator.disconnect()

        # Verify cleanup order (reverse of initialization)
        mock_output_handler.stop.assert_called_once()
        mock_input_handler.stop.assert_called_once()
        mock_network_manager.disconnect.assert_called()

        # Verify components cleared
        assert orchestrator.mixer is None
        assert orchestrator.network_manager is None
        assert orchestrator.input_handler is None
        assert orchestrator.output_handler is None

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    async def test_disconnect_handles_errors(
        self,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test disconnect continues cleanup even if components fail."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        # Make disconnect fail
        mock_network_manager.disconnect.side_effect = Exception("Disconnect failed")
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        # Make stop fail
        mock_input_handler.stop.side_effect = Exception("Stop failed")
        mock_input_handler_class.return_value = mock_input_handler

        # Create and connect
        config = AudioConfig(sample_rate=48000, num_channels=1, no_playback=True)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )
        await orchestrator.connect()

        # Disconnect should not raise (errors logged)
        await orchestrator.disconnect()

        # Verify components still cleared despite errors
        assert orchestrator.mixer is None
        assert orchestrator.network_manager is None
        assert orchestrator.input_handler is None


class TestVolumeControl:
    """Test volume control."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_set_master_volume(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test setting master volume."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        # Create and connect
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )
        await orchestrator.connect()

        # Set master volume
        orchestrator.set_master_volume(0.5)

        # Verify mixer master_volume was set
        assert mock_mixer.master_volume == 0.5

    def test_set_master_volume_not_connected(self):
        """Test setting master volume when not connected raises error."""
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )

        with pytest.raises(RuntimeError, match="Must connect"):
            orchestrator.set_master_volume(0.5)

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_set_stream_volume(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test setting stream volume."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        # Create and connect
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )
        await orchestrator.connect()

        # Set stream volume
        orchestrator.set_stream_volume("alice", 0.7)

        # Verify mixer method was called
        mock_mixer.set_stream_volume.assert_called_once_with("alice", 0.7)


class TestMetrics:
    """Test metrics aggregation."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_get_metrics(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test getting aggregated metrics."""
        # Setup mocks with metrics
        mock_mixer = Mock()
        mock_mixer.get_metrics.return_value = {'mixer_data': 'test'}
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        # Use Mock for get_metrics (synchronous method)
        mock_network_manager.get_metrics = Mock(return_value={
            'is_connected': True,
            'network_data': 'test'
        })
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler.get_metrics.return_value = {'input_data': 'test'}
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler.get_metrics.return_value = {'output_data': 'test'}
        mock_output_handler_class.return_value = mock_output_handler

        # Create and connect
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )
        await orchestrator.connect()

        # Get metrics
        metrics = orchestrator.get_metrics()

        # Verify aggregation
        assert metrics['is_connected'] is True
        assert metrics['mixer'] == {'mixer_data': 'test'}
        assert metrics['input'] == {'input_data': 'test'}
        assert metrics['output'] == {'output_data': 'test'}
        assert metrics['network'] == {'is_connected': True, 'network_data': 'test'}

    def test_get_metrics_not_connected(self):
        """Test getting metrics when not connected."""
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )

        metrics = orchestrator.get_metrics()

        # Should return empty metrics
        assert metrics['is_connected'] is False
        assert metrics['mixer'] == {}
        assert metrics['input'] == {}
        assert metrics['output'] is None
        assert metrics['network'] == {}


class TestContextManager:
    """Test context manager usage."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_context_manager(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test using orchestrator as context manager."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        # Create orchestrator
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )

        # Use as context manager
        async with orchestrator:
            # Verify connected
            mock_network_manager.connect.assert_called_once()

        # Verify disconnected after exit
        mock_network_manager.disconnect.assert_called()

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_context_manager_with_exception(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test context manager disconnects even if exception occurs."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        # Create orchestrator
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )

        # Exception should not prevent cleanup
        with pytest.raises(ValueError):
            async with orchestrator:
                raise ValueError("Test exception")

        # Verify disconnected despite exception
        mock_network_manager.disconnect.assert_called()


class TestAudioProcessingModule:
    """Test AudioProcessingModule (APM) creation and passing."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioProcessingModule')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_creates_apm_when_aec_enabled(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class,
        mock_apm_class
    ):
        """Test that APM is created when AEC is enabled."""
        # Setup mocks
        mock_apm = Mock()
        mock_apm_class.return_value = mock_apm

        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        # Create config with AEC enabled
        config = AudioConfig(enable_aec=True)

        orchestrator = AudioClientOrchestrator(
            url="wss://test.com",
            token="test-token",
            config=config
        )

        await orchestrator.connect()

        # Verify APM was created
        mock_apm_class.assert_called_once_with(
            echo_cancellation=True,
            noise_suppression=False,
            high_pass_filter=False,
            auto_gain_control=False
        )

        # Verify APM is stored
        assert orchestrator.apm is mock_apm

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioProcessingModule')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_creates_apm_with_all_features_enabled(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class,
        mock_apm_class
    ):
        """Test APM creation with all audio processing features."""
        mock_apm = Mock()
        mock_apm_class.return_value = mock_apm

        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        # Enable all audio processing features
        config = AudioConfig(
            enable_aec=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=True
        )

        orchestrator = AudioClientOrchestrator(
            url="wss://test.com",
            token="test-token",
            config=config
        )

        await orchestrator.connect()

        # Verify APM was created with all features
        mock_apm_class.assert_called_once_with(
            echo_cancellation=True,
            noise_suppression=True,
            high_pass_filter=True,
            auto_gain_control=True
        )

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_does_not_create_apm_when_disabled(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test that APM is NOT created when all features are disabled."""
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        # All audio processing disabled (default)
        config = AudioConfig()

        orchestrator = AudioClientOrchestrator(
            url="wss://test.com",
            token="test-token",
            config=config
        )

        await orchestrator.connect()

        # Verify APM is None
        assert orchestrator.apm is None

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioProcessingModule')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_passes_apm_to_input_handler(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class,
        mock_apm_class
    ):
        """Test that APM is passed to AudioInputHandler."""
        mock_apm = Mock()
        mock_apm_class.return_value = mock_apm

        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        config = AudioConfig(enable_aec=True)

        orchestrator = AudioClientOrchestrator(
            url="wss://test.com",
            token="test-token",
            config=config
        )

        await orchestrator.connect()

        # Verify APM was passed to input handler
        input_handler_call_kwargs = mock_input_handler_class.call_args[1]
        assert input_handler_call_kwargs['apm'] is mock_apm

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioProcessingModule')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_passes_apm_to_output_handler(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class,
        mock_apm_class
    ):
        """Test that APM is passed to AudioOutputHandler."""
        mock_apm = Mock()
        mock_apm_class.return_value = mock_apm

        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        config = AudioConfig(enable_aec=True)

        orchestrator = AudioClientOrchestrator(
            url="wss://test.com",
            token="test-token",
            config=config
        )

        await orchestrator.connect()

        # Verify APM was passed to output handler
        output_handler_call_kwargs = mock_output_handler_class.call_args[1]
        assert output_handler_call_kwargs['apm'] is mock_apm

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioProcessingModule')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_same_apm_instance_shared_between_handlers(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class,
        mock_apm_class
    ):
        """Test that the SAME APM instance is shared between input and output handlers."""
        mock_apm = Mock()
        mock_apm_class.return_value = mock_apm

        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        config = AudioConfig(enable_aec=True)

        orchestrator = AudioClientOrchestrator(
            url="wss://test.com",
            token="test-token",
            config=config
        )

        await orchestrator.connect()

        # Get APM passed to both handlers
        input_apm = mock_input_handler_class.call_args[1]['apm']
        output_apm = mock_output_handler_class.call_args[1]['apm']

        # Verify they are the SAME instance (critical for AEC to work)
        assert input_apm is output_apm
        assert input_apm is mock_apm

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioProcessingModule')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_creates_apm_with_noise_suppression_only(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class,
        mock_apm_class
    ):
        """Test APM creation with only noise suppression enabled."""
        mock_apm = Mock()
        mock_apm_class.return_value = mock_apm

        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler_class.return_value = mock_output_handler

        config = AudioConfig(noise_suppression=True)

        orchestrator = AudioClientOrchestrator(
            url="wss://test.com",
            token="test-token",
            config=config
        )

        await orchestrator.connect()

        # Verify APM was created with only noise suppression
        mock_apm_class.assert_called_once_with(
            echo_cancellation=False,
            noise_suppression=True,
            high_pass_filter=False,
            auto_gain_control=False
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
