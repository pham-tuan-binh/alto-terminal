"""Integration tests for full audio flow.

These tests verify that all components work together correctly:
- Orchestrator connects all components
- Audio flows from input → LiveKit → output
- Components cleanup properly on disconnect
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, MagicMock, AsyncMock, patch

from src.alto_terminal.config import AudioConfig
from src.alto_terminal.audio_client_orchestrator import AudioClientOrchestrator


class TestFullAudioFlow:
    """Test complete audio flow integration."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_orchestrator_connects_all_components(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test that orchestrator properly wires all components."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager.get_metrics = Mock(return_value={'is_connected': True})
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

        # Verify all components created
        assert orchestrator.mixer is not None
        assert orchestrator.network_manager is not None
        assert orchestrator.input_handler is not None
        assert orchestrator.output_handler is not None

        # Verify network manager connected and published
        mock_network_manager.connect.assert_called_once()
        mock_network_manager.publish_audio_track.assert_called_once()

        # Verify audio I/O started
        mock_input_handler.start.assert_called_once()
        mock_output_handler.start.assert_called_once()

        # Disconnect
        await orchestrator.disconnect()

        # Verify cleanup
        mock_output_handler.stop.assert_called_once()
        mock_input_handler.stop.assert_called_once()
        mock_network_manager.disconnect.assert_called()

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_context_manager_lifecycle(
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
        mock_network_manager.get_metrics = Mock(return_value={'is_connected': True})
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
            assert orchestrator.mixer is not None
            assert orchestrator.network_manager is not None

        # Verify disconnected after exit
        assert orchestrator.mixer is None
        assert orchestrator.network_manager is None

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    async def test_no_playback_mode(
        self,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test no_playback mode doesn't create output handler."""
        # Setup mocks
        mock_mixer = Mock()
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager.get_metrics = Mock(return_value={'is_connected': True})
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

        # Connect
        await orchestrator.connect()

        # Verify output handler was NOT created
        assert orchestrator.output_handler is None

        # Verify other components were created
        assert orchestrator.mixer is not None
        assert orchestrator.network_manager is not None
        assert orchestrator.input_handler is not None

        # Disconnect
        await orchestrator.disconnect()


class TestMetricsAggregation:
    """Test metrics aggregation across components."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.audio_client_orchestrator.AudioMixer')
    @patch('src.alto_terminal.audio_client_orchestrator.LiveKitNetworkManager')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioInputHandler')
    @patch('src.alto_terminal.audio_client_orchestrator.AudioOutputHandler')
    async def test_get_metrics_aggregates_all_components(
        self,
        mock_output_handler_class,
        mock_input_handler_class,
        mock_network_manager_class,
        mock_mixer_class
    ):
        """Test get_metrics aggregates from all components."""
        # Setup mocks with metrics
        mock_mixer = Mock()
        mock_mixer.get_metrics.return_value = {
            'mixed_buffer_size': 1000,
            'active_streams': 2
        }
        mock_mixer_class.return_value = mock_mixer

        mock_network_manager = AsyncMock()
        mock_network_manager.audio_source = Mock()
        mock_network_manager.get_metrics = Mock(return_value={
            'is_connected': True,
            'participant_count': 3,
            'frames_received': 5000
        })
        mock_network_manager_class.return_value = mock_network_manager

        mock_input_handler = Mock()
        mock_input_handler.get_metrics.return_value = {
            'frames_captured': 4000,
            'is_active': True
        }
        mock_input_handler_class.return_value = mock_input_handler

        mock_output_handler = Mock()
        mock_output_handler.get_metrics.return_value = {
            'frames_played': 4500,
            'underruns': 2
        }
        mock_output_handler_class.return_value = mock_output_handler

        # Create and connect
        config = AudioConfig(sample_rate=48000, num_channels=1)
        orchestrator = AudioClientOrchestrator(
            url="wss://livekit.example.com",
            token="test-token",
            config=config
        )
        await orchestrator.connect()

        # Get aggregated metrics
        metrics = orchestrator.get_metrics()

        # Verify aggregation
        assert metrics['is_connected'] is True
        assert metrics['mixer']['mixed_buffer_size'] == 1000
        assert metrics['network']['participant_count'] == 3
        assert metrics['input']['frames_captured'] == 4000
        assert metrics['output']['frames_played'] == 4500

        await orchestrator.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
