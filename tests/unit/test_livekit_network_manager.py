"""Unit tests for LiveKitNetworkManager.

These tests verify the core functionality of the LiveKitNetworkManager class:
- Room connection/disconnection
- Track publishing
- Event handlers (participant connected/disconnected, track subscribed/unsubscribed)
- Remote audio routing to mixer
- Metrics tracking
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, MagicMock, AsyncMock, patch, call
from src.alto_terminal.livekit_network_manager import LiveKitNetworkManager, NetworkMetrics


class TestLiveKitNetworkManagerInitialization:
    """Test LiveKitNetworkManager initialization."""

    def test_initialization(self):
        """Test manager initialization with all parameters."""
        mixer = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            no_playback=False
        )

        assert manager.url == "wss://livekit.example.com"
        assert manager.token == "test-token"
        assert manager.mixer is mixer
        assert manager.sample_rate == 48000
        assert manager.num_channels == 1
        assert manager.no_playback is False
        assert manager.room is None
        assert manager.audio_source is None

    def test_initialization_with_callbacks(self):
        """Test initialization with event callbacks."""
        on_participant_connected = Mock()
        on_participant_disconnected = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1,
            on_participant_connected=on_participant_connected,
            on_participant_disconnected=on_participant_disconnected
        )

        assert manager._on_participant_connected_cb is on_participant_connected
        assert manager._on_participant_disconnected_cb is on_participant_disconnected


class TestRoomConnection:
    """Test room connection/disconnection."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.livekit_network_manager.Room')
    async def test_connect(self, mock_room_class):
        """Test connecting to room."""
        mock_room = AsyncMock()
        mock_room.name = "test-room"
        mock_room_class.return_value = mock_room

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1
        )

        room = await manager.connect()

        # Verify Room was created
        mock_room_class.assert_called_once()

        # Verify event handlers were registered
        assert mock_room.on.call_count == 5
        mock_room.on.assert_any_call("participant_connected", manager._on_participant_connected)
        mock_room.on.assert_any_call("participant_disconnected", manager._on_participant_disconnected)
        mock_room.on.assert_any_call("track_subscribed", manager._on_track_subscribed)
        mock_room.on.assert_any_call("track_unsubscribed", manager._on_track_unsubscribed)
        mock_room.on.assert_any_call("data_received", manager._on_data_received)

        # Verify connection was called
        mock_room.connect.assert_called_once()
        connect_args = mock_room.connect.call_args[0]
        assert connect_args[0] == "wss://livekit.example.com"
        assert connect_args[1] == "test-token"

        # Verify room was returned
        assert room is mock_room
        assert manager.room is mock_room

        # Verify metrics
        metrics = manager.get_metrics()
        assert metrics['is_connected'] is True
        assert metrics['room_name'] == "test-room"

    @pytest.mark.asyncio
    @patch('src.alto_terminal.livekit_network_manager.Room')
    async def test_connect_already_connected(self, mock_room_class):
        """Test connecting when already connected."""
        mock_room = AsyncMock()
        mock_room.name = "test-room"
        mock_room_class.return_value = mock_room

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1
        )

        await manager.connect()
        await manager.connect()  # Connect again

        # Should only create room once
        assert mock_room_class.call_count == 1

    @pytest.mark.asyncio
    @patch('src.alto_terminal.livekit_network_manager.Room')
    async def test_connect_error(self, mock_room_class):
        """Test error during connection."""
        mock_room = AsyncMock()
        mock_room.connect.side_effect = Exception("Connection failed")
        mock_room_class.return_value = mock_room

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1
        )

        with pytest.raises(Exception, match="Connection failed"):
            await manager.connect()

        # Verify room was cleared
        assert manager.room is None

        # Verify metrics
        metrics = manager.get_metrics()
        assert metrics['is_connected'] is False

    @pytest.mark.asyncio
    @patch('src.alto_terminal.livekit_network_manager.Room')
    async def test_disconnect(self, mock_room_class):
        """Test disconnecting from room."""
        mock_room = AsyncMock()
        mock_room.name = "test-room"
        mock_room_class.return_value = mock_room

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1
        )

        await manager.connect()
        await manager.disconnect()

        # Verify room was disconnected
        mock_room.disconnect.assert_called_once()

        # Verify room was cleared
        assert manager.room is None
        assert manager.audio_source is None

        # Verify metrics
        metrics = manager.get_metrics()
        assert metrics['is_connected'] is False
        assert metrics['room_name'] is None

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self):
        """Test disconnecting when not connected."""
        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1
        )

        # Should not raise error
        await manager.disconnect()


class TestTrackPublishing:
    """Test track publishing."""

    @pytest.mark.asyncio
    @patch('src.alto_terminal.livekit_network_manager.Room')
    @patch('src.alto_terminal.livekit_network_manager.rtc.AudioSource')
    @patch('src.alto_terminal.livekit_network_manager.rtc.LocalAudioTrack')
    async def test_publish_audio_track(self, mock_track_class, mock_audio_source_class, mock_room_class):
        """Test publishing audio track."""
        mock_room = AsyncMock()
        mock_room.name = "test-room"
        mock_room.local_participant = AsyncMock()
        mock_room_class.return_value = mock_room

        mock_audio_source = Mock()
        mock_audio_source_class.return_value = mock_audio_source

        mock_track = Mock()
        mock_track_class.create_audio_track.return_value = mock_track

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1
        )

        await manager.connect()
        track = await manager.publish_audio_track()

        # Verify AudioSource was created
        mock_audio_source_class.assert_called_once_with(48000, 1, queue_size_ms=200)

        # Verify LocalAudioTrack was created
        mock_track_class.create_audio_track.assert_called_once_with(
            "microphone",
            mock_audio_source
        )

        # Verify track was published
        mock_room.local_participant.publish_track.assert_called_once()

        # Verify audio_source is stored
        assert manager.audio_source is mock_audio_source

        # Verify track was returned
        assert track is mock_track

    @pytest.mark.asyncio
    async def test_publish_audio_track_not_connected(self):
        """Test publishing track when not connected."""
        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1
        )

        with pytest.raises(RuntimeError, match="Must connect to room"):
            await manager.publish_audio_track()


class TestEventHandlers:
    """Test event handlers."""

    def test_on_participant_connected(self):
        """Test participant connected event handler."""
        mixer = Mock()
        on_participant_connected_cb = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            on_participant_connected=on_participant_connected_cb
        )

        # Simulate participant connected
        participant = Mock()
        participant.identity = "alice"
        participant.name = "Alice"

        manager._on_participant_connected(participant)

        # Verify callback was called
        on_participant_connected_cb.assert_called_once_with(participant)

        # Verify metrics
        metrics = manager.get_metrics()
        assert metrics['participant_count'] == 1

    def test_on_participant_disconnected(self):
        """Test participant disconnected event handler."""
        mixer = Mock()
        on_participant_disconnected_cb = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            on_participant_disconnected=on_participant_disconnected_cb
        )

        # Set initial participant count
        manager._metrics.participant_count = 2

        # Simulate participant disconnected
        participant = Mock()
        participant.identity = "alice"

        manager._on_participant_disconnected(participant)

        # Verify mixer stream was removed
        mixer.remove_stream.assert_called_once_with("alice")

        # Verify callback was called
        on_participant_disconnected_cb.assert_called_once_with(participant)

        # Verify metrics
        metrics = manager.get_metrics()
        assert metrics['participant_count'] == 1

    @pytest.mark.asyncio
    async def test_on_track_subscribed_audio(self):
        """Test track subscribed event handler for audio track."""
        mixer = Mock()
        on_track_subscribed_cb = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            on_track_subscribed=on_track_subscribed_cb
        )

        # Create mock audio track - use MagicMock and patch isinstance check
        track = MagicMock()
        track.name = "audio"
        track.kind = "audio"

        publication = Mock()
        participant = Mock()
        participant.identity = "alice"

        # Mock isinstance to return True for RemoteAudioTrack check
        with patch('src.alto_terminal.livekit_network_manager.isinstance') as mock_isinstance:
            # Make isinstance return True for RemoteAudioTrack check
            mock_isinstance.return_value = True

            # Mock asyncio.create_task to avoid actual task creation
            with patch('asyncio.create_task') as mock_create_task:
                mock_task = Mock()
                mock_create_task.return_value = mock_task

                manager._on_track_subscribed(track, publication, participant)

                # Verify mixer stream was added
                mixer.add_stream.assert_called_once_with("alice", volume=1.0)

                # Verify async task was created
                mock_create_task.assert_called_once()

                # Verify callback was called
                on_track_subscribed_cb.assert_called_once_with(track, publication, participant)

                # Verify metrics
                metrics = manager.get_metrics()
                assert metrics['tracks_subscribed'] == 1

    def test_on_track_subscribed_non_audio(self):
        """Test track subscribed for non-audio track."""
        mixer = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mixer,
            sample_rate=48000,
            num_channels=1
        )

        # Create mock video track
        track = Mock()
        track.name = "video"
        track.kind = "video"

        publication = Mock()
        participant = Mock()
        participant.identity = "alice"

        manager._on_track_subscribed(track, publication, participant)

        # Verify mixer stream was NOT added (not audio)
        mixer.add_stream.assert_not_called()

    def test_on_track_unsubscribed(self):
        """Test track unsubscribed event handler."""
        mixer = Mock()
        on_track_unsubscribed_cb = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            on_track_unsubscribed=on_track_unsubscribed_cb
        )

        # Set initial metrics
        manager._metrics.tracks_subscribed = 2

        # Add a mock task
        mock_task = Mock()
        manager._remote_audio_tasks["alice"] = mock_task

        # Simulate track unsubscribed
        track = Mock()
        track.name = "audio"
        track.kind = "audio"

        publication = Mock()
        participant = Mock()
        participant.identity = "alice"

        manager._on_track_unsubscribed(track, publication, participant)

        # Verify mixer stream was removed
        mixer.remove_stream.assert_called_once_with("alice")

        # Verify task was cancelled
        mock_task.cancel.assert_called_once()

        # Verify callback was called
        on_track_unsubscribed_cb.assert_called_once_with(track, publication, participant)

        # Verify metrics
        metrics = manager.get_metrics()
        assert metrics['tracks_subscribed'] == 1


class TestRemoteAudioProcessing:
    """Test remote audio processing."""

    @pytest.mark.asyncio
    async def test_handle_remote_audio_track(self):
        """Test processing remote audio frames."""
        mixer = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            no_playback=False
        )

        # Create mock track and stream
        track = Mock()

        # Create mock events with audio frames
        frame1 = Mock()
        frame1.data = np.ones(1000, dtype=np.int16).tobytes()
        frame1.num_channels = 1
        frame1.sample_rate = 48000

        frame2 = Mock()
        frame2.data = np.ones(1000, dtype=np.int16).tobytes()
        frame2.num_channels = 1
        frame2.sample_rate = 48000

        event1 = Mock()
        event1.frame = frame1

        event2 = Mock()
        event2.frame = frame2

        # Create async iterator
        async def mock_stream_iter():
            yield event1
            yield event2

        mock_stream = Mock()
        mock_stream.__aiter__ = lambda self: mock_stream_iter()

        # Patch AudioStream
        with patch('src.alto_terminal.livekit_network_manager.rtc.AudioStream') as mock_audio_stream:
            mock_audio_stream.return_value = mock_stream

            # Process audio
            await manager._handle_remote_audio_track(track, "alice")

            # Verify AudioStream was created
            mock_audio_stream.assert_called_once_with(
                track,
                sample_rate=48000,
                num_channels=1
            )

            # Verify frames were added to mixer (2 frames)
            assert mixer.add_audio_data.call_count == 2

            # Verify participant identity was used
            for call in mixer.add_audio_data.call_args_list:
                assert call[1]['stream_id'] == "alice"

            # Verify metrics
            metrics = manager.get_metrics()
            assert metrics['frames_received'] == 2

    @pytest.mark.asyncio
    async def test_handle_remote_audio_track_no_playback(self):
        """Test remote audio with no_playback=True."""
        mixer = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mixer,
            sample_rate=48000,
            num_channels=1,
            no_playback=True  # Don't route to mixer
        )

        # Create mock track and stream
        track = Mock()

        frame = Mock()
        frame.data = np.ones(1000, dtype=np.int16).tobytes()
        frame.num_channels = 1
        frame.sample_rate = 48000

        event = Mock()
        event.frame = frame

        async def mock_stream_iter():
            yield event

        mock_stream = Mock()
        mock_stream.__aiter__ = lambda self: mock_stream_iter()

        with patch('src.alto_terminal.livekit_network_manager.rtc.AudioStream') as mock_audio_stream:
            mock_audio_stream.return_value = mock_stream

            await manager._handle_remote_audio_track(track, "alice")

            # Verify frames were NOT added to mixer
            mixer.add_audio_data.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_remote_audio_track_cancellation(self):
        """Test cancelling remote audio task."""
        mixer = Mock()

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=mixer,
            sample_rate=48000,
            num_channels=1
        )

        # Create mock track that raises CancelledError
        track = Mock()

        # Create proper async iterator
        class MockAsyncIterator:
            async def __anext__(self):
                await asyncio.sleep(0)
                raise asyncio.CancelledError()

            def __aiter__(self):
                return self

        mock_stream = MockAsyncIterator()

        with patch('src.alto_terminal.livekit_network_manager.rtc.AudioStream') as mock_audio_stream:
            mock_audio_stream.return_value = mock_stream

            # Should raise CancelledError
            with pytest.raises(asyncio.CancelledError):
                await manager._handle_remote_audio_track(track, "alice")


class TestMetrics:
    """Test metrics tracking."""

    def test_initial_metrics(self):
        """Test initial metrics are zero."""
        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1
        )

        metrics = manager.get_metrics()
        assert metrics['is_connected'] is False
        assert metrics['room_name'] is None
        assert metrics['participant_count'] == 0
        assert metrics['tracks_subscribed'] == 0
        assert metrics['frames_received'] == 0
        assert metrics['active_audio_tasks'] == 0

    @pytest.mark.asyncio
    @patch('src.alto_terminal.livekit_network_manager.Room')
    async def test_metrics_after_connect(self, mock_room_class):
        """Test metrics after connecting."""
        mock_room = AsyncMock()
        mock_room.name = "test-room"
        mock_room_class.return_value = mock_room

        manager = LiveKitNetworkManager(
            url="wss://livekit.example.com",
            token="test-token",
            mixer=Mock(),
            sample_rate=48000,
            num_channels=1
        )

        await manager.connect()

        metrics = manager.get_metrics()
        assert metrics['is_connected'] is True
        assert metrics['room_name'] == "test-room"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
