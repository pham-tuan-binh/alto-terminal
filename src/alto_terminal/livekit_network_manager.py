"""LiveKit network manager for room and track management.

This module provides the LiveKitNetworkManager class for managing
LiveKit room connections, track publishing, and remote audio processing.

Key design decisions:
- Handles all LiveKit SDK interactions
- Routes remote audio to AudioMixer
- Provides event callbacks for orchestrator
- Manages AudioSource lifecycle
- All operations are async (LiveKit SDK requirement)
"""

import asyncio
import logging
import numpy as np
from livekit import rtc
from livekit.rtc import Room, RoomOptions, TrackPublishOptions, TrackSource
from dataclasses import dataclass
from typing import Optional, Callable, Dict

logger = logging.getLogger(__name__)

# Store active text stream tasks to prevent garbage collection
_active_text_tasks = set()


@dataclass
class NetworkMetrics:
    """Metrics for network manager.

    Attributes:
        is_connected: Whether connected to room
        room_name: Name of the room (if connected)
        participant_count: Number of remote participants
        tracks_subscribed: Number of remote tracks subscribed
        frames_received: Total frames received from remote participants
    """

    is_connected: bool = False
    room_name: Optional[str] = None
    participant_count: int = 0
    tracks_subscribed: int = 0
    frames_received: int = 0


class LiveKitNetworkManager:
    """Manage LiveKit room connection and audio tracks.

    This manager handles all interactions with the LiveKit SDK:
    - Connecting/disconnecting from rooms
    - Publishing local audio track
    - Subscribing to remote audio tracks
    - Routing remote audio to mixer
    - Event notifications

    Thread model:
    - All methods are async (LiveKit SDK requirement)
    - Remote audio processing runs in async tasks
    - Mixer calls are synchronous and thread-safe

    Example:
        >>> mixer = AudioMixer(48000, 1)
        >>> manager = LiveKitNetworkManager(
        ...     url="wss://livekit.example.com",
        ...     token="...",
        ...     mixer=mixer,
        ...     sample_rate=48000,
        ...     num_channels=1
        ... )
        >>> await manager.connect()
        >>> await manager.publish_audio_track()
        >>> # ... use manager.audio_source for input handler ...
        >>> await manager.disconnect()
    """

    def __init__(
        self,
        url: str,
        token: str,
        mixer,  # AudioMixer instance (avoid circular import)
        sample_rate: int,
        num_channels: int,
        no_playback: bool = False,
        on_participant_connected: Optional[Callable] = None,
        on_participant_disconnected: Optional[Callable] = None,
        on_track_subscribed: Optional[Callable] = None,
        on_track_unsubscribed: Optional[Callable] = None,
        on_transcription: Optional[Callable] = None,
        on_audio_data: Optional[Callable] = None,
    ):
        """Initialize LiveKit network manager.

        Args:
            url: LiveKit server URL (e.g., "wss://livekit.example.com")
            token: Authentication token
            mixer: AudioMixer instance to route remote audio to
            sample_rate: Sample rate in Hz (e.g., 48000)
            num_channels: Number of channels (typically 1 for mono)
            no_playback: If True, don't route remote audio to mixer
            on_participant_connected: Optional callback for participant connected
            on_participant_disconnected: Optional callback for participant disconnected
            on_track_subscribed: Optional callback for track subscribed
            on_track_unsubscribed: Optional callback for track unsubscribed
            on_transcription: Optional callback for transcription events (speaker, text, is_final)
            on_audio_data: Optional callback for audio data (participant_identity, audio_data)
        """
        self.url = url
        self.token = token
        self.mixer = mixer
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.no_playback = no_playback

        # Event callbacks
        self._on_participant_connected_cb = on_participant_connected
        self._on_participant_disconnected_cb = on_participant_disconnected
        self._on_track_subscribed_cb = on_track_subscribed
        self._on_track_unsubscribed_cb = on_track_unsubscribed
        self._on_transcription_cb = on_transcription
        self._on_audio_data_cb = on_audio_data

        # LiveKit objects (created during connect)
        self.room: Optional[Room] = None
        self.audio_source: Optional[rtc.AudioSource] = None

        # Active remote audio tasks
        self._remote_audio_tasks: Dict[str, asyncio.Task] = {}

        # Metrics
        self._metrics = NetworkMetrics()

        logger.info(
            f"Initialized LiveKitNetworkManager: {url}, "
            f"{sample_rate}Hz, {num_channels}ch, no_playback={no_playback}"
        )

    async def connect(self) -> Room:
        """Connect to LiveKit room.

        This establishes the WebSocket connection and sets up event handlers.

        Returns:
            Room object

        Raises:
            Exception: If connection fails
        """
        if self.room is not None:
            logger.warning("Already connected to room")
            return self.room

        logger.info(f"Connecting to LiveKit room at {self.url}...")

        try:
            # Create room
            self.room = Room()

            # Set up event handlers
            self.room.on("participant_connected", self._on_participant_connected)
            self.room.on("participant_disconnected", self._on_participant_disconnected)
            self.room.on("track_subscribed", self._on_track_subscribed)
            self.room.on("track_unsubscribed", self._on_track_unsubscribed)
            # Register text stream handler for transcriptions
            self.room.register_text_stream_handler(
                "lk.transcription", self._handle_text_stream
            )

            # Connect with auto-subscribe
            await self.room.connect(
                self.url, self.token, options=RoomOptions(auto_subscribe=True)
            )

            # Update metrics
            self._metrics.is_connected = True
            self._metrics.room_name = self.room.name

            logger.info(f"Connected to room: {self.room.name}")
            return self.room

        except Exception as e:
            logger.error(f"Failed to connect to room: {e}")
            self.room = None
            self._metrics.is_connected = False
            raise

    async def disconnect(self):
        """Disconnect from LiveKit room.

        This closes all remote audio tasks and disconnects from the room.
        """
        if self.room is None:
            logger.warning("Not connected to room")
            return

        logger.info("Disconnecting from LiveKit room...")

        try:
            # Cancel all remote audio tasks
            for task_id, task in list(self._remote_audio_tasks.items()):
                logger.debug(f"Cancelling remote audio task: {task_id}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            self._remote_audio_tasks.clear()

            # Disconnect from room
            await self.room.disconnect()

            # Update metrics
            self._metrics.is_connected = False
            self._metrics.room_name = None
            self._metrics.participant_count = 0
            self._metrics.tracks_subscribed = 0

            logger.info("Disconnected from room")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            raise
        finally:
            self.room = None
            self.audio_source = None

    async def publish_audio_track(self) -> rtc.LocalAudioTrack:
        """Create and publish local audio track.

        This creates an AudioSource and publishes it as a local audio track.
        The AudioSource can then be used by AudioInputHandler to send audio.

        Returns:
            LocalAudioTrack object

        Raises:
            RuntimeError: If not connected to room
            Exception: If track publishing fails
        """
        if self.room is None:
            raise RuntimeError("Must connect to room before publishing track")

        logger.info("Publishing audio track...")

        try:
            # Create audio source with buffering
            # queue_size_ms: Buffer size for handling jitter
            self.audio_source = rtc.AudioSource(
                self.sample_rate, self.num_channels, queue_size_ms=200
            )

            # Create local audio track
            track = rtc.LocalAudioTrack.create_audio_track(
                "microphone", self.audio_source
            )

            # Publish to room
            options = TrackPublishOptions(source=TrackSource.SOURCE_MICROPHONE)
            await self.room.local_participant.publish_track(track, options)

            logger.info("Audio track published successfully")
            return track

        except Exception as e:
            logger.error(f"Failed to publish audio track: {e}")
            self.audio_source = None
            raise

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle participant connected event.

        Args:
            participant: Remote participant that connected
        """
        logger.info(
            f"Participant connected: {participant.identity} ({participant.name})"
        )

        # Update metrics
        self._metrics.participant_count += 1

        # Call user callback if provided
        if self._on_participant_connected_cb:
            self._on_participant_connected_cb(participant)

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnected event.

        Args:
            participant: Remote participant that disconnected
        """
        logger.info(f"Participant disconnected: {participant.identity}")

        # Update metrics
        self._metrics.participant_count = max(0, self._metrics.participant_count - 1)

        # Remove participant's stream from mixer
        self.mixer.remove_stream(participant.identity)

        # Cancel remote audio task if exists
        if participant.identity in self._remote_audio_tasks:
            task = self._remote_audio_tasks[participant.identity]
            task.cancel()
            del self._remote_audio_tasks[participant.identity]

        # Call user callback if provided
        if self._on_participant_disconnected_cb:
            self._on_participant_disconnected_cb(participant)

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle track subscribed event.

        Args:
            track: Subscribed track
            publication: Track publication
            participant: Remote participant who published the track
        """
        logger.info(
            f"Track subscribed from {participant.identity}: {track.name} ({track.kind})"
        )

        # Update metrics
        self._metrics.tracks_subscribed += 1

        # If it's an audio track, start processing it
        if isinstance(track, rtc.RemoteAudioTrack):
            # Add stream to mixer with participant ID
            self.mixer.add_stream(participant.identity, volume=1.0)

            # Start async task to process audio
            task = asyncio.create_task(
                self._handle_remote_audio_track(track, participant.identity)
            )
            self._remote_audio_tasks[participant.identity] = task

        # Call user callback if provided
        if self._on_track_subscribed_cb:
            self._on_track_subscribed_cb(track, publication, participant)

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle track unsubscribed event.

        Args:
            track: Unsubscribed track
            publication: Track publication
            participant: Remote participant
        """
        logger.info(
            f"Track unsubscribed from {participant.identity}: "
            f"{track.name} ({track.kind})"
        )

        # Update metrics
        self._metrics.tracks_subscribed = max(0, self._metrics.tracks_subscribed - 1)

        # Remove stream from mixer
        self.mixer.remove_stream(participant.identity)

        # Cancel remote audio task
        if participant.identity in self._remote_audio_tasks:
            task = self._remote_audio_tasks[participant.identity]
            task.cancel()
            del self._remote_audio_tasks[participant.identity]

        # Call user callback if provided
        if self._on_track_unsubscribed_cb:
            self._on_track_unsubscribed_cb(track, publication, participant)

    async def _async_handle_text_stream(
        self, reader: rtc.TextStreamReader, participant_identity: str
    ):
        """Handle text stream from a participant (e.g., transcriptions).

        Args:
            reader: Text stream reader
            participant_identity: Identity of the participant sending the stream
        """
        info = reader.info

        logger.debug(
            f"Text stream received from {participant_identity}\n"
            f"  Topic: {info.topic}\n"
            f"  Timestamp: {info.timestamp}\n"
            f"  Size: {getattr(info, 'size', 'unknown')}"
        )

        try:
            # Process the stream incrementally using async for loop
            async for chunk in reader:
                # For transcription streams, each chunk is a transcript
                if info.topic == "lk.transcription":
                    # Determine if this is final or interim based on chunk content
                    # LiveKit typically sends interim results followed by final ones
                    is_final = not chunk.endswith("...")  # Simple heuristic

                    logger.debug(
                        f"Transcription chunk from {participant_identity}: "
                        f"'{chunk}' (final={is_final})"
                    )

                    # Call transcription callback if provided
                    if self._on_transcription_cb:
                        self._on_transcription_cb(participant_identity, chunk, is_final)
                else:
                    # Handle other text stream topics
                    logger.info(
                        f"Text stream from {participant_identity} ({info.topic}): {chunk}"
                    )

        except Exception as e:
            logger.error(
                f"Error processing text stream from {participant_identity}: {e}"
            )

    def _handle_text_stream(
        self, reader: rtc.TextStreamReader, participant_identity: str
    ):
        """Create async task for handling text stream.

        Args:
            reader: Text stream reader
            participant_identity: Identity of the participant sending the stream
        """
        task = asyncio.create_task(
            self._async_handle_text_stream(reader, participant_identity)
        )
        _active_text_tasks.add(task)
        task.add_done_callback(lambda t: _active_text_tasks.remove(t))

    async def _handle_remote_audio_track(
        self, track: rtc.RemoteAudioTrack, participant_identity: str
    ):
        """Process audio frames from a remote participant.

        This async task runs for the lifetime of the remote audio track,
        receiving frames and routing them to the mixer.

        Args:
            track: Remote audio track to process
            participant_identity: Identity of the participant
        """
        logger.info(
            f"Starting audio stream processing for participant: {participant_identity}"
        )

        try:
            # Create audio stream from track
            stream = rtc.AudioStream(
                track, sample_rate=self.sample_rate, num_channels=self.num_channels
            )

            # Process frames as they arrive
            async for event in stream:
                frame = event.frame

                # Convert frame data to numpy array (int16)
                audio_data = np.frombuffer(frame.data, dtype=np.int16)

                # Route to mixer (thread-safe, synchronous)
                if not self.no_playback:
                    self.mixer.add_audio_data(
                        audio_data, stream_id=participant_identity
                    )

                # Notify transcription callback with audio data for visualization
                # (reusing transcription callback mechanism for simplicity)
                # In a production system, you'd have a separate audio_data callback
                if hasattr(self, "_on_audio_data_cb") and self._on_audio_data_cb:
                    self._on_audio_data_cb(participant_identity, audio_data)

                # Update metrics
                self._metrics.frames_received += 1

                logger.debug(
                    f"Received audio frame from {participant_identity}: "
                    f"{len(audio_data)} samples, {frame.num_channels} channels, "
                    f"{frame.sample_rate} Hz"
                )

            logger.info(f"Audio stream ended for participant: {participant_identity}")

        except asyncio.CancelledError:
            logger.info(
                f"Audio stream cancelled for participant: {participant_identity}"
            )
            raise
        except Exception as e:
            logger.error(f"Error processing audio from {participant_identity}: {e}")

    def get_metrics(self) -> dict:
        """Get metrics for monitoring.

        Returns:
            Dictionary with metrics:
                - is_connected: Whether connected to room
                - room_name: Name of room (or None)
                - participant_count: Number of remote participants
                - tracks_subscribed: Number of remote tracks subscribed
                - frames_received: Total frames received
                - active_audio_tasks: Number of active remote audio tasks
        """
        return {
            "is_connected": self._metrics.is_connected,
            "room_name": self._metrics.room_name,
            "participant_count": self._metrics.participant_count,
            "tracks_subscribed": self._metrics.tracks_subscribed,
            "frames_received": self._metrics.frames_received,
            "active_audio_tasks": len(self._remote_audio_tasks),
        }
