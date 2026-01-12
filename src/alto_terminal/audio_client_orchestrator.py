"""Audio client orchestrator for component coordination.

This module provides the AudioClientOrchestrator class that wires together
all audio components and manages their lifecycle.

Key design decisions:
- Single entry point for creating and managing all components
- Proper initialization order (dependencies)
- Clean shutdown with proper cleanup order
- Unified API for external use
- Error handling with partial initialization cleanup
"""

import asyncio
import logging
from typing import Optional
from dataclasses import dataclass
from livekit.rtc import AudioProcessingModule

from .config import AudioConfig
from .audio_mixer import AudioMixer
from .audio_input_handler import AudioInputHandler
from .audio_output_handler import AudioOutputHandler
from .livekit_network_manager import LiveKitNetworkManager
from .tui_manager import TUIManager

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorMetrics:
    """Aggregated metrics from all components.

    Attributes:
        is_connected: Whether connected to LiveKit room
        mixer_metrics: Metrics from AudioMixer
        input_metrics: Metrics from AudioInputHandler
        output_metrics: Metrics from AudioOutputHandler
        network_metrics: Metrics from LiveKitNetworkManager
    """

    is_connected: bool = False
    mixer_metrics: dict = None
    input_metrics: dict = None
    output_metrics: dict = None
    network_metrics: dict = None


class AudioClientOrchestrator:
    """Orchestrate all audio components for LiveKit client.

    This orchestrator manages the lifecycle of all audio components:
    - AudioMixer (buffer management)
    - LiveKitNetworkManager (room/track management)
    - AudioInputHandler (microphone capture)
    - AudioOutputHandler (speaker playback)

    Initialization order (critical for dependencies):
    1. Store event loop (needed for audio callbacks)
    2. Create mixer (no dependencies)
    3. Connect to LiveKit room (needs mixer)
    4. Publish audio track (creates audio_source)
    5. Create input handler (needs audio_source, event_loop)
    6. Create output handler (needs mixer)
    7. Start audio I/O

    Example:
        >>> config = AudioConfig(sample_rate=48000, num_channels=1)
        >>> orchestrator = AudioClientOrchestrator(
        ...     url="wss://livekit.example.com",
        ...     token="...",
        ...     config=config
        ... )
        >>> await orchestrator.connect()
        >>> # ... audio is flowing ...
        >>> await orchestrator.disconnect()
    """

    def __init__(
        self, url: str, token: str, config: AudioConfig, enable_tui: bool = False
    ):
        """Initialize orchestrator.

        Args:
            url: LiveKit server URL
            token: Authentication token
            config: Audio configuration
            enable_tui: Whether to enable terminal UI visualization
        """
        self.url = url
        self.token = token
        self.config = config
        self.enable_tui = enable_tui

        # Components (created during connect)
        self.mixer: Optional[AudioMixer] = None
        self.network_manager: Optional[LiveKitNetworkManager] = None
        self.input_handler: Optional[AudioInputHandler] = None
        self.output_handler: Optional[AudioOutputHandler] = None
        self.tui_manager: Optional[TUIManager] = None
        self.apm: Optional[AudioProcessingModule] = None

        # Event loop reference (for audio callbacks)
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # TUI update task
        self._tui_update_task: Optional[asyncio.Task] = None

        logger.info(
            f"Initialized AudioClientOrchestrator: {url}, "
            f"config={config}, enable_tui={enable_tui}"
        )

    async def connect(self):
        """Connect and start all components.

        This performs initialization in the correct order:
        1. Store event loop
        2. Create mixer
        3. Connect to LiveKit room
        4. Publish audio track
        5. Create audio handlers
        6. Start audio I/O

        Raises:
            Exception: If any step fails (partial cleanup performed)
        """
        logger.info("Connecting audio client...")

        try:
            # Step 1: Store event loop reference for audio callbacks
            self._event_loop = asyncio.get_event_loop()
            logger.debug("Stored event loop reference")

            # Step 2: Create mixer (no dependencies)
            self.mixer = AudioMixer(
                sample_rate=self.config.sample_rate,
                channels=self.config.num_channels,
                master_volume=self.config.volume,
            )
            logger.debug("Created AudioMixer")

            # Step 2.5: Create AudioProcessingModule if AEC or audio processing is enabled
            if any(
                [
                    self.config.enable_aec,
                    self.config.noise_suppression,
                    self.config.high_pass_filter,
                    self.config.auto_gain_control,
                ]
            ):
                self.apm = AudioProcessingModule(
                    echo_cancellation=self.config.enable_aec,
                    noise_suppression=self.config.noise_suppression,
                    high_pass_filter=self.config.high_pass_filter,
                    auto_gain_control=self.config.auto_gain_control,
                )
                logger.info(
                    f"Created AudioProcessingModule: "
                    f"AEC={self.config.enable_aec}, "
                    f"NS={self.config.noise_suppression}, "
                    f"HPF={self.config.high_pass_filter}, "
                    f"AGC={self.config.auto_gain_control}"
                )

            # Step 3: Create TUI manager if enabled (before network manager)
            if self.enable_tui:
                self.tui_manager = TUIManager(
                    max_transcript_lines=20, max_audio_history=50
                )
                logger.debug("Created TUIManager")

            # Step 4: Create network manager and connect to room
            self.network_manager = LiveKitNetworkManager(
                url=self.url,
                token=self.token,
                mixer=self.mixer,
                sample_rate=self.config.sample_rate,
                num_channels=self.config.num_channels,
                no_playback=self.config.no_playback,
                on_transcription=self._on_transcription if self.enable_tui else None,
                on_audio_data=self._on_agent_audio_data if self.enable_tui else None,
            )

            await self.network_manager.connect()
            logger.debug("Connected to LiveKit room")

            # Step 5: Publish audio track (creates audio_source)
            await self.network_manager.publish_audio_track()
            logger.debug("Published audio track")

            # Step 6: Create input handler (needs audio_source, event_loop, optional apm)
            self.input_handler = AudioInputHandler(
                audio_source=self.network_manager.audio_source,
                event_loop=self._event_loop,
                sample_rate=self.config.sample_rate,
                num_channels=self.config.num_channels,
                samples_per_channel=self.config.samples_per_channel,
                device_index=self.config.input_device,
                on_audio_captured=self._on_user_audio_captured
                if self.enable_tui
                else None,
                apm=self.apm,
            )
            logger.debug("Created AudioInputHandler")

            # Step 7: Create output handler (needs mixer, optional apm)
            if not self.config.no_playback:
                self.output_handler = AudioOutputHandler(
                    mixer=self.mixer,
                    sample_rate=self.config.sample_rate,
                    num_channels=self.config.num_channels,
                    samples_per_channel=self.config.samples_per_channel,
                    device_index=self.config.output_device,
                    apm=self.apm,
                )
                logger.debug("Created AudioOutputHandler")

            # Step 8: Start audio I/O
            self.input_handler.start()
            logger.debug("Started audio input")

            if self.output_handler:
                self.output_handler.start()
                logger.debug("Started audio output")

            # Step 9: Start TUI if enabled
            if self.enable_tui and self.tui_manager:
                self.tui_manager.start(refresh_per_second=10)
                logger.debug("Started TUI")

                # Start TUI update loop
                self._tui_update_task = asyncio.create_task(self._tui_update_loop())
                logger.debug("Started TUI update loop")

            logger.info("Audio client connected and streaming")

        except Exception as e:
            logger.error(f"Failed to connect audio client: {e}")
            # Cleanup partial initialization
            await self._cleanup()
            raise

    async def disconnect(self):
        """Disconnect and stop all components.

        This performs cleanup in reverse order:
        1. Stop audio I/O
        2. Disconnect from LiveKit room
        3. Clear component references
        """
        logger.info("Disconnecting audio client...")

        try:
            await self._cleanup()
            logger.info("Audio client disconnected")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            raise

    async def _tui_update_loop(self):
        """Update TUI display periodically.

        Note: User audio visualization is now updated in real-time via
        the _on_user_audio_captured callback (called from OS audio thread).
        This loop only handles the TUI refresh.
        """
        try:
            while True:
                if self.tui_manager and self.tui_manager.is_running():
                    # Update the TUI display
                    self.tui_manager.update()

                await asyncio.sleep(0.1)  # Update every 100ms

        except asyncio.CancelledError:
            logger.debug("TUI update loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in TUI update loop: {e}")

    def _on_transcription(self, speaker: str, text: str, is_final: bool):
        """Handle transcription events from LiveKit.

        Args:
            speaker: Speaker identity (participant identity)
            text: Transcribed text
            is_final: Whether this is a final or interim transcript (ignored, all shown)
        """
        if self.tui_manager:
            # Determine speaker label
            if self.network_manager and self.network_manager.room:
                local_identity = self.network_manager.room.local_participant.identity
                speaker_label = "You" if speaker == local_identity else speaker
            else:
                speaker_label = speaker

            # Add to TUI
            self.tui_manager.add_transcript(speaker_label, text)
            logger.debug(f"Transcription: {speaker_label}: {text} (final={is_final})")

    def _on_agent_audio_data(self, participant_identity: str, audio_data):
        """Handle audio data from remote participants for visualization.

        Args:
            participant_identity: Identity of the participant
            audio_data: Audio samples (numpy array, int16)
        """
        if self.tui_manager:
            # Update agent audio visualization
            self.tui_manager.update_agent_audio_level(
                audio_data, stream_id=participant_identity
            )

    def _on_user_audio_captured(self, audio_data):
        """Handle captured user audio for visualization.

        Called from OS audio thread by AudioInputHandler.

        Args:
            audio_data: Captured audio samples (numpy array, int16)
        """
        if self.tui_manager:
            # Update user audio visualization
            # Note: This is called from OS audio thread, but TUI methods are thread-safe
            self.tui_manager.update_user_audio_level(audio_data)

    async def _cleanup(self):
        """Cleanup all components (internal helper).

        This is called both during normal shutdown and error recovery.
        Handles partial initialization gracefully.
        """
        # Stop TUI first
        if self._tui_update_task:
            try:
                self._tui_update_task.cancel()
                await self._tui_update_task
            except asyncio.CancelledError:
                pass
            finally:
                self._tui_update_task = None

        if self.tui_manager:
            try:
                self.tui_manager.stop()
                logger.debug("Stopped TUI")
            except Exception as e:
                logger.error(f"Error stopping TUI: {e}")
            finally:
                self.tui_manager = None

        # Stop audio output first (stops pulling from mixer)
        if self.output_handler:
            try:
                self.output_handler.stop()
                logger.debug("Stopped audio output")
            except Exception as e:
                logger.error(f"Error stopping audio output: {e}")
            finally:
                self.output_handler = None

        # Stop audio input (stops sending to LiveKit)
        if self.input_handler:
            try:
                self.input_handler.stop()
                logger.debug("Stopped audio input")
            except Exception as e:
                logger.error(f"Error stopping audio input: {e}")
            finally:
                self.input_handler = None

        # Disconnect from LiveKit room
        if self.network_manager:
            try:
                await self.network_manager.disconnect()
                logger.debug("Disconnected from LiveKit")
            except Exception as e:
                logger.error(f"Error disconnecting from LiveKit: {e}")
            finally:
                self.network_manager = None

        # Clear other references
        self.mixer = None
        self._event_loop = None

    def set_master_volume(self, volume: float):
        """Set master volume.

        Args:
            volume: Volume level (0.0 to 1.0)

        Raises:
            RuntimeError: If not connected
        """
        if not self.mixer:
            raise RuntimeError("Must connect before setting volume")

        self.mixer.master_volume = volume
        logger.info(f"Set master volume to {volume}")

    def set_stream_volume(self, stream_id: str, volume: float):
        """Set volume for specific stream (participant).

        Args:
            stream_id: Stream identifier (participant identity)
            volume: Volume level (0.0 to 1.0)

        Raises:
            RuntimeError: If not connected
        """
        if not self.mixer:
            raise RuntimeError("Must connect before setting volume")

        self.mixer.set_stream_volume(stream_id, volume)
        logger.info(f"Set stream {stream_id} volume to {volume}")

    def get_metrics(self) -> dict:
        """Get aggregated metrics from all components.

        Returns:
            Dictionary with metrics from all components:
                - is_connected: Overall connection status
                - mixer: AudioMixer metrics
                - input: AudioInputHandler metrics
                - output: AudioOutputHandler metrics (or None if no_playback)
                - network: LiveKitNetworkManager metrics
        """
        return {
            "is_connected": self.network_manager is not None
            and self.network_manager.get_metrics()["is_connected"],
            "mixer": self.mixer.get_metrics() if self.mixer else {},
            "input": self.input_handler.get_metrics() if self.input_handler else {},
            "output": self.output_handler.get_metrics()
            if self.output_handler
            else None,
            "network": self.network_manager.get_metrics()
            if self.network_manager
            else {},
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        return False
