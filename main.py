#!/usr/bin/env python3
"""
LiveKit Audio Client
Streams audio to and from LiveKit without using the agents framework.
Based on the Rust local_audio example.
"""

import asyncio
import argparse
import os
import logging
import threading
from typing import Optional
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from livekit import rtc
from livekit.rtc import (
    Room,
    RoomOptions,
    TrackPublishOptions,
    TrackSource,
)
from livekit.api import AccessToken, VideoGrants

# Audio configuration constants
SAMPLE_RATE = 48000
NUM_CHANNELS = 1
SAMPLES_PER_CHANNEL = SAMPLE_RATE // 20  # 50ms chunks for smoother playback

logger = logging.getLogger(__name__)


class AudioMixer:
    """Mix multiple audio streams together for playback.

    This mixer is designed to handle real-time audio with minimal latency:
    - Uses threading.Lock instead of asyncio.Lock for audio callback compatibility
    - Tracks buffer underruns to help diagnose audio issues
    - Handles buffer underruns gracefully by padding with silence
    """

    def __init__(self, sample_rate: int, channels: int, volume: float = 1.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.volume = volume
        self.buffer = np.array([], dtype=np.int16)
        self.lock = threading.Lock()  # Use threading.Lock for real-time audio callbacks
        self._underrun_count = 0
        self._last_underrun_log = 0

    def add_audio_data(self, audio_data: np.ndarray):
        """Add audio data to the mixer buffer (thread-safe, non-async)."""
        with self.lock:
            if len(self.buffer) == 0:
                self.buffer = audio_data.copy()
            else:
                # Append new audio data
                self.buffer = np.concatenate([self.buffer, audio_data])

    def get_audio_data(self, num_samples: int) -> np.ndarray:
        """Get audio data from the mixer buffer (thread-safe, non-async)."""
        with self.lock:
            if len(self.buffer) < num_samples:
                # Buffer underrun - we need to handle this gracefully
                self._underrun_count += 1

                # Log underruns occasionally (every 100 occurrences)
                if self._underrun_count - self._last_underrun_log >= 100:
                    logger.warning(
                        f"Audio buffer underrun (count: {self._underrun_count}). "
                        f"Buffer has {len(self.buffer)} samples, need {num_samples}"
                    )
                    self._last_underrun_log = self._underrun_count

                # Return available data padded with silence
                result = np.zeros(num_samples, dtype=np.int16)
                if len(self.buffer) > 0:
                    result[:len(self.buffer)] = self.buffer
                    self.buffer = np.array([], dtype=np.int16)
                return (result * self.volume).astype(np.int16)
            else:
                # Extract requested samples
                result = self.buffer[:num_samples]
                self.buffer = self.buffer[num_samples:]
                return (result * self.volume).astype(np.int16)

    def buffer_size(self) -> int:
        """Get current buffer size (in samples)."""
        with self.lock:
            return len(self.buffer)

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer = np.array([], dtype=np.int16)


class LiveKitAudioClient:
    """Main audio client for LiveKit."""

    def __init__(
        self,
        url: str,
        token: str,
        sample_rate: int = SAMPLE_RATE,
        channels: int = NUM_CHANNELS,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
        no_playback: bool = False,
        volume: float = 1.0,
    ):
        self.url = url
        self.token = token
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_device = input_device
        self.output_device = output_device
        self.no_playback = no_playback
        self.volume = volume

        self.room: Optional[Room] = None
        self.audio_source: Optional[rtc.AudioSource] = None
        self.mixer = AudioMixer(sample_rate, channels, volume)
        self.running = False
        self.input_stream: Optional[sd.InputStream] = None
        self.output_stream: Optional[sd.OutputStream] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self):
        """Connect to the LiveKit room."""
        logger.info(f"Connecting to LiveKit room at {self.url}...")

        # Store event loop reference for audio callbacks
        self.loop = asyncio.get_event_loop()

        self.room = Room()

        # Set up event handlers
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)
        self.room.on("participant_disconnected", self._on_participant_disconnected)

        # Connect to the room with auto-subscribe
        await self.room.connect(self.url, self.token, options=RoomOptions(auto_subscribe=True))

        logger.info(f"Connected to room: {self.room.name}")
        return self.room

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle participant connection."""
        logger.info(f"Participant connected: {participant.identity} ({participant.name})")

    def _on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant disconnection."""
        logger.info(f"Participant disconnected: {participant.identity}")

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle track subscription."""
        logger.info(
            f"Track subscribed from {participant.identity}: {track.name} ({track.kind})"
        )

        if isinstance(track, rtc.RemoteAudioTrack):
            # Start processing audio from this participant
            asyncio.create_task(
                self._handle_remote_audio_track(track, participant.identity)
            )

    def _on_track_unsubscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        """Handle track unsubscription."""
        logger.info(
            f"Track unsubscribed from {participant.identity}: {track.name} ({track.kind})"
        )

    async def _handle_remote_audio_track(
        self, track: rtc.RemoteAudioTrack, participant_identity: str
    ):
        """Process audio frames from a remote participant."""
        logger.info(f"Starting audio stream processing for participant: {participant_identity}")

        # Create an audio stream from the track
        stream = rtc.AudioStream(track, sample_rate=self.sample_rate, num_channels=self.channels)

        async for event in stream:
            frame = event.frame
            # Convert frame data to numpy array
            audio_data = np.frombuffer(frame.data, dtype=np.int16)

            # Add to mixer for playback (now synchronous)
            if not self.no_playback:
                self.mixer.add_audio_data(audio_data)

            logger.debug(
                f"Received audio frame from {participant_identity}: "
                f"{len(audio_data)} samples, {frame.num_channels} channels, "
                f"{frame.sample_rate} Hz, buffer size: {self.mixer.buffer_size()}"
            )

        logger.info(f"Audio stream ended for participant: {participant_identity}")

    async def publish_audio_track(self):
        """Create and publish the local audio track."""
        # Create audio source with queue for buffering
        self.audio_source = rtc.AudioSource(self.sample_rate, self.channels, queue_size_ms=200)

        # Create local audio track
        track = rtc.LocalAudioTrack.create_audio_track("microphone", self.audio_source)

        # Publish the track
        options = TrackPublishOptions(source=TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, options)

        logger.info("Audio track published to LiveKit successfully")
        return track

    def _audio_input_callback(self, indata, frames, time_info, status):
        """Callback for audio input from microphone."""
        if status:
            logger.warning(f"Audio input status: {status}")

        # Convert to int16 and create audio frame
        audio_data = (indata[:, 0] * 32767).astype(np.int16)

        # Create an audio frame and capture it
        frame = rtc.AudioFrame(
            data=audio_data.tobytes(),
            sample_rate=self.sample_rate,
            num_channels=self.channels,
            samples_per_channel=len(audio_data),
        )

        # Capture the frame asynchronously using the stored loop reference
        asyncio.run_coroutine_threadsafe(
            self.audio_source.capture_frame(frame), self.loop
        )

    def _audio_output_callback(self, outdata, frames, time_info, status):
        """Callback for audio output to speakers."""
        if status:
            logger.warning(f"Audio output status: {status}")

        try:
            # Get audio data from mixer (now synchronous and thread-safe)
            audio_data = self.mixer.get_audio_data(frames)
            # Normalize to float32 in range [-1, 1]
            outdata[:, 0] = audio_data.astype(np.float32) / 32767.0
        except Exception as e:
            logger.error(f"Error in output callback: {e}")
            outdata.fill(0)

    async def start_audio_streams(self):
        """Start audio input and output streams."""
        self.running = True

        # Start input stream (microphone)
        logger.info(f"Starting audio input stream on device {self.input_device}...")
        self.input_stream = sd.InputStream(
            device=self.input_device,
            channels=1,
            samplerate=self.sample_rate,
            callback=self._audio_input_callback,
            blocksize=SAMPLES_PER_CHANNEL,
        )
        self.input_stream.start()
        logger.info("Audio input stream started")

        # Start output stream (speakers) if not disabled
        if not self.no_playback:
            logger.info(f"Starting audio output stream on device {self.output_device}...")
            self.output_stream = sd.OutputStream(
                device=self.output_device,
                channels=1,
                samplerate=self.sample_rate,
                callback=self._audio_output_callback,
                blocksize=SAMPLES_PER_CHANNEL,
            )
            self.output_stream.start()
            logger.info(f"Audio output stream started with volume: {self.volume * 100:.1f}%")
        else:
            logger.info("Audio playback disabled")

    async def stop(self):
        """Stop all audio streams and disconnect."""
        logger.info("Shutting down...")
        self.running = False

        # Stop audio streams
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()

        # Disconnect from room
        if self.room:
            await self.room.disconnect()

        logger.info("Disconnected")


def list_audio_devices():
    """List all available audio input and output devices."""
    print("\nAvailable audio input devices:")
    print("─" * 50)
    for i, device in enumerate(sd.query_devices()):
        if device['max_input_channels'] > 0:
            print(f"{i}. {device['name']}")
            print(f"   └─ Sample rate: {device['default_samplerate']} Hz")
            print(f"   └─ Input channels: {device['max_input_channels']}")
            print()

    print("\nAvailable audio output devices:")
    print("─" * 50)
    for i, device in enumerate(sd.query_devices()):
        if device['max_output_channels'] > 0:
            print(f"{i}. {device['name']}")
            print(f"   └─ Sample rate: {device['default_samplerate']} Hz")
            print(f"   └─ Output channels: {device['max_output_channels']}")
            print()

    # Show default devices
    default_input = sd.query_devices(kind='input')
    default_output = sd.query_devices(kind='output')
    print(f"Default input device: {default_input['name']}")
    print(f"Default output device: {default_output['name']}")


async def main():
    """Main entry point."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="LiveKit Audio Client - Stream audio to/from LiveKit"
    )

    # Device management
    parser.add_argument(
        "-l", "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    parser.add_argument(
        "-i", "--input-device",
        type=int,
        help="Audio input device index (see --list-devices)"
    )
    parser.add_argument(
        "-o", "--output-device",
        type=int,
        help="Audio output device index (see --list-devices)"
    )

    # Audio configuration
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"Sample rate in Hz (default: {SAMPLE_RATE})"
    )
    parser.add_argument(
        "-c", "--channels",
        type=int,
        default=NUM_CHANNELS,
        help=f"Number of channels (default: {NUM_CHANNELS})"
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Disable audio playback (capture only)"
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=1.0,
        help="Master playback volume (0.0 to 1.0, default: 1.0)"
    )

    # LiveKit connection
    parser.add_argument(
        "--url",
        help="LiveKit server URL (or set LIVEKIT_URL env var)"
    )
    parser.add_argument(
        "--api-key",
        help="LiveKit API key (or set LIVEKIT_API_KEY env var)"
    )
    parser.add_argument(
        "--api-secret",
        help="LiveKit API secret (or set LIVEKIT_API_SECRET env var)"
    )
    parser.add_argument(
        "--room-name",
        default="audio-room",
        help="LiveKit room name to join (default: audio-room)"
    )
    parser.add_argument(
        "--identity",
        default="python-audio-streamer",
        help="Participant identity (default: python-audio-streamer)"
    )

    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return

    # Validate volume
    if not 0.0 <= args.volume <= 1.0:
        logger.error("Volume must be between 0.0 and 1.0")
        return

    # Get LiveKit connection details
    url = args.url or os.environ.get("LIVEKIT_URL")
    api_key = args.api_key or os.environ.get("LIVEKIT_API_KEY")
    api_secret = args.api_secret or os.environ.get("LIVEKIT_API_SECRET")

    if not all([url, api_key, api_secret]):
        logger.error(
            "LiveKit URL, API key, and API secret must be provided via "
            "arguments or environment variables (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)"
        )
        return

    # Generate access token
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(args.identity)
        .with_name(args.identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=args.room_name,
            )
        )
        .to_jwt()
    )

    # Create and start the audio client
    client = LiveKitAudioClient(
        url=url,
        token=token,
        sample_rate=args.sample_rate,
        channels=args.channels,
        input_device=args.input_device,
        output_device=args.output_device,
        no_playback=args.no_playback,
        volume=args.volume,
    )

    try:
        # Connect to room
        await client.connect()

        # Publish audio track
        track = await client.publish_audio_track()

        # Start audio streaming
        await client.start_audio_streams()

        logger.info(
            f"Audio streaming started. Room: {args.room_name}, "
            f"Identity: {args.identity}. Press Ctrl+C to stop."
        )

        # Demonstrate muting/unmuting
        await asyncio.sleep(1)
        track.mute()
        logger.info("Microphone muted for 5 seconds...")
        await asyncio.sleep(5)
        track.unmute()
        logger.info("Microphone unmuted.")

        # Wait for Ctrl+C
        await asyncio.Event().wait()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
