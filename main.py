#!/usr/bin/env python3
"""LiveKit Audio Client - Stream audio to/from LiveKit rooms.

This script provides a command-line interface for connecting to LiveKit rooms
and streaming audio. It supports microphone input, speaker output, and
per-participant audio mixing.

Example usage:
    # List available audio devices
    python main.py --list-devices

    # Connect to a room
    python main.py --room-name my-room --identity my-name

    # Use specific audio devices
    python main.py --input-device 2 --output-device 3

    # Capture only (no playback)
    python main.py --no-playback
"""

import os
import asyncio
import logging
import argparse
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

from src.alto_terminal.config import AudioConfig
from src.alto_terminal.audio_client_orchestrator import AudioClientOrchestrator
from src.alto_terminal.utils import list_audio_devices

logger = logging.getLogger(__name__)


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
        default=48000,
        help="Sample rate in Hz (default: 48000)"
    )
    parser.add_argument(
        "-c", "--channels",
        type=int,
        default=1,
        help="Number of channels (default: 1)"
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

    # TUI visualization
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Enable terminal UI with conversation and audio visualization"
    )

    args = parser.parse_args()

    # Set up logging
    # When TUI is enabled, reduce logging to WARNING to avoid interfering with display
    log_level = logging.WARNING if args.tui else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
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

    # Create audio configuration
    config = AudioConfig(
        sample_rate=args.sample_rate,
        num_channels=args.channels,
        input_device=args.input_device,
        output_device=args.output_device,
        volume=args.volume,
        no_playback=args.no_playback
    )

    # Create orchestrator
    orchestrator = AudioClientOrchestrator(
        url=url,
        token=token,
        config=config,
        enable_tui=args.tui
    )

    try:
        # Connect and start streaming (all in one call)
        await orchestrator.connect()

        if not args.tui:
            logger.info(
                f"Audio streaming started. Room: {args.room_name}, "
                f"Identity: {args.identity}. Press Ctrl+C to stop."
            )

        # Wait for Ctrl+C
        await asyncio.Event().wait()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await orchestrator.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
