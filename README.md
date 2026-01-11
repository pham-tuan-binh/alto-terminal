# LiveKit Audio Client

A Python audio client that streams audio to and from LiveKit without using the agents framework. This is a Python equivalent of the Rust `local_audio` example from the LiveKit Rust SDK.

## Features

- Stream audio from your microphone to a LiveKit room
- Receive and play audio from other participants
- Support for custom audio devices
- Adjustable volume control
- Mute/unmute functionality
- No dependency on the LiveKit agents framework - uses only the client SDK

## Installation

Install dependencies using uv (or pip):

```bash
uv pip install -e .
```

Or manually:

```bash
pip install livekit numpy sounddevice
```

## Usage

### List Available Audio Devices

```bash
python main.py --list-devices
```

### Basic Usage

Set your LiveKit credentials as environment variables:

```bash
export LIVEKIT_URL="wss://your-livekit-server.com"
export LIVEKIT_API_KEY="your-api-key"
export LIVEKIT_API_SECRET="your-api-secret"
```

Then run the client:

```bash
python main.py --room-name my-room --identity my-username
```

### Advanced Usage

Specify custom audio devices and settings:

```bash
python main.py \
    --url wss://your-livekit-server.com \
    --api-key your-api-key \
    --api-secret your-api-secret \
    --room-name my-room \
    --identity my-username \
    --input-device 0 \
    --output-device 1 \
    --sample-rate 48000 \
    --volume 0.8 \
    --verbose
```

### Capture-Only Mode

Disable audio playback (capture only):

```bash
python main.py --no-playback
```

## Command-Line Options

```
-l, --list-devices          List available audio devices and exit
-i, --input-device INT      Audio input device index
-o, --output-device INT     Audio output device index
-s, --sample-rate INT       Sample rate in Hz (default: 48000)
-c, --channels INT          Number of channels (default: 1)
--no-playback              Disable audio playback (capture only)
--volume FLOAT             Master playback volume (0.0 to 1.0, default: 1.0)
--url URL                  LiveKit server URL
--api-key KEY              LiveKit API key
--api-secret SECRET        LiveKit API secret
--room-name NAME           LiveKit room name (default: audio-room)
--identity NAME            Participant identity (default: python-audio-streamer)
-v, --verbose              Enable verbose logging
```

## How It Works

The client uses the LiveKit Python client SDK (`livekit-rtc`) to:

1. **Connect to a LiveKit room** with auto-subscribe enabled
2. **Capture audio** from your microphone using sounddevice
3. **Publish audio frames** to LiveKit via `AudioSource.capture_frame()`
4. **Subscribe to remote audio tracks** from other participants
5. **Mix and play back** received audio through your speakers

### Architecture

- `LiveKitAudioClient`: Main class managing the LiveKit connection and audio streams
- `AudioMixer`: Mixes multiple remote audio streams for playback
- `sounddevice`: Handles microphone capture and speaker playback
- Callbacks bridge the synchronous audio callbacks with async LiveKit APIs

## Example: Two Clients

Run two clients to test audio streaming between them:

**Terminal 1:**
```bash
python main.py --identity alice --room-name test-room
```

**Terminal 2:**
```bash
python main.py --identity bob --room-name test-room
```

Alice and Bob will be able to hear each other in the room.

## Differences from the Rust Example

This Python implementation closely mirrors the Rust `local_audio` example but with some differences:

- Uses `sounddevice` instead of `cpal` for audio I/O
- Simplified echo cancellation (not included in this basic version)
- Uses Python's `asyncio` for async operations instead of Tokio
- Event handlers use LiveKit's Python SDK event system

## Requirements

- Python 3.8+
- LiveKit server (cloud or self-hosted)
- Microphone and speakers (optional)

## License

MIT
