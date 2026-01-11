# LiveKit Audio Client

A Python audio client that streams audio to and from LiveKit without using the agents framework. This is a Python equivalent of the Rust `local_audio` example from the LiveKit Rust SDK.

## Features

- Stream audio from your microphone to a LiveKit room
- Receive and play audio from other participants
- Support for custom audio devices
- Adjustable volume control
- Mute/unmute functionality
- **Terminal UI (TUI) visualization** with conversation transcripts and real-time audio levels
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

### Terminal UI Mode

Enable the Terminal User Interface for visual feedback:

```bash
python main.py --room-name my-room --identity my-username --tui
```

The TUI provides:
- **Top panel**: Conversation transcripts showing messages from all participants
- **Bottom-left panel**: Your microphone audio visualization (VU meter and waveform) - **real-time from your mic**
- **Bottom-right panel**: Agent/remote participant audio visualization - **real-time from remote audio**

**Note**: The TUI currently supports transcription display when connected to LiveKit rooms with transcription enabled (e.g., when using LiveKit Agents that publish transcriptions to the `lk.transcription` data topic).

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
--tui                      Enable terminal UI with visualization
-v, --verbose              Enable verbose logging
```

## How It Works

The client uses the LiveKit Python client SDK (`livekit-rtc`) to:

1. **Connect to a LiveKit room** with auto-subscribe enabled
2. **Capture audio** from your microphone using sounddevice
3. **Publish audio frames** to LiveKit via `AudioSource.capture_frame()`
4. **Subscribe to remote audio tracks** from other participants
5. **Mix and play back** received audio through your speakers

## Architecture

Alto Terminal is built with a modular, well-tested architecture consisting of five key components:

```
AudioClientOrchestrator
    ├─> AudioMixer (buffer management & mixing)
    ├─> LiveKitNetworkManager (room & track management)
    ├─> AudioInputHandler (microphone capture)
    └─> AudioOutputHandler (speaker playback)
```

### Component Overview

- **AudioMixer**: Thread-safe audio buffer management with per-stream volume control and lazy mixing
- **AudioInputHandler**: Captures microphone audio and bridges OS audio thread → async event loop
- **AudioOutputHandler**: Pulls mixed audio from mixer and plays through speakers
- **LiveKitNetworkManager**: Manages LiveKit room connection, track publishing, and remote audio processing
- **AudioClientOrchestrator**: Coordinates component lifecycle and provides unified API

### Key Features

- **Separation of Concerns**: Each component has a single, well-defined responsibility
- **Thread Safety**: Proper synchronization between OS audio threads and async event loop
- **Per-Participant Mixing**: Individual volume control and metrics for each remote participant
- **Graceful Degradation**: Handles buffer underruns and network issues without crashing
- **Comprehensive Testing**: 103 tests (99 unit + 4 integration) with >80% coverage

### Understanding the Code

For a deep dive into the architecture, see:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Component diagrams, data flows, threading model
- **[DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)** - Why things are built this way
- **[TUTORIAL.md](docs/TUTORIAL.md)** - Step-by-step walkthrough of audio flow

### Project Structure

```
alto-terminal/
├── src/alto_terminal/          # Main package
│   ├── config.py               # Audio configuration
│   ├── audio_mixer.py          # Buffer management & mixing
│   ├── audio_input_handler.py  # Microphone capture
│   ├── audio_output_handler.py # Speaker playback
│   ├── livekit_network_manager.py # LiveKit room management
│   ├── audio_client_orchestrator.py # Component coordination
│   └── utils.py                # Device listing utilities
├── tests/                      # Comprehensive test suite
│   ├── unit/                   # 99 unit tests
│   └── integration/            # 4 integration tests
├── docs/                       # Architecture documentation
│   ├── ARCHITECTURE.md
│   ├── DESIGN_DECISIONS.md
│   └── TUTORIAL.md
└── main.py                     # CLI entry point
```

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
