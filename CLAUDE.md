# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alto Terminal is a LiveKit audio client that streams audio to/from LiveKit rooms without using the LiveKit Agents framework. It uses only the LiveKit Python client SDK (`livekit-rtc`) and provides real-time bidirectional audio streaming with features like echo cancellation, noise suppression, and optional TUI visualization.

## Development Commands

### Package Management
This project uses `uv` for all Python operations:

```bash
# Install dependencies
uv sync

# Add new packages
uv add <package>

# Add test dependencies
uv add --dev pytest pytest-cov

# Run the application
uv run python main.py --room-name my-room --identity alice
```

### Running the Application

```bash
# List available audio devices
uv run python main.py --list-devices

# Basic usage with environment variables
export LIVEKIT_URL="wss://your-server.com"
export LIVEKIT_API_KEY="your-key"
export LIVEKIT_API_SECRET="your-secret"
uv run python main.py --room-name test-room --identity alice

# With echo cancellation and noise suppression
uv run python main.py --room-name test --identity alice --enable-aec --noise-suppression

# With TUI visualization
uv run python main.py --room-name test --identity alice --tui

# Verbose logging
uv run python main.py --room-name test --identity alice -v
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/alto_terminal --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_audio_mixer.py

# Run specific test
uv run pytest tests/unit/test_audio_mixer.py::TestAudioMixer::test_mix_multiple_streams

# Run integration tests only
uv run pytest tests/integration/

# Run unit tests only
uv run pytest tests/unit/
```

### Code Quality

```bash
# Syntax check all Python files
uv run python -m py_compile src/alto_terminal/*.py

# Type checking (if using mypy)
uv run mypy src/alto_terminal/
```

## Architecture

Alto Terminal uses a **5-component modular architecture** with strict separation of concerns:

```
AudioClientOrchestrator (lifecycle coordinator)
    ├─> AudioMixer (thread-safe buffer & mixing)
    ├─> LiveKitNetworkManager (room & track management)
    ├─> AudioInputHandler (mic capture → LiveKit)
    └─> AudioOutputHandler (LiveKit → speakers)
```

### Critical Threading Model

The system operates across **three execution contexts**:

1. **OS Audio Threads** (high-priority, real-time)
   - Run audio callbacks from `sounddevice`
   - MUST complete in <10ms
   - NO blocking operations allowed
   - NO async/await allowed

2. **Async Event Loop Thread** (I/O operations)
   - Runs all LiveKit SDK operations
   - Handles room connection, track subscription
   - Processes remote audio frames

3. **Thread-Safe Components**
   - `AudioMixer` uses `threading.Lock` (NOT `asyncio.Lock`)
   - Can be called from ANY thread
   - Bridges OS audio threads ↔ async event loop

### Key Design Patterns

**Bridging OS Audio Thread → Async Event Loop:**
```python
def _audio_callback(self, indata, frames, time_info, status):
    # Running in OS audio thread (synchronous)
    audio_data = (indata[:, 0] * 32767).astype(np.int16)
    frame = rtc.AudioFrame(...)

    # Bridge to async event loop (non-blocking)
    asyncio.run_coroutine_threadsafe(
        self.audio_source.capture_frame(frame),
        self.event_loop  # Stored during initialization
    )
```

**Thread-Safe Mixer Access:**
```python
class AudioMixer:
    def __init__(self):
        self._lock = threading.Lock()  # NOT asyncio.Lock!

    def add_audio_data(self, audio_data, stream_id):
        with self._lock:  # Can be called from any thread
            self._streams[stream_id] = ...
```

### Component Initialization Order

The initialization order is **critical** due to dependencies:

1. Store event loop reference (needed by AudioInputHandler)
2. Create AudioMixer (no dependencies)
3. Create AudioProcessingModule if AEC enabled (no dependencies)
4. Create LiveKitNetworkManager (needs mixer)
5. Connect to LiveKit room (establishes connection)
6. Publish audio track (creates AudioSource)
7. Create AudioInputHandler (needs AudioSource + event loop + optional APM)
8. Create AudioOutputHandler (needs mixer + optional APM)
9. Start audio I/O (start handlers)

**If this order is violated, the system will fail to initialize.**

### Audio Processing Module (AEC)

The system supports WebRTC-based audio processing via `AudioProcessingModule`:

**Features:**
- Echo cancellation (AEC) - removes speaker audio from microphone
- Noise suppression - reduces background noise
- High-pass filter - removes low-frequency rumble
- Auto gain control - normalizes microphone volume

**Critical AEC Requirements:**
- APM requires **10ms frames** (480 samples @ 48kHz)
- System uses **50ms frames**, so must split into 5x 10ms chunks
- **Capture stream** (mic) processed via `apm.process_stream()`
- **Reverse stream** (speakers) processed via `apm.process_reverse_stream()`
- Same APM instance shared between input and output handlers

**AEC Data Flow:**
```
Speaker Audio → AudioOutputHandler → process_reverse_stream() → APM
                                                                   ↓
Mic Audio → AudioInputHandler → process_stream() → APM → Cleaned Audio
```

### Audio Format Conversions

**Internal Format:** `int16` (2 bytes per sample)

**Conversions:**
- Input: `float32` (sounddevice) → `int16` (mixer) via `× 32767`
- Output: `int16` (mixer) → `float32` (sounddevice) via `÷ 32767`
- LiveKit: expects `int16` PCM (no conversion needed)

**Why int16?**
- LiveKit's native format (minimize conversions)
- 50% memory savings vs float32
- Sufficient precision for audio (96 dB dynamic range)

## File Organization

```
src/alto_terminal/
├── config.py                    # AudioConfig dataclass with all settings
├── audio_mixer.py               # Thread-safe buffer management & mixing
├── audio_input_handler.py       # Microphone capture (OS thread → event loop)
├── audio_output_handler.py      # Speaker playback (event loop → OS thread)
├── livekit_network_manager.py   # Room connection & track management
├── audio_client_orchestrator.py # Lifecycle coordinator (start/stop all)
├── tui_manager.py               # Terminal UI visualization
└── utils.py                     # Device listing utilities

tests/
├── unit/                        # 99 unit tests (isolated component tests)
└── integration/                 # 4 integration tests (component interactions)

docs/
├── ARCHITECTURE.md              # Deep dive into system design
├── DESIGN_DECISIONS.md          # Why things are built this way
└── TUTORIAL.md                  # Step-by-step walkthrough
```

## Common Modification Patterns

### Adding a New Audio Processing Feature

1. Add configuration to `AudioConfig` in `config.py`
2. Update `AudioInputHandler` or `AudioOutputHandler` to use the feature
3. Update `AudioClientOrchestrator` to initialize the feature
4. Add CLI flag in `main.py`
5. Update README with usage example
6. Add tests in `tests/unit/`

### Modifying Audio Processing Logic

**For microphone audio (before sending to LiveKit):**
- Modify `AudioInputHandler._audio_callback()`
- Insert processing between conversion and `capture_frame()`
- Remember: runs in OS audio thread, must be FAST (<10ms)

**For speaker audio (after receiving from LiveKit):**
- Modify `AudioOutputHandler._audio_callback()` or `AudioMixer.get_audio_data()`
- Insert processing before format conversion
- Remember: runs in OS audio thread, must be FAST (<10ms)

### Adding Echo Cancellation Features

When modifying AEC:
- APM requires 10ms frames - always split 50ms → 5x 10ms chunks
- Process reverse stream (speakers) BEFORE capture stream (mic)
- Use same APM instance for both input and output handlers
- Test with actual echo scenarios (play audio from speakers)

## Important Constraints

### Threading Constraints

1. **Audio callbacks MUST complete in <10ms:**
   - No blocking I/O
   - No network calls
   - No long computations
   - No `await` (callbacks are synchronous)

2. **AudioMixer MUST use `threading.Lock`:**
   - OS audio threads can't use `asyncio.Lock`
   - Must be callable from both audio threads and event loop

3. **Use `run_coroutine_threadsafe()` to bridge threads:**
   - Only way to call async code from OS audio thread
   - Non-blocking (returns immediately)
   - Stores event loop reference in `__init__`

### Audio Format Constraints

1. **Sample rate: 48000 Hz** (standard for real-time audio)
2. **Channels: 1 (mono)** (sufficient for voice, reduces bandwidth)
3. **Block size: 50ms (2400 samples)** (balance latency vs CPU)
4. **Internal format: int16** (LiveKit native, memory efficient)

### Initialization Constraints

1. **Must initialize in exact order** (see "Component Initialization Order")
2. **Event loop must be stored before AudioInputHandler creation**
3. **AudioSource created by `publish_audio_track()`, not directly**
4. **APM must be created before handlers if AEC enabled**

## Testing Guidelines

### Unit Test Structure

Each component has isolated tests with mocked dependencies:
- Mock `rtc.AudioSource` for AudioInputHandler
- Mock `AudioMixer` for AudioOutputHandler
- Mock `rtc.Room` for LiveKitNetworkManager

### Integration Test Focus

Integration tests verify:
- Component wiring through orchestrator
- Initialization order correctness
- Cleanup on errors (partial initialization)
- Metrics aggregation

### Manual Testing Scenarios

1. **Two-client test:** Alice and Bob in same room
2. **Echo test:** Play audio from speakers, verify AEC works
3. **Network hiccup:** Disconnect WiFi briefly, check recovery
4. **Device switching:** Change audio device while running
5. **TUI test:** Verify real-time audio visualization

## Error Handling Philosophy

1. **Graceful degradation:** Return silence on underrun, don't crash
2. **Log but continue:** Audio thread MUST NOT raise exceptions
3. **Cleanup on failure:** Orchestrator cleans up partial initialization
4. **Informative metrics:** Track errors, underruns, frames for debugging

## Documentation References

- **ARCHITECTURE.md** - Component diagrams, data flows, threading model
- **DESIGN_DECISIONS.md** - Why each design choice was made, alternatives considered
- **TUTORIAL.md** - Step-by-step walkthrough of audio flow
- **README.md** - User-facing usage documentation
