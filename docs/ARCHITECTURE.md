# Architecture Overview

This document provides a comprehensive overview of the Alto Terminal audio client architecture, explaining how components interact, data flows through the system, and key design patterns.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Threading Model](#threading-model)
5. [Component Details](#component-details)
6. [Initialization Order](#initialization-order)
7. [Error Handling](#error-handling)

---

## System Overview

Alto Terminal is a LiveKit audio client that streams audio between a microphone, LiveKit room, and speakers. The architecture is designed around five key principles:

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Testability**: Components are loosely coupled and easily mockable
3. **Thread Safety**: Proper synchronization between OS audio threads and async event loop
4. **Error Recovery**: Graceful degradation and cleanup on failures
5. **Observability**: Comprehensive metrics from all components

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AudioClientOrchestrator                      │
│                  (Lifecycle & Coordination)                     │
└────┬──────────────┬─────────────┬────────────────┬─────────────┘
     │              │             │                │
     ▼              ▼             ▼                ▼
┌─────────┐  ┌─────────────┐  ┌────────┐  ┌──────────────┐
│  Audio  │  │   LiveKit   │  │ Audio  │  │    Audio     │
│  Mixer  │  │   Network   │  │ Input  │  │    Output    │
│         │  │   Manager   │  │Handler │  │   Handler    │
└─────────┘  └─────────────┘  └────────┘  └──────────────┘
     ▲              ▲             │                │
     │              │             ▼                ▼
     │              │         ┌────────┐      ┌─────────┐
     └──────────────┴─────────┤  SDK   │      │ Speakers│
                              │LiveKit │      └─────────┘
                              └────────┘
```

---

## Component Architecture

### 1. AudioMixer

**Responsibility**: Buffer management and audio mixing

**Key Features**:
- Per-stream audio buffers (one per remote participant)
- Lazy mixing (only when audio is requested)
- Per-stream volume control
- Master volume control
- Thread-safe operations (uses `threading.Lock`)
- Graceful underrun handling (silence padding)

**Location**: `src/alto_terminal/audio_mixer.py`

**API**:
```python
mixer = AudioMixer(sample_rate=48000, channels=1, master_volume=1.0)
mixer.add_stream("alice", volume=1.0)
mixer.add_audio_data(audio_data, stream_id="alice")
mixed_audio = mixer.get_audio_data(num_samples=2400)
mixer.set_stream_volume("alice", 0.5)
metrics = mixer.get_metrics()
```

### 2. AudioInputHandler

**Responsibility**: Microphone capture and format conversion

**Key Features**:
- Encapsulates sounddevice InputStream
- Converts float32 (sounddevice) → int16 (LiveKit)
- Bridges OS audio thread → async event loop
- Error tracking and metrics

**Location**: `src/alto_terminal/audio_input_handler.py`

**API**:
```python
handler = AudioInputHandler(
    audio_source=livekit_audio_source,
    event_loop=asyncio.get_event_loop(),
    sample_rate=48000,
    num_channels=1,
    samples_per_channel=2400
)
handler.start()
# ... audio is captured ...
handler.stop()
```

### 3. AudioOutputHandler

**Responsibility**: Speaker playback and format conversion

**Key Features**:
- Encapsulates sounddevice OutputStream
- Pulls from AudioMixer in real-time
- Converts int16 (mixer) → float32 (sounddevice)
- Underrun detection and metrics

**Location**: `src/alto_terminal/audio_output_handler.py`

**API**:
```python
handler = AudioOutputHandler(
    mixer=mixer,
    sample_rate=48000,
    num_channels=1,
    samples_per_channel=2400
)
handler.start()
# ... audio is played ...
handler.stop()
```

### 4. LiveKitNetworkManager

**Responsibility**: Room connection and track management

**Key Features**:
- Room connection/disconnection
- Audio track publishing
- Remote audio track processing
- Event routing (participants, tracks)
- Per-participant stream management

**Location**: `src/alto_terminal/livekit_network_manager.py`

**API**:
```python
manager = LiveKitNetworkManager(
    url="wss://livekit.example.com",
    token="...",
    mixer=mixer,
    sample_rate=48000,
    num_channels=1
)
await manager.connect()
await manager.publish_audio_track()
# ... audio streams ...
await manager.disconnect()
```

### 5. AudioClientOrchestrator

**Responsibility**: Component coordination and lifecycle management

**Key Features**:
- Single entry point for all operations
- Proper initialization order
- Clean shutdown with error handling
- Unified API for external use
- Metrics aggregation

**Location**: `src/alto_terminal/audio_client_orchestrator.py`

**API**:
```python
orchestrator = AudioClientOrchestrator(
    url="wss://livekit.example.com",
    token="...",
    config=AudioConfig(sample_rate=48000, num_channels=1)
)
await orchestrator.connect()  # Initializes all components
# ... audio streams ...
await orchestrator.disconnect()  # Clean shutdown
```

---

## Data Flow Diagrams

### Microphone → LiveKit Flow

```
┌──────────┐
│Microphone│
└────┬─────┘
     │ float32 audio samples
     ▼
┌─────────────────────────────────────┐
│  AudioInputHandler                  │
│  (OS Audio Thread)                  │
├─────────────────────────────────────┤
│ 1. Receive float32 samples          │
│ 2. Convert to int16 (×32767)        │
│ 3. Create AudioFrame                │
│ 4. asyncio.run_coroutine_threadsafe │
└────┬────────────────────────────────┘
     │ int16 audio frame
     ▼
┌─────────────────────────────────────┐
│  AudioSource.capture_frame()        │
│  (Async Event Loop)                 │
├─────────────────────────────────────┤
│ 1. Queue frame for transmission     │
│ 2. Buffer with 200ms queue          │
└────┬────────────────────────────────┘
     │ buffered frames
     ▼
┌─────────────────────────────────────┐
│  LiveKit SDK                        │
│  (WebSocket/Network)                │
├─────────────────────────────────────┤
│ 1. Encode audio (Opus)              │
│ 2. Send over WebSocket              │
└────┬────────────────────────────────┘
     │
     ▼
   Network
```

### LiveKit → Speakers Flow

```
   Network
     │
     ▼
┌─────────────────────────────────────┐
│  LiveKit SDK                        │
│  (WebSocket/Network)                │
├─────────────────────────────────────┤
│ 1. Receive encoded audio (Opus)     │
│ 2. Decode to PCM                    │
└────┬────────────────────────────────┘
     │ decoded frames
     ▼
┌─────────────────────────────────────┐
│  LiveKitNetworkManager              │
│  (_handle_remote_audio_track)       │
│  (Async Event Loop)                 │
├─────────────────────────────────────┤
│ 1. Iterate AudioStream              │
│ 2. Extract frame.data (bytes)       │
│ 3. Convert to int16 array           │
│ 4. Route to mixer                   │
└────┬────────────────────────────────┘
     │ int16 samples
     ▼
┌─────────────────────────────────────┐
│  AudioMixer                         │
│  (Thread-safe, Synchronous)         │
├─────────────────────────────────────┤
│ 1. add_audio_data(stream_id)        │
│ 2. Buffer per participant           │
│ 3. Lazy mixing on request           │
└────┬────────────────────────────────┘
     │ mixed int16 samples
     ▼
┌─────────────────────────────────────┐
│  AudioOutputHandler                 │
│  (OS Audio Thread)                  │
├─────────────────────────────────────┤
│ 1. mixer.get_audio_data()           │
│ 2. Convert int16 → float32 (÷32767) │
│ 3. Write to output buffer           │
└────┬────────────────────────────────┘
     │ float32 audio samples
     ▼
┌──────────┐
│ Speakers │
└──────────┘
```

---

## Threading Model

Understanding the threading model is crucial for understanding the architecture. The system uses three types of execution contexts:

### 1. OS Audio Threads (Real-Time, High Priority)

**Used By**: AudioInputHandler callbacks, AudioOutputHandler callbacks

**Characteristics**:
- High-priority, real-time threads managed by sounddevice
- Must complete FAST (< 10ms typically)
- No blocking operations allowed
- No async/await allowed

**Example**:
```python
def _audio_callback(self, indata, frames, time_info, status):
    # This runs in OS audio thread - must be FAST!
    audio_data = (indata[:, 0] * 32767).astype(np.int16)

    # Bridge to async world (non-blocking call)
    asyncio.run_coroutine_threadsafe(
        self.audio_source.capture_frame(frame),
        self.event_loop  # Reference to async event loop
    )
```

### 2. Async Event Loop Thread (I/O Operations)

**Used By**: LiveKitNetworkManager, AudioClientOrchestrator

**Characteristics**:
- Single thread running asyncio event loop
- Handles all LiveKit SDK operations (network I/O)
- Processes remote audio frames
- Can await async operations

**Example**:
```python
async def _handle_remote_audio_track(self, track, participant_identity):
    # This runs in async event loop
    stream = rtc.AudioStream(track, ...)

    async for event in stream:  # Async iteration
        frame = event.frame
        audio_data = np.frombuffer(frame.data, dtype=np.int16)

        # Call synchronous, thread-safe mixer method
        self.mixer.add_audio_data(audio_data, stream_id=participant_identity)
```

### 3. Synchronous/Thread-Safe Components

**Used By**: AudioMixer

**Characteristics**:
- Can be called from any thread (OS audio or async event loop)
- Uses `threading.Lock` for synchronization
- All methods are synchronous (no async/await)
- Critical for bridging the two execution contexts

**Example**:
```python
def add_audio_data(self, audio_data, stream_id="default"):
    with self._lock:  # threading.Lock, not asyncio.Lock
        # Thread-safe operation
        self._streams[stream_id] = np.concatenate([...])
```

### Thread Interaction Diagram

```
┌─────────────────────────┐         ┌──────────────────────────┐
│   OS Audio Thread       │         │  Async Event Loop Thread │
│   (Input Callback)      │         │                          │
├─────────────────────────┤         ├──────────────────────────┤
│ 1. Capture audio        │         │ 4. Receive coroutine     │
│ 2. Convert float→int16  │         │ 5. Execute in event loop │
│ 3. Bridge via           │────────>│ 6. audio_source.         │
│    run_coroutine_       │         │    capture_frame()       │
│    threadsafe()         │         │ 7. Queue to LiveKit      │
└─────────────────────────┘         └──────────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────────────┐
                                    │     LiveKit SDK          │
                                    │     (Network I/O)        │
                                    └──────────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────────────┐
                                    │ Remote Audio Processing  │
                                    │ (Async iteration)        │
                                    ├──────────────────────────┤
                                    │ 1. Decode frames         │
                                    │ 2. Extract int16 data    │
                                    │ 3. Call mixer (sync)     │
                                    └───────┬──────────────────┘
                                            │
                                            ▼
┌─────────────────────────┐         ┌──────────────────────────┐
│   OS Audio Thread       │<────────│     AudioMixer           │
│   (Output Callback)     │         │     (Thread-Safe)        │
├─────────────────────────┤         ├──────────────────────────┤
│ 1. Request samples      │────────>│ Uses threading.Lock      │
│ 2. mixer.get_audio_data │         │ Can be called from       │
│ 3. Convert int16→float32│         │ ANY thread               │
│ 4. Write to speakers    │         │                          │
└─────────────────────────┘         └──────────────────────────┘
```

---

## Component Details

### AudioMixer Deep Dive

**Buffer Strategy**: FIFO (First-In-First-Out) with lazy mixing

**Why Lazy Mixing?**
```python
# Without lazy mixing (eager):
mixer.add_audio_data(alice_audio, "alice")  # Mixes immediately
mixer.add_audio_data(bob_audio, "bob")      # Mixes again
# Problem: If Alice has data but Bob doesn't, partial mixing occurs

# With lazy mixing:
mixer.add_audio_data(alice_audio, "alice")  # Just buffers
mixer.add_audio_data(bob_audio, "bob")      # Just buffers
mixer.get_audio_data(2400)                  # Mix when requested
# Benefit: Mixes all available streams at once
```

**Mixing Algorithm**:
```python
def _mix_streams(self):
    # 1. Find minimum buffer size (only mix what ALL streams have)
    min_size = min(len(buf) for buf in self._streams.values() if len(buf) > 0)

    # 2. Average samples from all streams (prevents clipping)
    mixed = np.zeros(min_size, dtype=np.float32)
    for stream_id, buffer in self._streams.items():
        if len(buffer) >= min_size:
            volume = self._stream_volumes[stream_id]
            mixed += buffer[:min_size] * volume

    # 3. Average and apply master volume
    mixed = (mixed / active_streams) * self.master_volume

    # 4. Convert to int16 and clip
    mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
```

**Underrun Handling**:
When requested samples exceed available data, the mixer pads with silence rather than blocking or throwing errors. This prevents audio glitches.

### LiveKitNetworkManager Deep Dive

**Event Flow**:
```
Room Connection
    ↓
Event Handlers Registered
    ├─ participant_connected → _on_participant_connected()
    ├─ participant_disconnected → _on_participant_disconnected()
    ├─ track_subscribed → _on_track_subscribed()
    └─ track_unsubscribed → _on_track_unsubscribed()
        ↓
Track Subscription (if audio)
    ├─ mixer.add_stream(participant_identity)
    └─ asyncio.create_task(_handle_remote_audio_track())
        ↓
Remote Audio Processing Loop
    └─ async for event in AudioStream:
        ├─ Extract frame.data
        ├─ Convert to int16
        └─ mixer.add_audio_data(stream_id=participant_identity)
```

**Task Management**:
Each remote participant gets an async task that processes their audio. Tasks are stored in `_remote_audio_tasks` dict and properly cancelled on disconnect.

---

## Initialization Order

The initialization order is critical due to dependencies between components:

```
1. Store event loop reference
   └─ Needed by: AudioInputHandler (for run_coroutine_threadsafe)

2. Create AudioMixer
   └─ No dependencies

3. Create LiveKitNetworkManager
   └─ Requires: AudioMixer (for routing remote audio)

4. Connect to LiveKit room
   └─ Establishes WebSocket connection

5. Publish audio track
   └─ Creates: AudioSource
   └─ Needed by: AudioInputHandler

6. Create AudioInputHandler
   └─ Requires: AudioSource, event_loop

7. Create AudioOutputHandler
   └─ Requires: AudioMixer

8. Start audio I/O
   └─ AudioInputHandler.start()
   └─ AudioOutputHandler.start()
```

**Code Example**:
```python
async def connect(self):
    # Step 1
    self._event_loop = asyncio.get_event_loop()

    # Step 2
    self.mixer = AudioMixer(...)

    # Step 3-4
    self.network_manager = LiveKitNetworkManager(...)
    await self.network_manager.connect()

    # Step 5
    await self.network_manager.publish_audio_track()

    # Step 6
    self.input_handler = AudioInputHandler(
        audio_source=self.network_manager.audio_source,  # Dependency!
        event_loop=self._event_loop,  # Dependency!
        ...
    )

    # Step 7
    self.output_handler = AudioOutputHandler(
        mixer=self.mixer,  # Dependency!
        ...
    )

    # Step 8
    self.input_handler.start()
    self.output_handler.start()
```

---

## Error Handling

### Graceful Degradation

**AudioMixer Underrun**:
```python
# Instead of raising exception:
if len(self._mixed_buffer) < num_samples:
    self._total_underruns += 1
    result = np.zeros(num_samples, dtype=np.int16)  # Return silence
    return result
```

**Audio Callback Errors**:
```python
def _audio_callback(self, ...):
    try:
        # Process audio
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
        # Log but DON'T raise - audio thread must continue
```

### Cleanup on Partial Initialization

```python
async def connect(self):
    try:
        # ... initialization steps ...
    except Exception as e:
        logger.error(f"Failed: {e}")
        await self._cleanup()  # Clean up what was created
        raise

async def _cleanup(self):
    # Cleanup in reverse order
    if self.output_handler:
        self.output_handler.stop()

    if self.input_handler:
        self.input_handler.stop()

    if self.network_manager:
        await self.network_manager.disconnect()

    # Clear references
    self.mixer = None
    self.network_manager = None
    # ...
```

---

## Testing Strategy

### Unit Tests (99 tests)

Each component is tested in isolation with mocked dependencies:

- **AudioMixer**: 26 tests (mixing, volume, underruns, thread safety)
- **AudioInputHandler**: 19 tests (lifecycle, conversion, error handling)
- **AudioOutputHandler**: 22 tests (lifecycle, conversion, underrun detection)
- **LiveKitNetworkManager**: 19 tests (connection, events, remote audio)
- **AudioClientOrchestrator**: 13 tests (initialization, cleanup, metrics)

### Integration Tests (4 tests)

Test component interactions with mocked LiveKit SDK:

- Orchestrator connects all components
- Context manager lifecycle
- No playback mode
- Metrics aggregation

### Manual Testing

```bash
# List devices
python main.py --list-devices

# Connect to room
python main.py --room-name test --identity alice -v

# Monitor metrics (in code)
metrics = orchestrator.get_metrics()
print(f"Frames received: {metrics['network']['frames_received']}")
print(f"Underruns: {metrics['output']['underruns']}")
```

---

## Performance Characteristics

### Latency

- **Block Size**: 50ms (2400 samples @ 48kHz)
- **Buffering**: 200ms queue in AudioSource
- **Total Latency**: ~250-300ms (typical for real-time audio)

### CPU Usage

- **Mixer**: O(N) where N = number of active streams
- **Format Conversions**: Minimal (simple multiplication/division)
- **Thread Overhead**: Low (callbacks are fast, no blocking)

### Memory Usage

- **Per Stream**: ~2400 samples × 2 bytes = 4.8 KB
- **Mixed Buffer**: ~2400 samples × 2 bytes = 4.8 KB
- **Total**: < 1 MB for typical usage (< 10 participants)

---

## Future Enhancements

Potential improvements that maintain architectural principles:

1. **Jitter Buffer**: Replace simple FIFO with adaptive jitter buffer
2. **Resampling**: Support different sample rates per stream
3. **Recording**: Add recording component that subscribes to mixer
4. **Visualization**: Add spectrum analyzer component
5. **Plugins**: Allow custom audio processing plugins

All enhancements should:
- Maintain separation of concerns
- Preserve thread safety
- Be independently testable
- Not break existing APIs
