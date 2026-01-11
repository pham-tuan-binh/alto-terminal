# Design Decisions

This document explains the key design decisions made in the Alto Terminal audio client, including the rationale, alternatives considered, and trade-offs.

## Table of Contents

1. [Why 5 Components?](#why-5-components)
2. [Why threading.Lock Instead of asyncio.Lock?](#why-threadinglock-instead-of-asynciolock)
3. [Why run_coroutine_threadsafe()?](#why-run_coroutine_threadsafe)
4. [Why int16 Internal Format?](#why-int16-internal-format)
5. [Why 50ms Block Size?](#why-50ms-block-size)
6. [Why Per-Stream Tracking?](#why-per-stream-tracking)
7. [Why Lazy Mixing?](#why-lazy-mixing)
8. [Why Graceful Underrun Handling?](#why-graceful-underrun-handling)
9. [Why Simple FIFO Buffer?](#why-simple-fifo-buffer)
10. [Why No Muting/Unmuting in Orchestrator?](#why-no-mutingunmuting-in-orchestrator)

---

## Why 5 Components?

### Decision

Split the monolithic `LiveKitAudioClient` into 5 focused components:
1. AudioMixer
2. AudioInputHandler
3. AudioOutputHandler
4. LiveKitNetworkManager
5. AudioClientOrchestrator

### Rationale

**Single Responsibility**: Each component has one clear purpose:
- Mixer: Mix audio streams
- Input: Capture microphone
- Output: Play to speakers
- Network: Manage LiveKit connection
- Orchestrator: Coordinate lifecycle

**Testability**: Easier to test each component in isolation with mocked dependencies. The original 515-line god class was difficult to test.

**Maintainability**: Changes to one aspect (e.g., audio mixing algorithm) don't affect unrelated code (e.g., network management).

**Reusability**: Components can be used independently. For example, AudioMixer could be used in a recording application without LiveKit.

### Alternatives Considered

**3 Components** (Mixer, I/O Handler, Network Manager):
- Pros: Fewer files, simpler structure
- Cons: I/O handler would handle both input and output, violating SRP

**7+ Components** (Separate converters, event handlers, etc.):
- Pros: Even more focused responsibilities
- Cons: Over-engineering, too many files for a relatively simple application

### Trade-offs

- **More Files**: 5 components mean more files to navigate
- **Initialization Order**: Dependencies between components require careful initialization (handled by orchestrator)
- **Slight Performance Overhead**: Function call boundaries between components (negligible in practice)

---

## Why threading.Lock Instead of asyncio.Lock?

### Decision

Use `threading.Lock` in AudioMixer, not `asyncio.Lock`.

```python
# In AudioMixer.__init__:
self._lock = threading.Lock()  # NOT asyncio.Lock()
```

### Rationale

**Audio Callbacks Run in OS Threads**: sounddevice audio callbacks execute in OS-managed threads, not in the asyncio event loop.

**asyncio.Lock Requires Async Context**: You can't use `asyncio.Lock` from synchronous code:
```python
# This DOESN'T work:
async def add_audio_data(self, audio_data):
    async with self._lock:  # Requires async context
        # ...

# Called from OS audio thread (synchronous):
def _audio_callback(self, outdata, frames, ...):
    audio_data = self.mixer.get_audio_data(frames)  # Can't await!
```

**threading.Lock Works Everywhere**: Can be used from both OS threads and async event loop.

### Alternatives Considered

**Make All Mixer Methods Async**:
```python
async def get_audio_data(self, num_samples):
    async with self._lock:
        # ...

# In audio callback:
def _audio_callback(self, outdata, frames, ...):
    asyncio.run_coroutine_threadsafe(
        self.mixer.get_audio_data(frames),
        self.event_loop
    ).result()  # Wait for result (BLOCKS!)
```

Problems:
- `.result()` blocks the audio thread (causes glitches)
- Adds latency (30-50ms for event loop roundtrip)
- Unnecessarily complex

**Lock-Free Design with Atomic Operations**:
- Pros: Better performance, no lock contention
- Cons: Much more complex, harder to get right, not worth it for this use case

### Trade-offs

- **Can't Use Async Features in Mixer**: threading.Lock means mixer methods must be synchronous
- **Slightly Less "Pythonic"**: mixing threading and asyncio patterns

---

## Why run_coroutine_threadsafe()?

### Decision

Use `asyncio.run_coroutine_threadsafe()` to bridge from OS audio thread to async event loop:

```python
def _audio_callback(self, indata, frames, ...):
    # Running in OS audio thread
    audio_data = (indata[:, 0] * 32767).astype(np.int16)
    frame = rtc.AudioFrame(...)

    # Bridge to async world
    asyncio.run_coroutine_threadsafe(
        self.audio_source.capture_frame(frame),
        self.event_loop
    )
```

### Rationale

**LiveKit SDK is Async**: `audio_source.capture_frame()` is an async method that must run in the event loop.

**Audio Callback is Synchronous**: OS audio thread is synchronous, can't use `await`.

**Non-Blocking**: `run_coroutine_threadsafe()` schedules the coroutine and returns immediately, doesn't block the audio thread.

### Alternatives Considered

**Block and Wait for Result**:
```python
future = asyncio.run_coroutine_threadsafe(...)
future.result(timeout=0.01)  # Wait up to 10ms
```

Problems:
- Blocks audio thread (causes glitches)
- Timeout adds complexity (what if it times out?)

**Queue-Based Approach**:
```python
# In audio callback:
self.frame_queue.put(frame)

# In separate thread:
while True:
    frame = self.frame_queue.get()
    asyncio.run(self.audio_source.capture_frame(frame))
```

Problems:
- More complex (extra thread, queue management)
- Same end result (still need to bridge to event loop)

### Trade-offs

- **Must Store Event Loop Reference**: Need to capture `asyncio.get_event_loop()` during initialization
- **Fire and Forget**: Can't easily get result or handle errors from the coroutine

---

## Why int16 Internal Format?

### Decision

Use int16 as the internal audio format in AudioMixer.

### Rationale

**LiveKit Native Format**: LiveKit SDK expects int16 PCM audio:
```python
frame = rtc.AudioFrame(
    data=audio_data.tobytes(),  # int16 bytes
    sample_rate=48000,
    num_channels=1,
    samples_per_channel=len(audio_data)
)
```

**Minimize Conversions**:
- Input: float32 (sounddevice) → int16 (mixer) — 1 conversion
- Mixer: int16 → int16 — no conversion
- Output: int16 (mixer) → float32 (sounddevice) — 1 conversion

vs. using float32 internally:
- Input: float32 → float32 — no conversion
- Mixer: float32 → int16 (for LiveKit) — 1 conversion
- LiveKit: int16 → float32 (received) — 1 conversion
- Mixer: float32 → float32 — no conversion
- Output: float32 → float32 — no conversion

Total: **2 conversions** vs. **3 conversions** with float32 internal

**Efficient Storage**: int16 is 2 bytes vs. 4 bytes for float32 (50% memory savings).

### Alternatives Considered

**Use float32 Internally**:
- Pros: No conversion for sounddevice, better precision for mixing
- Cons: More conversions with LiveKit, more memory usage

**Use float64**:
- Pros: Maximum precision
- Cons: 4× memory usage, unnecessary precision for audio

### Trade-offs

- **Limited Precision**: int16 has 16-bit precision (~96 dB dynamic range), sufficient for most audio but not audiophile-grade
- **Clipping Concerns**: Must carefully clip values when mixing to avoid overflow

---

## Why 50ms Block Size?

### Decision

Use 50ms audio blocks (2400 samples @ 48kHz):

```python
SAMPLES_PER_CHANNEL = SAMPLE_RATE // 20  # 50ms
```

### Rationale

**Balance Latency and CPU**:
- Smaller blocks (10ms): Lower latency but higher CPU (5× more callbacks/sec)
- Larger blocks (100ms): Lower CPU but higher latency

**50ms is Standard**: Common in VoIP and real-time audio applications (Opus codec default is 20-60ms).

**Acceptable Latency**: 50ms is perceptually fast enough for interactive audio conversations.

### Alternatives Considered

**10ms Blocks**:
- Pros: Lower latency (~260ms total vs ~300ms)
- Cons: 5× more audio callbacks, higher CPU usage, more thread context switching

**100ms Blocks**:
- Pros: Lower CPU usage (half the callbacks)
- Cons: Noticeable latency (~350ms total), choppier feel

### Trade-offs

- **Fixed Block Size**: All audio processing uses same block size, can't optimize per component
- **Latency**: 50ms per block adds to overall latency

### Measurement

```python
# CPU usage (approximate, single stream):
# 10ms: ~2-3% CPU
# 50ms: ~1-2% CPU
# 100ms: ~0.5-1% CPU
```

---

## Why Per-Stream Tracking?

### Decision

Track each remote participant as a separate stream in AudioMixer:

```python
mixer.add_stream("alice", volume=1.0)
mixer.add_audio_data(audio_data, stream_id="alice")
mixer.set_stream_volume("alice", 0.5)
```

### Rationale

**Independent Volume Control**: Can adjust volume per participant:
```python
# Quiet loud participant:
mixer.set_stream_volume("loud_alice", 0.3)

# Boost quiet participant:
mixer.set_stream_volume("quiet_bob", 1.5)
```

**Individual Metrics**: Track frames, underruns, buffer size per participant for debugging.

**Future Features**: Enables:
- Mute specific participants
- Record individual participants
- Spatial audio (pan participants left/right)

### Alternatives Considered

**Single Mixed Stream**:
```python
mixer.add_audio_data(audio_data)  # No stream_id
```

- Pros: Simpler implementation
- Cons: No per-participant control, all-or-nothing volume

**External Mixing**:
Mix audio before sending to mixer:
```python
# In network manager:
total_audio = alice_audio + bob_audio
mixer.add_audio_data(total_audio)
```

- Pros: Simpler mixer
- Cons: Loses per-participant information, can't adjust volumes independently

### Trade-offs

- **Memory**: Each stream needs its own buffer (~5 KB per participant)
- **Complexity**: More bookkeeping (stream IDs, volumes, metrics)

---

## Why Lazy Mixing?

### Decision

Mix audio only when `get_audio_data()` is called, not when audio is added:

```python
def add_audio_data(self, audio_data, stream_id):
    with self._lock:
        # Just buffer, don't mix yet
        self._streams[stream_id] = np.concatenate([...])

def get_audio_data(self, num_samples):
    with self._lock:
        if len(self._mixed_buffer) < num_samples:
            self._mix_streams()  # Mix now
        return self._mixed_buffer[:num_samples]
```

### Rationale

**Prevents Partial Mixing**:

Without lazy mixing (eager):
```python
# Alice sends audio first
mixer.add_audio_data(alice_audio, "alice")  # Mixes immediately (only Alice)
# 10ms later, Bob sends audio
mixer.add_audio_data(bob_audio, "bob")      # Mixes again (only Bob)
# Result: First part has only Alice, second part has only Bob
```

With lazy mixing:
```python
mixer.add_audio_data(alice_audio, "alice")  # Just buffers
mixer.add_audio_data(bob_audio, "bob")      # Just buffers
mixer.get_audio_data(2400)                  # Mixes all available streams
# Result: Properly mixed audio with both Alice and Bob
```

**Efficiency**: Mix once when needed, not multiple times as audio arrives.

### Alternatives Considered

**Eager Mixing with Synchronization**:
Wait for all streams before mixing:
```python
def add_audio_data(self, ...):
    self._streams[stream_id] = ...

    # Wait for all streams to have data
    if all(len(buf) >= THRESHOLD for buf in self._streams.values()):
        self._mix_streams()
```

- Pros: Mixes when all data available
- Cons: Adds latency (waiting for slowest stream), complex logic

**Timed Mixing**:
Mix every N milliseconds:
```python
async def _mix_periodically(self):
    while True:
        await asyncio.sleep(0.05)  # Every 50ms
        self._mix_streams()
```

- Pros: Regular mixing schedule
- Cons: Adds thread, timing complexity, might mix when not needed

### Trade-offs

- **Must Pull to Trigger Mixing**: No audio output means no mixing (not a problem in practice)
- **Buffer Size Varies**: Mixed buffer size depends on how often `get_audio_data()` is called

---

## Why Graceful Underrun Handling?

### Decision

When the mixer runs out of audio, return silence instead of raising an error:

```python
if len(self._mixed_buffer) < num_samples:
    self._total_underruns += 1
    logger.warning(f"Underrun #{self._total_underruns}")

    # Return silence
    result = np.zeros(num_samples, dtype=np.int16)
    if len(self._mixed_buffer) > 0:
        result[:len(self._mixed_buffer)] = self._mixed_buffer
    return result
```

### Rationale

**Audio Must Continue**: Audio callbacks MUST return data. Raising an exception causes:
- sounddevice to abort
- Audio stream to crash
- Restart required

**Silence is Better Than Glitches**:
- Silence: User notices gap, but audio continues
- Error: Pops, clicks, stream crash

**Common in Real-Time Audio**: Standard practice in audio systems (JACK, PulseAudio, etc.).

### Alternatives Considered

**Repeat Last Frame**:
```python
if underrun:
    return self._last_frame  # Repeat
```

- Pros: Smoother than silence
- Cons: Can sound robotic, more complex

**Raise Exception**:
```python
if underrun:
    raise BufferUnderrunError()
```

- Pros: Forces caller to handle issue
- Cons: Crashes audio stream

**Block Until Data Available**:
```python
while len(self._mixed_buffer) < num_samples:
    time.sleep(0.001)  # Wait for data
```

- Pros: Never returns incomplete data
- Cons: Blocks audio thread (catastrophic - causes glitches)

### Trade-offs

- **Silent Gaps**: Users may notice brief silences during network issues
- **Masks Problems**: Underruns are logged but system continues, might hide issues

---

## Why Simple FIFO Buffer?

### Decision

Use a simple First-In-First-Out queue with numpy arrays:

```python
self._mixed_buffer = np.concatenate([self._mixed_buffer, mixed])
```

### Rationale

**Sufficient for Interactive Audio**: FIFO with graceful underruns handles typical network jitter.

**Simple to Understand**: Easy to reason about, easy to debug.

**Low Latency**: No artificial delays, audio plays as soon as available.

### Alternatives Considered

**Jitter Buffer** (adaptive buffer):
```python
# Maintain target buffer size
target_size = SAMPLE_RATE * 0.100  # 100ms

if buffer_size < target_size * 0.8:
    # Too empty, speed up playback slightly
    playback_rate = 1.01
elif buffer_size > target_size * 1.2:
    # Too full, slow down playback
    playback_rate = 0.99
```

- Pros: Handles variable network latency better, fewer underruns
- Cons: More complex, adds latency, requires resampling

**Circular Buffer** (fixed size):
```python
self._buffer = np.zeros(MAX_SIZE, dtype=np.int16)
self._write_pos = 0
self._read_pos = 0
```

- Pros: Fixed memory, no allocations
- Cons: Arbitrary size limit, overruns lose data

### Trade-offs

- **Underruns Possible**: Network hiccups cause silence gaps
- **Memory Grows**: Buffer can grow if audio arrives faster than consumed (rare in practice)

### When to Upgrade

Consider jitter buffer if:
- Underrun rate > 1% of frames
- Network latency is highly variable
- Users report frequent audio gaps

---

## Why No Muting/Unmuting in Orchestrator?

### Decision

Don't expose track muting/unmuting in `AudioClientOrchestrator` API.

### Rationale

**Out of Scope**: Orchestrator manages component lifecycle, not runtime audio control.

**Track Reference Not Exposed**: `publish_audio_track()` returns track to network manager, not stored in orchestrator.

**Use LiveKit API Directly**: If needed, get track from room:
```python
# In main.py or application code:
async with orchestrator:
    track = orchestrator.network_manager.room.local_participant.tracks[...]
    track.mute()
```

### Alternatives Considered

**Store Track Reference**:
```python
class AudioClientOrchestrator:
    def __init__(self, ...):
        self.published_track = None

    async def connect(self):
        # ...
        self.published_track = await self.network_manager.publish_audio_track()

    def mute(self):
        if self.published_track:
            self.published_track.mute()
```

- Pros: Convenient API
- Cons: Adds complexity, not core to orchestrator's purpose

### Trade-offs

- **Less Convenient**: Muting requires accessing network_manager internals
- **Cleaner Separation**: Orchestrator stays focused on lifecycle

---

## Summary of Key Principles

These design decisions follow several core principles:

1. **Simplicity First**: Choose simple solutions unless complexity is justified
2. **Fail Gracefully**: Degrade rather than crash (silence on underrun, log on error)
3. **Thread Safety**: Use appropriate primitives (threading.Lock, run_coroutine_threadsafe)
4. **Performance**: Optimize hot paths (lazy mixing, minimal conversions)
5. **Maintainability**: Clear separation, good abstractions, comprehensive tests

When making future changes, consider these principles and document your decisions!
