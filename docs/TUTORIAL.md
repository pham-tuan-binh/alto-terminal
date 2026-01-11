# Tutorial: Understanding Audio Flow

This tutorial walks through how audio flows through the Alto Terminal system, from your microphone to remote speakers and back.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Starting the Client](#starting-the-client)
3. [Microphone to LiveKit](#microphone-to-livekit)
4. [LiveKit to Speakers](#livekit-to-speakers)
5. [Understanding the Code](#understanding-the-code)
6. [Common Scenarios](#common-scenarios)

---

## Prerequisites

Before starting, understand these key concepts:

### Audio Formats

- **Sample Rate**: 48000 Hz (48 kHz) - how many samples per second
- **Channels**: 1 (mono audio)
- **Block Size**: 2400 samples (50ms @ 48 kHz)
- **Format**: int16 (-32768 to 32767) or float32 (-1.0 to 1.0)

### Threading Model

- **OS Audio Thread**: High-priority thread for audio I/O (can't block!)
- **Async Event Loop**: Thread for LiveKit network operations
- **Bridge**: `asyncio.run_coroutine_threadsafe()` connects the two

---

## Starting the Client

```bash
python main.py --room-name demo --identity alice
```

### What Happens (Step by Step)

#### 1. Parse Arguments and Create Config
```python
config = AudioConfig(
    sample_rate=48000,
    num_channels=1,
    samples_per_channel=2400,  # 50ms
    input_device=None,  # Use default
    output_device=None,
    volume=1.0,
    no_playback=False
)
```

#### 2. Create Orchestrator
```python
orchestrator = AudioClientOrchestrator(
    url="wss://livekit.example.com",
    token="<jwt-token>",
    config=config
)
```

#### 3. Connect (Initializes All Components)
```python
await orchestrator.connect()
```

This single call does:
1. Stores event loop reference
2. Creates AudioMixer
3. Connects to LiveKit room
4. Publishes audio track
5. Creates AudioInputHandler
6. Creates AudioOutputHandler
7. Starts audio I/O

---

## Microphone to LiveKit

Let's trace one 50ms block of audio from your microphone to the network.

### Step 1: Microphone Captures Audio

Your microphone captures sound and sends it to sounddevice:

```
Microphone → sounddevice → OS Audio Driver
```

### Step 2: AudioInputHandler Callback (OS Audio Thread)

```python
def _audio_callback(self, indata, frames, time_info, status):
    # indata: shape (2400, 1), dtype float32, range [-1.0, 1.0]

    # Extract mono channel
    audio_data = indata[:, 0]  # shape (2400,)
```

**Key Point**: This runs in an OS audio thread. Must be FAST!

### Step 3: Convert float32 → int16

```python
    # Multiply by 32767 to convert to int16 range
    audio_data = (audio_data * 32767).astype(np.int16)
    # Now: shape (2400,), dtype int16, range [-32768, 32767]
```

**Why multiply by 32767 and not 32768?**
- float32 range: [-1.0, 1.0]
- int16 range: [-32768, 32767]
- -1.0 × 32768 = -32768 (fine)
- 1.0 × 32768 = 32768 (OVERFLOW! max is 32767)
- Solution: multiply by 32767

### Step 4: Create AudioFrame

```python
    frame = rtc.AudioFrame(
        data=audio_data.tobytes(),  # Convert to bytes
        sample_rate=48000,
        num_channels=1,
        samples_per_channel=2400
    )
```

### Step 5: Bridge to Async World

```python
    # Schedule coroutine to run in event loop (non-blocking!)
    asyncio.run_coroutine_threadsafe(
        self.audio_source.capture_frame(frame),
        self.event_loop
    )
    # Returns immediately, doesn't wait for result
```

**Key Point**: This bridges OS audio thread → async event loop without blocking.

### Step 6: LiveKit SDK Sends Audio (Async Event Loop)

```python
async def capture_frame(self, frame):
    # Runs in async event loop
    # 1. Queue frame in internal buffer (200ms queue)
    # 2. Encode with Opus codec
    # 3. Send over WebSocket
    # 4. Handle network buffering
```

### Complete Flow Diagram

```
┌─────────────┐
│ Microphone  │
│  (analog)   │
└──────┬──────┘
       │ analog signal
       ▼
┌──────────────────────┐
│  Audio Interface     │
│  (ADC)               │
└──────┬───────────────┘
       │ PCM samples
       ▼
┌──────────────────────────────────────┐
│  OS Audio Thread                     │
│  (sounddevice callback)              │
├──────────────────────────────────────┤
│  Input: float32 × 2400 samples       │
│    │                                 │
│    ├─> Multiply by 32767             │
│    │                                 │
│    ├─> Convert to int16              │
│    │                                 │
│    ├─> Create AudioFrame             │
│    │                                 │
│    └─> run_coroutine_threadsafe()  ──┼──┐
└──────────────────────────────────────┘  │
                                          │
       ┌──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Async Event Loop Thread             │
│  (AudioSource.capture_frame)         │
├──────────────────────────────────────┤
│  Queue frame (200ms buffer)          │
│    │                                 │
│    ├─> Encode with Opus              │
│    │                                 │
│    └─> Send via WebSocket            │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Network              │
│  (WebSocket/RTP)      │
└───────────────────────┘
```

---

## LiveKit to Speakers

Now let's trace audio coming FROM the network TO your speakers.

### Step 1: LiveKit SDK Receives Audio (Async Event Loop)

```python
# In LiveKitNetworkManager._handle_remote_audio_track:
async for event in stream:
    frame = event.frame  # Decoded Opus → PCM
```

### Step 2: Extract int16 Audio Data

```python
    # frame.data is bytes
    audio_data = np.frombuffer(frame.data, dtype=np.int16)
    # shape: (2400,), dtype: int16
```

### Step 3: Route to Mixer

```python
    # Add to mixer with participant's identity as stream_id
    self.mixer.add_audio_data(
        audio_data,
        stream_id=participant_identity  # e.g., "bob"
    )
```

**Key Point**: Each participant gets their own stream in the mixer.

### Step 4: AudioMixer Buffers Data

```python
def add_audio_data(self, audio_data, stream_id="default"):
    with self._lock:  # Thread-safe
        # Just buffer, don't mix yet (lazy mixing)
        if stream_id not in self._streams:
            self._streams[stream_id] = np.array([], dtype=np.int16)

        self._streams[stream_id] = np.concatenate([
            self._streams[stream_id],
            audio_data
        ])
```

### Step 5: AudioOutputHandler Requests Audio (OS Audio Thread)

```python
def _audio_callback(self, outdata, frames, time_info, status):
    # Runs in OS audio thread
    # Request 2400 samples
    audio_data = self.mixer.get_audio_data(2400)
```

### Step 6: Mixer Performs Lazy Mixing

```python
def get_audio_data(self, num_samples):
    with self._lock:
        # Mix only if needed
        if len(self._mixed_buffer) < num_samples:
            self._mix_streams()  # Mix NOW

        # Return requested samples
        return self._mixed_buffer[:num_samples]

def _mix_streams(self):
    # Find minimum buffer size across all streams
    min_size = min(len(buf) for buf in self._streams.values())

    # Mix by averaging (prevents clipping)
    mixed = np.zeros(min_size, dtype=np.float32)
    for stream_id, buffer in self._streams.items():
        volume = self._stream_volumes[stream_id]
        mixed += buffer[:min_size] * volume

    # Average and apply master volume
    mixed = (mixed / active_streams) * self.master_volume

    # Clip and convert to int16
    mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
```

### Step 7: Convert int16 → float32

```python
    # Divide by 32767 to convert to float32 range
    outdata[:, 0] = audio_data.astype(np.float32) / 32767.0
    # Now: shape (2400,), dtype float32, range [-1.0, 1.0]
```

### Step 8: sounddevice Plays Audio

```
float32 samples → OS Audio Driver → DAC → Speakers
```

### Complete Flow Diagram

```
┌──────────────────────┐
│  Network             │
│  (WebSocket/RTP)     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Async Event Loop Thread             │
│  (Remote Audio Processing)           │
├──────────────────────────────────────┤
│  Receive & decode frame              │
│    │                                 │
│    ├─> Extract int16 data            │
│    │                                 │
│    └─> mixer.add_audio_data()  ──────┼──┐
└──────────────────────────────────────┘  │
                                          │
       ┌──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  AudioMixer (Thread-Safe)            │
│  (Can be called from any thread)     │
├──────────────────────────────────────┤
│  Per-Stream Buffers:                 │
│    - "bob": [int16 samples]          │
│    - "alice": [int16 samples]        │
│                                      │
│  get_audio_data() triggers mixing:   │
│    1. Find min buffer size           │
│    2. Average all streams            │
│    3. Apply volumes                  │
│    4. Return mixed int16             │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  OS Audio Thread                     │
│  (sounddevice callback)              │
├──────────────────────────────────────┤
│  Request samples from mixer          │
│    │                                 │
│    ├─> Divide by 32767               │
│    │                                 │
│    ├─> Convert to float32            │
│    │                                 │
│    └─> Write to output buffer        │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Audio Interface     │
│  (DAC)               │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Speakers            │
│  (analog)            │
└──────────────────────┘
```

---

## Understanding the Code

### Finding Key Code Sections

#### AudioInputHandler Callback
**File**: `src/alto_terminal/audio_input_handler.py:141`

```python
def _audio_callback(self, indata, frames, time_info, status):
    # This is where microphone audio enters the system
```

#### AudioOutputHandler Callback
**File**: `src/alto_terminal/audio_output_handler.py:138`

```python
def _audio_callback(self, outdata, frames, time_info, status):
    # This is where mixed audio goes to speakers
```

#### Mixer Lazy Mixing
**File**: `src/alto_terminal/audio_mixer.py:160`

```python
def _mix_streams(self):
    # This is where multiple audio streams are combined
```

#### Remote Audio Processing
**File**: `src/alto_terminal/livekit_network_manager.py:387`

```python
async def _handle_remote_audio_track(self, track, participant_identity):
    # This is where remote audio enters the mixer
```

---

## Common Scenarios

### Scenario 1: Two People Talking (Alice and Bob)

1. **Alice speaks**: Microphone → AudioInputHandler → LiveKit → Bob's speakers
2. **Bob speaks**: Microphone → AudioInputHandler → LiveKit → Alice's speakers
3. **Alice's mixer**: Contains one stream ("bob")
4. **Bob's mixer**: Contains one stream ("alice")

### Scenario 2: Conference Call (Alice, Bob, Charlie)

1. **Alice's mixer** contains two streams:
   - "bob": Bob's audio
   - "charlie": Charlie's audio
   - Mixer averages both: `(bob + charlie) / 2`

2. **Bob's mixer** contains two streams:
   - "alice": Alice's audio
   - "charlie": Charlie's audio

3. **Each person** hears everyone else mixed together

### Scenario 3: Adjusting Volume

```python
# Alice wants to hear Bob louder and Charlie quieter
orchestrator.set_stream_volume("bob", 1.5)
orchestrator.set_stream_volume("charlie", 0.5)

# In mixer:
# mixed = (bob_audio * 1.5 + charlie_audio * 0.5) / 2
```

### Scenario 4: Buffer Underrun

**What happens**:
1. Network is slow, Bob's audio doesn't arrive in time
2. Mixer has no data for Bob's stream
3. Only Alice's stream has data
4. Mixer returns Alice's audio only (no mixing needed)
5. If ALL streams are empty: Mixer returns silence

**Code**:
```python
# In mixer.get_audio_data():
if len(self._mixed_buffer) < num_samples:
    # Underrun!
    self._total_underruns += 1
    return np.zeros(num_samples, dtype=np.int16)  # Silence
```

---

## Next Steps

### Experiment with the Code

1. **Add Logging**:
```python
# In mixer._mix_streams():
print(f"Mixing {len(self._streams)} streams, min_size={min_size}")
```

2. **Monitor Metrics**:
```python
# In main.py:
while True:
    await asyncio.sleep(5)
    metrics = orchestrator.get_metrics()
    print(f"Frames received: {metrics['network']['frames_received']}")
    print(f"Underruns: {metrics['output']['underruns']}")
```

3. **Adjust Buffer Sizes**:
```python
# In config.py:
SAMPLES_PER_CHANNEL = SAMPLE_RATE // 50  # 20ms (experimental)
```

### Read More

- **ARCHITECTURE.md**: Complete component documentation
- **DESIGN_DECISIONS.md**: Why things are built this way
- **Unit Tests**: `tests/unit/` - see how components are tested
- **Integration Tests**: `tests/integration/` - see how components work together

### Build Your Own Feature

Try adding a new feature, like recording:

```python
class AudioRecorder:
    def __init__(self, mixer):
        self.mixer = mixer
        self.recording = []

    def record_frame(self):
        # Pull from mixer
        audio = self.mixer.get_audio_data(2400)
        self.recording.append(audio)

    def save_to_file(self, filename):
        # Concatenate and save
        data = np.concatenate(self.recording)
        # ... write to WAV file ...
```

Happy coding!
