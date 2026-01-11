"""Unit tests for AudioMixer.

These tests verify the core functionality of the AudioMixer class:
- Stream lifecycle (add/remove streams)
- Audio mixing (single and multiple streams)
- Volume control (per-stream and master)
- Buffer underrun handling
- Thread safety
- Metrics tracking
"""

import pytest
import numpy as np
import threading
import time
from src.alto_terminal.audio_mixer import AudioMixer, StreamMetrics


class TestAudioMixerInitialization:
    """Test AudioMixer initialization."""

    def test_initialization_default_volume(self):
        """Test mixer initialization with default master volume."""
        mixer = AudioMixer(48000, 1)
        assert mixer.sample_rate == 48000
        assert mixer.channels == 1
        assert mixer.master_volume == 1.0
        assert mixer.buffer_size() == 0

    def test_initialization_custom_volume(self):
        """Test mixer initialization with custom master volume."""
        mixer = AudioMixer(48000, 1, master_volume=0.5)
        assert mixer.master_volume == 0.5


class TestStreamLifecycle:
    """Test stream management (add/remove)."""

    def test_add_stream(self):
        """Test adding a new stream."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice", volume=0.5)

        metrics = mixer.get_metrics("alice")
        assert metrics['stream_id'] == "alice"
        assert metrics['volume'] == 0.5
        assert metrics['buffer_size'] == 0
        assert metrics['samples_added'] == 0

    def test_add_duplicate_stream(self):
        """Test adding duplicate stream resets it."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice", volume=0.5)
        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 100, stream_id="alice")

        # Re-adding should reset
        mixer.add_stream("alice", volume=1.0)
        metrics = mixer.get_metrics("alice")
        assert metrics['volume'] == 1.0
        # Buffer is reset
        assert metrics['buffer_size'] == 0

    def test_remove_stream(self):
        """Test removing a stream."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")
        assert mixer.get_metrics("alice") != {}

        mixer.remove_stream("alice")
        assert mixer.get_metrics("alice") == {}

    def test_auto_create_stream_on_add_audio(self):
        """Test stream auto-creation when adding audio."""
        mixer = AudioMixer(48000, 1)

        # Add audio without explicitly creating stream
        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="bob")

        # Stream should be auto-created
        metrics = mixer.get_metrics("bob")
        assert metrics['stream_id'] == "bob"
        assert metrics['samples_added'] == 1000


class TestSingleStreamAudio:
    """Test audio flow with a single stream."""

    def test_add_and_retrieve_audio(self):
        """Test adding and retrieving audio from single stream."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")

        # Add 1000 samples with value 1000
        audio_in = np.ones(1000, dtype=np.int16) * 1000
        mixer.add_audio_data(audio_in, stream_id="alice")

        # Retrieve 500 samples
        audio_out = mixer.get_audio_data(500)
        assert len(audio_out) == 500
        # Values should match (single stream, no mixing)
        assert np.all(audio_out == 1000)

        # Retrieve remaining 500 samples
        audio_out = mixer.get_audio_data(500)
        assert len(audio_out) == 500
        assert np.all(audio_out == 1000)

        # Buffer should now be empty
        assert mixer.buffer_size() == 0

    def test_multiple_add_operations(self):
        """Test multiple sequential add operations."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")

        # Add audio in chunks
        for i in range(5):
            audio = np.ones(200, dtype=np.int16) * (i + 1) * 100
            mixer.add_audio_data(audio, stream_id="alice")

        # Total should be 1000 samples
        # Retrieve all at once
        audio_out = mixer.get_audio_data(1000)
        assert len(audio_out) == 1000


class TestMultipleStreamMixing:
    """Test mixing multiple streams together."""

    def test_two_stream_mixing(self):
        """Test mixing two streams with averaging."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")
        mixer.add_stream("bob")

        # Alice: 1000 samples at amplitude 1000
        # Bob: 1000 samples at amplitude 2000
        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 1000, stream_id="alice")
        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 2000, stream_id="bob")

        # Mixed output should be average: (1000 + 2000) / 2 = 1500
        audio_out = mixer.get_audio_data(1000)
        assert len(audio_out) == 1000
        # Check values are approximately 1500 (allowing for rounding)
        assert np.allclose(audio_out, 1500, atol=1)

    def test_three_stream_mixing(self):
        """Test mixing three streams."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")
        mixer.add_stream("bob")
        mixer.add_stream("charlie")

        # Three streams with different amplitudes
        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 1000, stream_id="alice")
        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 2000, stream_id="bob")
        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 3000, stream_id="charlie")

        # Mixed output should be average: (1000 + 2000 + 3000) / 3 = 2000
        audio_out = mixer.get_audio_data(1000)
        assert len(audio_out) == 1000
        assert np.allclose(audio_out, 2000, atol=1)

    def test_mixing_with_different_buffer_sizes(self):
        """Test mixing when streams have different amounts of data."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")
        mixer.add_stream("bob")

        # Alice has 500 samples, Bob has 1000 samples
        mixer.add_audio_data(np.ones(500, dtype=np.int16) * 1000, stream_id="alice")
        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 2000, stream_id="bob")

        # Should only mix the first 500 samples (minimum)
        audio_out = mixer.get_audio_data(500)
        assert len(audio_out) == 500
        # First 500 should be mixed from both streams
        assert np.allclose(audio_out, 1500, atol=1)

        # Next 500 samples should only have Bob's data (Alice is empty)
        audio_out = mixer.get_audio_data(500)
        assert len(audio_out) == 500
        # Only Bob's stream remains
        assert np.allclose(audio_out, 2000, atol=1)


class TestVolumeControl:
    """Test volume control (per-stream and master)."""

    def test_per_stream_volume(self):
        """Test per-stream volume control."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice", volume=0.5)

        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 1000, stream_id="alice")
        audio_out = mixer.get_audio_data(1000)

        # Volume 0.5 should halve amplitude
        assert np.allclose(audio_out, 500, atol=1)

    def test_set_stream_volume(self):
        """Test changing stream volume at runtime."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice", volume=1.0)

        # Change volume to 0.25
        mixer.set_stream_volume("alice", 0.25)

        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 1000, stream_id="alice")
        audio_out = mixer.get_audio_data(1000)

        # Volume 0.25 should quarter amplitude
        assert np.allclose(audio_out, 250, atol=1)

    def test_master_volume(self):
        """Test master volume affects all streams."""
        mixer = AudioMixer(48000, 1, master_volume=0.5)
        mixer.add_stream("alice")
        mixer.add_stream("bob")

        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 1000, stream_id="alice")
        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 1000, stream_id="bob")

        # Both streams at 1000, averaged to 1000, then master volume 0.5 = 500
        audio_out = mixer.get_audio_data(1000)
        assert np.allclose(audio_out, 500, atol=1)

    def test_combined_volumes(self):
        """Test per-stream volume combined with master volume."""
        mixer = AudioMixer(48000, 1, master_volume=0.5)
        mixer.add_stream("alice", volume=0.5)

        mixer.add_audio_data(np.ones(1000, dtype=np.int16) * 1000, stream_id="alice")
        audio_out = mixer.get_audio_data(1000)

        # Stream volume 0.5 * master volume 0.5 = 0.25 of original
        assert np.allclose(audio_out, 250, atol=1)


class TestBufferUnderrun:
    """Test buffer underrun handling."""

    def test_underrun_returns_silence(self):
        """Test underrun returns zeros (silence)."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")

        # Request audio when buffer is empty
        audio_out = mixer.get_audio_data(1000)

        # Should return silence
        assert len(audio_out) == 1000
        assert np.all(audio_out == 0)

    def test_partial_underrun(self):
        """Test partial underrun pads with silence."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")

        # Add 500 samples but request 1000
        mixer.add_audio_data(np.ones(500, dtype=np.int16) * 1000, stream_id="alice")
        audio_out = mixer.get_audio_data(1000)

        # Should return 1000 samples (500 real + 500 silence)
        assert len(audio_out) == 1000
        assert np.all(audio_out[:500] == 1000)
        assert np.all(audio_out[500:] == 0)

    def test_underrun_counter(self):
        """Test underrun counter increments."""
        mixer = AudioMixer(48000, 1)

        # Cause multiple underruns
        for _ in range(5):
            mixer.get_audio_data(1000)

        metrics = mixer.get_metrics()
        assert metrics['total_underruns'] == 5


class TestThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_add_and_get(self):
        """Test concurrent add/get operations from multiple threads."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")

        errors = []

        def add_audio():
            """Add audio repeatedly."""
            try:
                for _ in range(100):
                    mixer.add_audio_data(np.ones(100, dtype=np.int16), stream_id="alice")
                    time.sleep(0.001)  # Small delay to interleave operations
            except Exception as e:
                errors.append(e)

        def get_audio():
            """Get audio repeatedly."""
            try:
                for _ in range(100):
                    mixer.get_audio_data(100)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Run concurrent add/get operations
        threads = [
            threading.Thread(target=add_audio),
            threading.Thread(target=get_audio)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0

    def test_concurrent_multiple_streams(self):
        """Test concurrent operations on multiple streams."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")
        mixer.add_stream("bob")

        errors = []

        def add_to_stream(stream_id):
            """Add audio to specific stream."""
            try:
                for _ in range(50):
                    mixer.add_audio_data(np.ones(100, dtype=np.int16), stream_id=stream_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Multiple threads adding to different streams
        threads = [
            threading.Thread(target=add_to_stream, args=("alice",)),
            threading.Thread(target=add_to_stream, args=("bob",)),
            threading.Thread(target=lambda: [mixer.get_audio_data(100) for _ in range(50)])
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestMetrics:
    """Test metrics tracking."""

    def test_samples_added_metric(self):
        """Test samples_added metric is tracked correctly."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")

        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="alice")
        mixer.add_audio_data(np.ones(500, dtype=np.int16), stream_id="alice")

        metrics = mixer.get_metrics("alice")
        assert metrics['samples_added'] == 1500

    def test_buffer_size_metric(self):
        """Test buffer_size metric reflects current buffer."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")

        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="alice")
        metrics = mixer.get_metrics("alice")
        assert metrics['buffer_size'] > 0

        # Consume some audio
        mixer.get_audio_data(500)
        metrics_after = mixer.get_metrics("alice")
        assert metrics_after['buffer_size'] < metrics['buffer_size']

    def test_global_metrics(self):
        """Test global metrics aggregation."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")
        mixer.add_stream("bob")

        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="alice")
        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="bob")

        global_metrics = mixer.get_metrics()
        assert global_metrics['active_streams'] == 2
        assert global_metrics['master_volume'] == 1.0
        assert 'stream_metrics' in global_metrics
        assert 'alice' in global_metrics['stream_metrics']
        assert 'bob' in global_metrics['stream_metrics']


class TestBufferManagement:
    """Test buffer operations."""

    def test_clear_all_buffers(self):
        """Test clearing all buffers."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")
        mixer.add_stream("bob")

        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="alice")
        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="bob")

        # Clear all
        mixer.clear()

        assert mixer.buffer_size() == 0
        assert mixer.get_metrics("alice")['buffer_size'] == 0
        assert mixer.get_metrics("bob")['buffer_size'] == 0

    def test_clear_specific_stream(self):
        """Test clearing specific stream buffer."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")
        mixer.add_stream("bob")

        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="alice")
        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="bob")

        # Clear only alice
        mixer.clear(stream_id="alice")

        assert mixer.get_metrics("alice")['buffer_size'] == 0
        assert mixer.get_metrics("bob")['buffer_size'] > 0

    def test_buffer_size_method(self):
        """Test buffer_size() method."""
        mixer = AudioMixer(48000, 1)
        mixer.add_stream("alice")

        assert mixer.buffer_size() == 0

        # Add audio data - mixing happens lazily when audio is requested
        mixer.add_audio_data(np.ones(1000, dtype=np.int16), stream_id="alice")
        assert mixer.buffer_size() == 0  # Not mixed yet

        # Request audio triggers mixing
        mixer.get_audio_data(500)
        # After getting 500, we have 500 left in the mixed buffer
        assert mixer.buffer_size() == 500

        # Get the remaining audio
        mixer.get_audio_data(500)
        assert mixer.buffer_size() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
