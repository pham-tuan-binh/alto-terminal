"""Utility functions for audio device management.

This module provides helper functions for working with audio devices,
including listing available devices and querying device information.
"""

import sounddevice as sd


def list_audio_devices():
    """List all available audio input and output devices.

    This function prints a formatted list of all audio devices,
    showing their index, name, sample rate, and channel count.
    Also displays the default input and output devices.
    """
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
