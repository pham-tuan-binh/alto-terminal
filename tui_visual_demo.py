#!/usr/bin/env python3
"""Standalone test for TUI visualization without LiveKit connection.

This tests the TUI display with synthetic data to verify:
1. Rich rendering works
2. 3-panel layout displays correctly
3. Audio visualization updates
4. Transcript display works
"""

import asyncio
import numpy as np
import time
from src.alto_terminal.tui_manager import TUIManager


async def test_tui():
    """Test TUI with synthetic data."""

    # Create TUI manager
    tui = TUIManager(max_transcript_lines=20, max_audio_history=50)

    print("Starting TUI test...")
    print("You should see 3 panels:")
    print("  - Top: Conversation transcripts")
    print("  - Bottom-left: User audio visualization")
    print("  - Bottom-right: Agent audio visualization")
    print("\nPress Ctrl+C to exit\n")

    await asyncio.sleep(2)

    # Start TUI
    tui.start(refresh_per_second=10)

    try:
        # Add some initial transcripts
        tui.add_transcript("You", "Hello, testing the TUI!")
        tui.add_transcript("Agent", "Hi! I can see your message.")

        iteration = 0
        while True:
            iteration += 1

            # Simulate user speaking (varying audio levels)
            user_audio = np.random.randint(-8000, 8000, size=480).astype(np.int16)
            # Add some periodic "speech" bursts
            if iteration % 20 < 10:
                user_audio = (user_audio * 3).clip(-32767, 32767).astype(np.int16)
            tui.update_user_audio_level(user_audio)

            # Simulate agent speaking (different pattern)
            agent_audio = np.random.randint(-6000, 6000, size=480).astype(np.int16)
            if iteration % 30 < 8:
                agent_audio = (agent_audio * 4).clip(-32767, 32767).astype(np.int16)
            tui.update_agent_audio_level(agent_audio, stream_id="agent-1")

            # Add transcripts occasionally
            if iteration % 50 == 0:
                tui.add_transcript("You", "This is a partial transcript...")

            # Add final transcripts occasionally
            if iteration % 100 == 0:
                tui.add_transcript("You", f"Final message #{iteration // 100}")
                tui.add_transcript("Agent", f"Got your message #{iteration // 100}!")

            # Update display
            tui.update()

            await asyncio.sleep(0.05)  # 20 Hz updates

    except KeyboardInterrupt:
        print("\n\nStopping TUI test...")
    finally:
        tui.stop()
        print("TUI test completed!")


if __name__ == "__main__":
    asyncio.run(test_tui())
