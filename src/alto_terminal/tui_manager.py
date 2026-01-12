"""TUI Manager for visualizing audio streams and conversation transcripts.

This module provides a Rich-based Terminal User Interface with three panels:
1. Top panel: Conversation transcripts (agent and user messages)
2. Bottom-left panel: User audio stream visualization
3. Bottom-right panel: Agent audio stream visualization
"""

import asyncio
import threading
from typing import Optional, List, Dict
from collections import deque
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.text import Text


class TUIManager:
    """Manages the terminal user interface for audio and transcript visualization."""

    def __init__(self, max_transcript_lines: int = 20, max_audio_history: int = 50):
        """Initialize the TUI manager.

        Args:
            max_transcript_lines: Maximum number of transcript lines to display
            max_audio_history: Number of audio level samples to keep for visualization
        """
        self.console = Console()
        self.max_transcript_lines = max_transcript_lines
        self.max_audio_history = max_audio_history

        # Thread-safe storage for transcripts
        self._transcript_lock = threading.Lock()
        self._transcripts: deque = deque(maxlen=max_transcript_lines)
        self._last_speaker: Optional[str] = None

        # Thread-safe storage for audio levels
        self._audio_lock = threading.Lock()
        self._user_audio_levels: deque = deque(maxlen=self.max_audio_history)
        self._agent_audio_levels: Dict[str, deque] = {}  # Per-participant levels

        # Live display
        self._live: Optional[Live] = None
        self._layout: Optional[Layout] = None
        self._running = False

    def add_transcript(self, speaker: str, message: str):
        """Add a transcript message to the conversation panel.

        Args:
            speaker: Name of the speaker (e.g., "User", "Agent", participant identity)
            message: The transcribed text
        """
        with self._transcript_lock:
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Check if we should merge with the previous message
            if self._last_speaker == speaker and self._transcripts:
                prev_timestamp, prev_speaker, prev_message = self._transcripts.pop()
                merged_message = prev_message + message
                self._transcripts.append((prev_timestamp, speaker, merged_message))
            else:
                # New speaker or first message
                self._transcripts.append((timestamp, speaker, message))

            self._last_speaker = speaker

    def update_user_audio_level(self, audio_data: np.ndarray):
        """Update user audio visualization with new audio data.

        Args:
            audio_data: Audio samples (int16 format)
        """
        # Calculate RMS level (Root Mean Square)
        if len(audio_data) > 0:
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            # Normalize to 0-1 range (int16 max is 32767)
            level = min(rms / 32767.0, 1.0)
        else:
            level = 0.0

        with self._audio_lock:
            self._user_audio_levels.append(level)

    def update_agent_audio_level(
        self, audio_data: np.ndarray, stream_id: str = "agent"
    ):
        """Update agent audio visualization with new audio data.

        Args:
            audio_data: Audio samples (int16 format)
            stream_id: Identifier for the audio stream (participant identity)
        """
        # Calculate RMS level
        if len(audio_data) > 0:
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            level = min(rms / 32767.0, 1.0)
        else:
            level = 0.0

        with self._audio_lock:
            if stream_id not in self._agent_audio_levels:
                self._agent_audio_levels[stream_id] = deque(
                    maxlen=self.max_audio_history
                )
            self._agent_audio_levels[stream_id].append(level)

    def _create_layout(self) -> Layout:
        """Create the 3-panel layout."""
        layout = Layout()

        # Split into top (2/3) and bottom (1/3)
        layout.split_column(
            Layout(name="transcript", ratio=2), Layout(name="audio_viz", ratio=1)
        )

        # Split bottom into left (user) and right (agent)
        layout["audio_viz"].split_row(
            Layout(name="user_audio"), Layout(name="agent_audio")
        )

        return layout

    def _render_transcript_panel(self) -> Panel:
        """Render the transcript panel."""
        with self._transcript_lock:
            if not self._transcripts:
                content = Text("Waiting for conversation...", style="dim")
            else:
                lines = []
                for timestamp, speaker, message in self._transcripts:
                    # Color code by speaker
                    if speaker.lower() in ["user", "you"]:
                        speaker_color = "cyan"
                    else:
                        speaker_color = "green"

                    line = Text()
                    line.append(f"[{timestamp}] ", style="dim")
                    line.append(f"{speaker}: ", style=f"bold {speaker_color}")
                    line.append(message)
                    lines.append(line)

                content = Text("\n").join(lines)

        return Panel(
            content, title="[bold]Conversation", border_style="blue", padding=(1, 2)
        )

    def _render_audio_bar(self, levels: List[float], max_width: int = 40) -> str:
        """Render an audio level bar chart.

        Args:
            levels: List of audio levels (0.0 to 1.0)
            max_width: Maximum width of the bar chart

        Returns:
            String representation of the bar chart
        """
        if not levels:
            return "─" * max_width

        # Get the most recent level
        current_level = levels[-1] if levels else 0.0

        # Create VU meter bar
        filled_width = int(current_level * max_width)
        bar = "█" * filled_width + "░" * (max_width - filled_width)

        # Add color based on level
        if current_level < 0.3:
            style = "green"
        elif current_level < 0.7:
            style = "yellow"
        else:
            style = "red"

        return f"[{style}]{bar}[/{style}]"

    def _render_audio_waveform(
        self, levels: List[float], max_width: int = 40, max_height: int = 8
    ) -> str:
        """Render a simple waveform visualization.

        Args:
            levels: List of audio levels (0.0 to 1.0)
            max_width: Maximum width of the waveform
            max_height: Maximum height of the waveform

        Returns:
            Multi-line string representation of the waveform
        """
        if not levels:
            return "\n".join(["─" * max_width] * max_height)

        # Take the most recent samples
        recent_levels = list(levels)[-max_width:]

        # Normalize and scale to height
        lines = []
        for h in range(max_height, 0, -1):
            threshold = h / max_height
            line = ""
            for level in recent_levels:
                if level >= threshold:
                    if level >= 0.7:
                        line += "[red]█[/red]"
                    elif level >= 0.4:
                        line += "[yellow]█[/yellow]"
                    else:
                        line += "[green]█[/green]"
                else:
                    line += " "
            # Pad to max_width
            line += " " * (max_width - len(recent_levels))
            lines.append(line)

        return "\n".join(lines)

    def _render_user_audio_panel(self) -> Panel:
        """Render the user audio visualization panel."""
        with self._audio_lock:
            levels = list(self._user_audio_levels)

        # Current level indicator
        current_level = levels[-1] if levels else 0.0
        level_text = Text()
        level_text.append("Level: ", style="bold")
        level_text.append(f"{current_level:.2%}")

        # VU meter
        vu_meter = self._render_audio_bar(levels, max_width=30)

        # Waveform
        waveform = self._render_audio_waveform(levels, max_width=35, max_height=6)

        # Build content with proper markup handling
        content_parts = [
            level_text,
            Text("\n\n"),
            Text.from_markup(vu_meter),
            Text("\n\n"),
            Text.from_markup(waveform),
        ]

        content = Text()
        for part in content_parts:
            content.append(part)

        return Panel(
            content, title="[bold cyan]User Audio", border_style="cyan", padding=(1, 1)
        )

    def _render_agent_audio_panel(self) -> Panel:
        """Render the agent audio visualization panel."""
        with self._audio_lock:
            # Combine all agent streams
            all_levels = []
            for stream_id, levels in self._agent_audio_levels.items():
                all_levels.extend(levels)

            if all_levels:
                # Use the most recent levels
                recent_levels = all_levels[-self.max_audio_history :]
            else:
                recent_levels = []

        # Current level indicator
        current_level = recent_levels[-1] if recent_levels else 0.0
        level_text = Text()
        level_text.append("Level: ", style="bold")
        level_text.append(f"{current_level:.2%}")

        # VU meter
        vu_meter = self._render_audio_bar(recent_levels, max_width=30)

        # Waveform
        waveform = self._render_audio_waveform(
            recent_levels, max_width=35, max_height=6
        )

        # Show active streams
        with self._audio_lock:
            stream_count = len(self._agent_audio_levels)

        streams_text = Text()
        streams_text.append(f"Active streams: {stream_count}", style="dim")

        # Build content with proper markup handling
        content_parts = [
            level_text,
            Text("\n\n"),
            Text.from_markup(vu_meter),
            Text("\n\n"),
            Text.from_markup(waveform),
            Text("\n\n"),
            streams_text,
        ]

        content = Text()
        for part in content_parts:
            content.append(part)

        return Panel(
            content,
            title="[bold green]Agent Audio",
            border_style="green",
            padding=(1, 1),
        )

    def _render(self) -> Layout:
        """Render the complete TUI."""
        if self._layout is None:
            self._layout = self._create_layout()

        # Update panels
        self._layout["transcript"].update(self._render_transcript_panel())
        self._layout["user_audio"].update(self._render_user_audio_panel())
        self._layout["agent_audio"].update(self._render_agent_audio_panel())

        return self._layout

    def start(self, refresh_per_second: int = 10):
        """Start the TUI display.

        Args:
            refresh_per_second: Update frequency in Hz
        """
        if self._running:
            return

        self._running = True
        self._layout = self._create_layout()

        # Start live display
        self._live = Live(
            self._render(),
            console=self.console,
            screen=True,
            refresh_per_second=refresh_per_second,
        )
        self._live.start()

    def stop(self):
        """Stop the TUI display."""
        if not self._running:
            return

        self._running = False

        if self._live:
            self._live.stop()
            self._live = None

    def update(self):
        """Update the TUI display (call periodically)."""
        if self._running and self._live:
            self._live.update(self._render())

    def is_running(self) -> bool:
        """Check if TUI is currently running."""
        return self._running

    def clear_transcripts(self):
        """Clear all transcripts."""
        with self._transcript_lock:
            self._transcripts.clear()

    def clear_audio_levels(self):
        """Clear all audio level history."""
        with self._audio_lock:
            self._user_audio_levels.clear()
            self._agent_audio_levels.clear()
