"""
LiveKit Voice AI Agent with Standard STT → LLM → TTS Pipeline

This agent demonstrates a basic voice AI implementation using the LiveKit Agents framework.
It listens to user speech, processes it through an LLM, and responds with synthesized speech.

Requirements:
- Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in environment or .env file
- Install: uv add livekit-agents livekit-plugins-openai livekit-plugins-deepgram livekit-plugins-silero

Run:
    uv run python agent.py start
"""

import logging
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, silero

# Configure logging
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent / '.env')


class VoiceAgent(Agent):
    """
    A basic voice AI agent that can listen and respond to users.

    This agent uses:
    - Deepgram for Speech-to-Text (STT)
    - OpenAI GPT-4 for Language Model (LLM)
    - OpenAI TTS for Text-to-Speech (TTS)
    - Silero for Voice Activity Detection (VAD)
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a helpful voice assistant. You communicate through natural speech.

                Guidelines:
                - Keep responses concise and conversational
                - Speak naturally, as if having a real conversation
                - Avoid using special characters or symbols that are hard to pronounce
                - Be friendly, professional, and helpful
                - If you don't understand something, politely ask for clarification
            """,
            # STT: Deepgram Nova 2 - high accuracy speech recognition
            stt=deepgram.STT(model="nova-2-general"),

            # LLM: OpenAI GPT-4o Mini - fast and cost-effective
            llm=openai.LLM(model="gpt-4o-mini"),

            # TTS: OpenAI TTS - natural sounding voice
            tts=openai.TTS(voice="alloy"),

            # VAD: Silero - voice activity detection to know when user is speaking
            vad=silero.VAD.load()
        )

    async def on_enter(self):
        """
        Called when the agent first enters a room.
        Generates an initial greeting to start the conversation.
        """
        logger.info("Agent entered room, generating initial greeting")
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    """
    Entrypoint function that is called when an agent is assigned to a room.

    Args:
        ctx: Job context containing room connection and metadata
    """
    logger.info(f"Starting agent for room: {ctx.room.name}")

    # Create an agent session
    session = AgentSession()

    # Start the agent session with our voice agent and the room
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room
    )

    logger.info("Agent session started successfully")


if __name__ == "__main__":
    # Run the agent worker
    # This will connect to LiveKit and wait for job assignments
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
