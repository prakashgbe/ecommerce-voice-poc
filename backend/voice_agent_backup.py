
import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import google

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

class EcommerceVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a friendly e-commerce customer service assistant. "
                "Speak clearly and concisely in English. "
                "For this initial test, answer general questions politely. "
                "Do not claim to have checked an order or performed a transaction."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-3.8-live",
            voice="Aoede",
        )
    )

    await ctx.connect()
    await session.start(
        room=ctx.room,
        agent=EcommerceVoiceAgent(),
    )
    await session.generate_reply(
        instructions="Greet the customer and ask how you can help."
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
