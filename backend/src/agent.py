import logging
import json
import os
import random
from datetime import datetime
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# --- IMPROV SCENARIOS ---
SCENARIOS = [
    "You are a time-travelling tour guide explaining smartphones to someone from the 1800s.",
    "You are a barista telling a customer their latte is a portal to another dimension.",
    "You are a waiter explaining that their order escaped the kitchen.",
    "You are a customer returning a clearly cursed object to a skeptical shop owner.",
    "You are a pizza delivery person who delivered to the wrong century.",
    "You are a therapist counseling a robot that thinks it’s becoming human.",
    "You are a librarian trying to quiet down time-traveling vikings.",
    "You are a tech support agent helping someone use a magic wand.",
    "You are a wedding planner helping a ghost marry a vampire.",
    "You are a museum tour guide and all the exhibits come to life.",
]

POSITIVE_REACTIONS = [
    "That was awesome! I loved how you handled {detail}. Great energy!",
    "Amazing performance! Your use of {detail} really stood out.",
    "Nice! The {detail} part was especially fun.",
    "Fantastic! You really leaned into it with {detail}.",
]

NEUTRAL_REACTIONS = [
    "Good effort! The {detail} worked, but you can push it a bit more.",
    "Not bad! I liked {detail}, though the pacing could improve.",
    "Solid! You used {detail} nicely, but it felt like you held back.",
    "Pretty good! {detail} had potential — keep exploring it.",
]

CRITICAL_REACTIONS = [
    "Hmm, that felt a little flat. The {detail} could use stronger delivery.",
    "I see the idea with {detail}, but it lacked energy.",
    "You had something with {detail}, but it didn’t fully land.",
    "The {detail} started well but fizzled — don’t be afraid to go big!",
]

def get_random_reaction(detail="your performance"):
    rtype = random.choice(["positive", "neutral", "critical"])
    if rtype == "positive":
        template = random.choice(POSITIVE_REACTIONS)
    elif rtype == "neutral":
        template = random.choice(NEUTRAL_REACTIONS)
    else:
        template = random.choice(CRITICAL_REACTIONS)
    return template.format(detail=detail)


# --- AGENT ---
class ImprovBattleAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are the host of IMPROV BATTLE — energetic, fun, and clear.\n"
                "GAME FLOW:\n"
                "- Welcome contestant, get their name.\n"
                "- Explain the game briefly.\n"
                "- Run exactly 3 improv rounds.\n"
                "Each round:\n"
                "1. Present a scenario.\n"
                "2. Say 'Action!' and listen.\n"
                "3. When contestant finishes, call get_host_reaction.\n"
                "4. Share reaction.\n"
                "After 3 rounds, call end_game.\n"
                "Keep responses short, punchy, and natural.\n"
            )
        )

        self.state = {
            "player_name": "Ajit",
            "current_round": 0,
            "max_rounds": 3,
            "phase": "intro",
            "scenarios_used": [],
            "reactions": [],
            "game_started": False
        }

    @function_tool
    async def start_game(self, context: RunContext, player_name: str):
        self.state["player_name"] = player_name
        self.state["game_started"] = True
        self.state["phase"] = "intro"
        return f"Game started for {player_name}! Let's begin."

    @function_tool
    async def present_scenario(self, context: RunContext):
        if self.state["current_round"] >= self.state["max_rounds"]:
            return "All rounds complete."

        available = [s for s in SCENARIOS if s not in self.state["scenarios_used"]]
        if not available:
            available = SCENARIOS

        scenario = random.choice(available)
        self.state["scenarios_used"].append(scenario)
        self.state["current_round"] += 1
        self.state["phase"] = "awaiting_improv"

        return f"Round {self.state['current_round']}: {scenario}"

    @function_tool
    async def get_host_reaction(self, context: RunContext, performance_summary: str):
        reaction = get_random_reaction(performance_summary)
        self.state["phase"] = "reacting"

        self.state["reactions"].append({
            "round": self.state["current_round"],
            "scenario": self.state["scenarios_used"][-1],
            "reaction": reaction
        })

        return reaction

    @function_tool
    async def end_game(self, context: RunContext):
        self.state["phase"] = "done"
        return f"Improv Battle complete! {self.state['player_name']} finished {self.state['current_round']} rounds."


# --- LIVEKIT RUNTIME ---
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    tts = murf.TTS(
        voice="en-US-terrell",
        style="Conversation",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True
    )

    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=tts,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    await session.start(
        agent=ImprovBattleAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    await session.agent.say(
        "Welcome to IMPROV BATTLE! I'm your host. What’s your name?",
        allow_interruptions=True
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
