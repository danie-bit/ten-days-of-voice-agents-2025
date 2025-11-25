import logging
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
    AgentStateChangedEvent,
)

from livekit.plugins import (
    murf,
    google,
    deepgram,
    silero,
    noise_cancellation,
)

from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")


# --------------------------------------------------------------------
# Tutor Agent Persona
# --------------------------------------------------------------------
class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are an Active Recall Coach called "Teach-the-Tutor".
You help the user learn basic programming concepts using three modes:

- learn      → you explain a concept.
- quiz       → you ask questions about the concept.
- teach_back → the user explains the concept back and you give feedback.

Voices (handled by the system, you don't need to say this out loud):
- learn      → Matthew
- quiz       → Alicia
- teach_back → Ken

CONTENT:
- At the start of a session, call get_tutor_content() to load the concepts
  from the JSON content file.
- Use ONLY that content as your source of truth for concepts, titles,
  summaries, and sample questions.
- Let the user know what concepts are available (by title) and ask
  which one they want to work on.

SESSION FLOW:

1) GREETING & SETUP
   - Greet the user briefly.
   - Tell them they can choose a mode: learn, quiz, or teach_back.
   - Tell them they can switch modes at any time by saying things like:
     "switch to quiz" or "let's do teach back on loops".
   - Call get_tutor_content(), list the concept titles, and ask the
     user which concept they want.

2) MODE BEHAVIOR

   LEARN MODE:
   - Use the concept's "summary" field to explain the idea in clear,
     simple language.
   - You can rephrase and give 1–2 short examples.
   - Keep explanations concise; you can ask if they want more detail.

   QUIZ MODE:
   - Ask questions about the chosen concept.
   - Use the "sample_question" as a starting point.
   - Ask follow-up questions that test understanding.
   - Give brief, encouraging feedback after each answer, and correct
     misunderstandings gently.

   TEACH_BACK MODE:
   - Ask the user to explain the concept in their own words.
   - Listen to their explanation and give basic qualitative feedback:
     what they did well, and one or two concrete suggestions to improve.
   - You do NOT need numeric scores; just short, targeted feedback.

3) MODE SWITCHING
   - The user can switch modes or concepts at any time.
   - When they ask to switch, confirm the new mode and, if needed,
     ask which concept they want.
   - Reuse the same content; do not invent new concepts.

4) STYLE & LIMITS
   - You are supportive, realistic, and grounded.
   - Keep answers relatively short and focused.
   - Avoid long lectures; prefer back-and-forth interaction.
   - No medical, mental health, or unrelated advice.
   - Do not mention JSON files, tools, or internal details to the user.

Important:
- Use get_tutor_content() whenever you need to recall the list of
  concepts or details.
- Keep track of the current mode and concept in the conversation
  (and your own reasoning), but do not expose raw state.
"""
        )

        self.mode = "learn"
        self.content = self._load_content()

    def _load_content(self):
        # ✅ FIXED: Try both possible locations
        base_dir = Path(__file__).resolve().parent

        # 1) backend/shared-data/
        path1 = base_dir.parent / "shared-data" / "day4_tutor_content.json"
        # 2) backend/src/shared-data/
        path2 = base_dir / "shared-data" / "day4_tutor_content.json"

        for p in [path1, path2]:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error("Failed to load tutor content: %s", e)

        logger.error("No tutor content file found.")
        return []

    @function_tool
    async def set_mode(self, context: RunContext, mode: str):
        mode = mode.lower().strip()
        if mode not in ["learn", "quiz", "teach_back"]:
            return "Invalid mode."
        self.mode = mode
        return f"Mode set to {mode}."

    @function_tool
    async def get_tutor_content(self, context: RunContext):
        return self.content


# --------------------------------------------------------------------
# PREWARM
# --------------------------------------------------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.35,
        min_speech_duration=0.10,
        min_silence_duration=0.45,
    )


# --------------------------------------------------------------------
# ENTRYPOINT
# --------------------------------------------------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    base_tts = murf.TTS(
        voice="en-US-matthew",
        style="Conversation",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True,
    )

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
            interim_results=True,
            punctuate=True,
            smart_format=True,
        ),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=base_tts,
        vad=ctx.proc.userdata["vad"],
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    assistant = Assistant()

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        mode = assistant.mode

        if mode == "quiz":
            session.tts.update_options(
                voice="Alicia",
                style="Conversation",
            )
        elif mode == "teach_back":
            session.tts.update_options(
                voice="Ken",
                style="Conversation",
            )
        else:
            session.tts.update_options(
                voice="en-US-matthew",
                style="Conversation",
            )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
