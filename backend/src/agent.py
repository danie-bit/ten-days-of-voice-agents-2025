import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

import json
import os

# ----- Coffee order state -----

ORDER_TEMPLATE = {
    "drinkType": "",
    "size": "",
    "milk": "",
    "extras": [],
    "name": "",
}

# For now, assume single user → one order in memory
current_order = None


def is_order_complete(order: dict) -> bool:
    return (
        order["drinkType"]
        and order["size"]
        and order["milk"]
        and order["name"]
    )


def next_missing_field(order: dict) -> str | None:
    for field in ["drinkType", "size", "milk", "extras", "name"]:
        if field == "extras":
            # extras can be empty list – not mandatory
            continue
        if not order[field]:
            return field
    return None


def parse_user_reply(order: dict, field: str, user_text: str) -> dict:
    text = user_text.lower()

    if field == "drinkType":
        # naive parsing – you can refine later
        if "latte" in text:
            order["drinkType"] = "latte"
        elif "cappuccino" in text:
            order["drinkType"] = "cappuccino"
        elif "americano" in text:
            order["drinkType"] = "americano"
        elif "espresso" in text:
            order["drinkType"] = "espresso"
        else:
            order["drinkType"] = user_text.strip()

    elif field == "size":
        if "small" in text:
            order["size"] = "small"
        elif "medium" in text:
            order["size"] = "medium"
        elif "large" in text:
            order["size"] = "large"
        else:
            order["size"] = user_text.strip()

    elif field == "milk":
        if "oat" in text:
            order["milk"] = "oat milk"
        elif "almond" in text:
            order["milk"] = "almond milk"
        elif "soy" in text:
            order["milk"] = "soy milk"
        elif "regular" in text or "normal" in text or "cow" in text:
            order["milk"] = "regular milk"
        else:
            order["milk"] = user_text.strip()

    elif field == "name":
        # very simple – use entire text as name
        order["name"] = user_text.strip().title()

    # extras: we’ll allow user to mention them at any time
    extras = []
    if "extra shot" in text:
        extras.append("extra shot")
    if "whipped cream" in text:
        extras.append("whipped cream")
    if "caramel" in text:
        extras.append("caramel drizzle")
    if "vanilla" in text:
        extras.append("vanilla syrup")

    if extras:
        # merge unique extras
        order["extras"] = list({*order.get("extras", []), *extras})

    return order


def save_order_to_json(order: dict) -> None:
    """Save order safely inside backend/src/orders folder."""

    # 1️⃣ Always save relative to agent.py location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    orders_dir = os.path.join(base_dir, "orders")
    os.makedirs(orders_dir, exist_ok=True)  # creates folder if missing

    # 2️⃣ Add useful metadata
    order["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    order["order_id"] = f"ORD-{int(datetime.now().timestamp())}"

    # 3️⃣ Main file → history of ALL orders
    main_file = os.path.join(orders_dir, "orders.json")

    # 4️⃣ Load previous history
    data = []
    if os.path.exists(main_file):
        try:
            with open(main_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []

    # 5️⃣ Append new order
    data.append(order)

    # 6️⃣ Save history
    with open(main_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # 7️⃣ Also save each order separately (optional but useful)
    single_file = os.path.join(orders_dir, f"{order['order_id']}.json")
    with open(single_file, "w", encoding="utf-8") as f:
        json.dump(order, f, indent=2)

    print(f"✔ Order saved to: {single_file}")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly barista for 'Moonbrew Coffee'.
Your job is to take coffee orders using conversation.
You must collect:
  - drinkType (latte, espresso, americano, etc.)
  - size (small / medium / large)
  - milk (oat / almond / soy / regular)
  - name (for the cup)
Extras are optional: extra shot, caramel drizzle, whipped cream, vanilla syrup.

--- RULES ---
• Ask follow-up questions until all required fields are filled.
• Be friendly and brief.
• When order is complete → summarize + confirm.
• Then save the order to JSON using save_order_to_json().
• If user asks unrelated question → politely bring them back to ordering.
"""
        )

    async def on_message(self, ctx: AgentSession, message: str):
        """
        This function replaces @Agent.llm_chat_handler.
        It automatically runs whenever user speaks.
        """
        global current_order

        user_text = message.strip().lower()

        # Start new order
        if current_order is None or is_order_complete(current_order):
            current_order = ORDER_TEMPLATE.copy()
            await ctx.send("Welcome to Moonbrew Coffee! ☕ What would you like to order?")
            return

        # Fill the missing field
        missing = next_missing_field(current_order)
        if missing:
            current_order = parse_user_reply(current_order, missing, user_text)

        # Ask next info
        missing = next_missing_field(current_order)
        if missing:
            prompts = {
                "size": "Which size do you prefer — small, medium, or large?",
                "milk": "Any milk preference? (regular / oat / almond / soy)",
                "name": "What name should I write on the cup?",
            }
            await ctx.send(prompts.get(missing, "Tell me more about your order!"))
            return

        # Optional extras
        if not current_order["extras"]:
            await ctx.send(
                "Any extras? (extra shot / caramel / whipped cream / vanilla) — or say 'no extras'"
            )
            return

        # FINAL STEP → SAVE ORDER
        save_order_to_json(current_order)

        summary = (
            f"☕ Order Summary:\n"
            f"- {current_order['size'].title()} {current_order['drinkType'].title()}\n"
            f"- Milk: {current_order['milk']}\n"
            f"- Extras: {', '.join(current_order['extras']) or 'None'}\n"
            f"- Name: {current_order['name']}\n"
            "Thanks! I'll prepare that quickly. Want to order another one?"
        )

        await ctx.send(summary)
        current_order = None  # Reset for next customer

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
