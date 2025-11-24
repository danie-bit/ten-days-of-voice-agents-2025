import logging
import asyncio

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
    ChatMessageEvent,   # ✅ NEW: import event type for "message"
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

import json
import os
from datetime import datetime

# ✅ STATIC PATH AT PROJECT ROOT
WELLNESS_LOG_PATH = r"C:\Users\study\PycharmProjects\MurfaiChallengeNov\ten-days-of-voice-agents-2025\wellness_log.json"


def load_wellness_history() -> list[dict]:
    """Load previous entries from wellness_log.json"""
    if not os.path.exists(WELLNESS_LOG_PATH):
        return []
    try:
        with open(WELLNESS_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def append_wellness_entry(entry: dict) -> None:
    """Append key-value entry to wellness_log.json as a list of dicts"""
    history = load_wellness_history()
    history.append(entry)

    try:
        with open(WELLNESS_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print("\n✔ JSON SAVED SUCCESSFULLY")
        print("PATH:", WELLNESS_LOG_PATH)
        print("DATA:", entry)

    except Exception as e:
        print("\n❌ ERROR WHILE SAVING JSON:", e)


# ---------- Coffee order state (still here, for Day 2) ----------

ORDER_TEMPLATE = {
    "drinkType": "",
    "size": "",
    "milk": "",
    "extras": [],
    "name": "",
}

current_order = None  # not used in Day 3, but keeping it


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
            continue
        if not order[field]:
            return field
    return None


def parse_user_reply(order: dict, field: str, user_text: str) -> dict:
    text = user_text.lower()

    if field == "drinkType":
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
        order["name"] = user_text.strip().title()

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
        order["extras"] = list({*order.get("extras", []), *extras})

    return order


def save_order_to_json(order: dict) -> None:
    """Save order safely inside backend/src/orders folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    orders_dir = os.path.join(base_dir, "orders")
    os.makedirs(orders_dir, exist_ok=True)

    order["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    order["order_id"] = f"ORD-{int(datetime.now().timestamp())}"

    main_file = os.path.join(orders_dir, "orders.json")

    data = []
    if os.path.exists(main_file):
        try:
            with open(main_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []

    data.append(order)

    with open(main_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    single_file = os.path.join(orders_dir, f"{order['order_id']}.json")
    with open(single_file, "w", encoding="utf-8") as f:
        json.dump(order, f, indent=2)

    print(f"✔ Order saved to: {single_file}")


# ---------- Wellness state (Day 3) ----------

current_checkin = None
current_stage = "idle"  # idle, mood, energy, stress, goals, selfcare, recap


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a calm, supportive, and grounded daily health & wellness companion.
Your job is to have short daily check-ins with the user.

RULES:
- You are NOT a doctor, therapist, or clinician.
- You NEVER diagnose or mention specific mental health disorders.
- You NEVER give medical advice or tell the user to start/stop medication.
- Focus on mood, energy, stressors, simple goals, and gentle self-care.
- Keep responses short, human, and practical.
- Do NOT use the word 'recap' unless the check-in is fully complete.
- Only give the recap after all questions have been answered.
- Until then, ask only one question at a time.
- for each question expect the straight answer if you got that , go for next question
"""
        )

    async def handle_message(self, ctx: AgentSession, message: str):
        """
        Custom state machine for Day 3:
        mood → energy → stress → goals → selfcare → recap → save JSON
        """
        global current_checkin, current_stage

        user_text = (message or "").strip()
        if not user_text:
            return

        # START NEW CHECK-IN
        if current_checkin is None or current_stage == "idle":
            history = load_wellness_history()

            last_line = ""
            if history:
                last = history[-1]
                last_mood = last.get("mood", "unknown")
                last_energy = last.get("energy", "unknown")
                last_line = (
                    f"Last time you mentioned feeling '{last_mood}' "
                    f"with energy '{last_energy}'. "
                )

            current_checkin = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mood": "",
                "energy": "",
                "stress": "",
                "goals": [],
                "self_care": "",
            }
            current_stage = "mood"

            ctx.say(
                f"Good to see you again. {last_line}"
                "How are you feeling today, in your own words?"
            )
            return

        # MOOD
        if current_stage == "mood":
            current_checkin["mood"] = user_text
            current_stage = "energy"
            ctx.say(
                "Thanks for sharing that. "
                "How would you describe your energy today? For example low, okay, or high."
            )
            return

        # ENERGY
        if current_stage == "energy":
            current_checkin["energy"] = user_text
            current_stage = "stress"
            ctx.say(
                "Got it. Is there anything stressing you out or sitting on your mind right now?"
            )
            return

        # STRESS
        if current_stage == "stress":
            current_checkin["stress"] = user_text
            current_stage = "goals"
            ctx.say(
                "Thanks for being honest. "
                "What are one to three things you’d like to get done today? "
                "They can be very small."
            )
            return

        # GOALS
        if current_stage == "goals":
            text = user_text.replace(" and ", ",")
            parts = [p.strip() for p in text.split(",") if p.strip()]
            current_checkin["goals"] = parts or [user_text.strip()]
            current_stage = "selfcare"
            ctx.say(
                "Nice. And is there anything you want to do just for yourself today? "
                "For example a short walk, some rest, hobbies, or time offline."
            )
            return

        # SELF-CARE
        if current_stage == "selfcare":
            current_checkin["self_care"] = user_text
            current_stage = "recap"

            goals_str = "; ".join(current_checkin["goals"]) or "no specific goals"
            recap = (
                "Here is what I heard for today:\n"
                f"- Mood: {current_checkin['mood']}\n"
                f"- Energy: {current_checkin['energy']}\n"
                f"- Stress: {current_checkin['stress']}\n"
                f"- Goals: {goals_str}\n"
                f"- Self-care: {current_checkin['self_care']}\n"
                "Does this sound right to you?"
            )
            ctx.say(recap)
            return

        # RECAP CONFIRMATION → SAVE JSON
        # FINAL SAVE – write JSON directly, do NOT read first, do NOT call helpers
        if current_stage == "recap":
            # Build a clean JSON object with just what we care about
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mood": current_checkin.get("mood", ""),
                "energy": current_checkin.get("energy", ""),
                "stress": current_checkin.get("stress", ""),
                "goals": current_checkin.get("goals", []),
                "self_care": current_checkin.get("self_care", ""),
            }

            try:
                # This will CREATE wellness_log.json if it does not exist
                with open(WELLNESS_LOG_PATH, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2)

                print("\n✔ wellness_log.json CREATED")
                print("PATH:", WELLNESS_LOG_PATH)
                print("DATA:", entry)

            except Exception as e:
                print("\n❌ ERROR while writing wellness_log.json:", e)

            # Say closing line to user
            ctx.say(
                "Thanks for checking in today. "
                "I’ve saved today’s check-in. If you want to talk again later, I’ll be here."
            )

            # Reset for next time
            current_checkin = None
            current_stage = "idle"
            return


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # ✅ CORRECT EVENT HOOK FOR CHAT MESSAGES
    @session.on("message")
    def _on_user_message(ev: ChatMessageEvent):
        # we only care about USER messages, not assistant responses
        if ev.item.role != "user":
            return
        text = ev.item.text_content or ""
        if not text:
            return
        # schedule our async state-machine handler
        asyncio.create_task(session.agent.handle_message(session, text))

    # ---------- metrics ----------
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
