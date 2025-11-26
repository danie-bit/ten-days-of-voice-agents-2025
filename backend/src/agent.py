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
You are a Sales Development Representative  for Doctor Reddy’s Pharma.
Your goals:
1. Greet the visitor warmly.
2. Understand what they need.
3. If they ask about product/company → use FAQ.if they asked anything not in the faq tell them i will take this to our team out their . and take the following details to get back to 
4. Collect lead info naturally:
   - Name
   - Company
   - Email
   - Role
   - Use case
   - Team size
   - Timeline
5. When user says “that’s all / I’m done”:
   - Give a summary of lead
   - Write JSON using save_lead()
Do NOT mention tools, JSON, or code.
Keep responses short and professional.
"""
        )

        # Load FAQ data
        base_dir = Path(__file__).resolve().parent
        faq_path = base_dir / "shared-data" / "day5_drreddy_faq.json"
        if faq_path.exists():
            with open(faq_path, "r", encoding="utf-8") as f:
                self.faq_data = json.load(f)
        else:
            self.faq_data = []

        # Lead data collection
        self.lead = {
            "name": "",
            "company": "",
            "email": "",
            "role": "",
            "use_case": "",
            "team_size": "",
            "timeline": ""
        }

    # -------- JSON Save Tool --------
    @function_tool
    async def save_lead(self, context: RunContext, lead: dict):
        base_dir = Path(__file__).resolve().parent
        log_path = base_dir / "sales_leads.json"

        # Load existing file
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except:
                data = []
        else:
            data = []

        data.append(lead)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return "Lead saved successfully."

    # -------- FAQ Search Tool --------
    @function_tool
    async def search_faq(self, context: RunContext, query: str):
        q = query.lower()
        results = []

        # --- Search About ---
        if "about" in q or "company" in q or "who are you" in q:
            results.append({"answer": self.faq_data.get("about", "No company info found.")})

        # --- Search Products ---
        if "product" in q or "medicine" in q or "offer" in q:
            products = self.faq_data.get("products", [])
            if products:
                results.append({"answer": "We offer: " + ", ".join(products)})

        # --- Search Pricing ---
        if "price" in q or "cost" in q or "pricing" in q:
            pricing = self.faq_data.get("pricing", {})
            if pricing:
                pr_text = "; ".join([f"{k.replace('_', ' ').title()} – {v}" for k, v in pricing.items()])
                results.append({"answer": pr_text})

        # --- Search FAQ List ---
        for item in self.faq_data.get("faq", []):
            if q in item["question"].lower() or q in item["answer"].lower():
                results.append(item)

        return results or [{"answer": "I'm not fully sure, but I will confirm with the team."}]


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

    assistant = Assistant()   # simple init — NO mode / tutor logic
    # ---------------------------
    # MESSAGE HANDLING LOGIC (🔥 NEW)
    # ---------------------------
    import asyncio  # <-- make sure this import exists at the TOP

    @session.on("message")  # this MUST be NON-ASYNC
    def _on_user_message(msg: str, _: str):
        # Use asyncio to handle message safely
        asyncio.create_task(handle_user_message(msg))

    # ---- async handler that does the REAL work ----
    async def handle_user_message(msg: str):
        user_text = msg.lower().strip()

        # END OF CONVERSATION → SAVE JSON
        if any(x in user_text for x in ["that's all", "that is all", "done", "thanks", "bye", "goodbye"]):
            await assistant.save_lead(session, assistant.lead)
            await session.send("Thank you! I’ll pass your information to our team and someone will reach you soon.")
            assistant.lead = {key: "" for key in assistant.lead}  # reset lead
            return

        # FAQ SEARCH
        faq_results = await assistant.search_faq(session, user_text)
        if faq_results:
            answer = faq_results[0].get("answer", None)
            if answer:
                await session.send(answer)
                return

        # LEAD COLLECTION
        if not assistant.lead["name"]:
            assistant.lead["name"] = msg.strip()
            await session.send("May I know your company name?")
            return

        if not assistant.lead["company"]:
            assistant.lead["company"] = msg.strip()
            await session.send("Great! What's your role or designation?")
            return

        if not assistant.lead["role"]:
            assistant.lead["role"] = msg.strip()
            await session.send("What would you like to use our services for?")
            return

        if not assistant.lead["use_case"]:
            assistant.lead["use_case"] = msg.strip()
            await session.send("How many people are in your team?")
            return

        if not assistant.lead["team_size"]:
            assistant.lead["team_size"] = msg.strip()
            await session.send("When are you planning to start? (now / soon / later)")
            return

        if not assistant.lead["timeline"]:
            assistant.lead["timeline"] = msg.strip()
            await session.send("Thank you. If that’s all, just say 'that's all' and I will summarise your details.")
            return

        # Fallback
        await session.send("Let me know how I can assist further.")

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
