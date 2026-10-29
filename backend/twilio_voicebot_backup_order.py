
import asyncio
import base64
import logging
import os
import re
import struct
import json
import urllib.request
import urllib.error

try:
    from rag_service import generate_policy_answer
except Exception:
    generate_policy_answer = None

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from google import genai
from google.genai import types
import uvicorn

load_dotenv()
os.environ.setdefault("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("twilio_voicebot")

HOST = "0.0.0.0"
PORT = int(os.getenv("TWILIO_WS_PORT", "8765"))
MODEL = os.getenv("TWILIO_GEMINI_MODEL", "gemini-3.8-live")

app = FastAPI(title="Isolated Twilio Voice Bridge")
client = genai.Client()


API_URL = os.getenv("ECOMMERCE_API_URL", "http://127.0.0.1:8000")


def api_request(path: str, method: str = "GET", payload: dict | None = None,
                access_token: str | None = None) -> dict:
    """Call the existing e-commerce FastAPI backend without logging credentials."""
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{API_URL}{path}", data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail", "Request failed")
        except Exception:
            detail = "Request failed"
        return {"error": str(detail), "status_code": exc.code}
    except Exception as exc:
        logger.warning("E-commerce backend request failed: %s", type(exc).__name__)
        return {"error": "The customer service backend is unavailable."}


def live_tools():
    """Function declarations available to Gemini Live for this call."""
    return [
        types.FunctionDeclaration(
            name="verify_customer",
            description="Verify a caller before accessing any order information. Ask for customer ID and four-digit verification PIN first. Never call without both values.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "customer_id": types.Schema(type="STRING", description="Customer ID"),
                    "verification_pin": types.Schema(type="STRING", description="Four-digit customer verification PIN"),
                },
                required=["customer_id", "verification_pin"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_my_orders",
            description="Retrieve the authenticated caller's order history. Only call after verify_customer succeeds.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="get_my_order_details",
            description="Retrieve one order belonging to the authenticated caller. Only call after verify_customer succeeds.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"order_id": types.Schema(type="STRING", description="Order ID")},
                required=["order_id"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_commerce_policies",
            description="Look up official e-commerce policies for refunds, returns, shipping, delivery, or cancellations.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"question": types.Schema(type="STRING", description="Customer's policy question")},
                required=["question"],
            ),
        ),
    ]


async def execute_live_tool(name: str, args: dict, state: dict) -> dict:
    """Execute a declared tool; order APIs are blocked until backend verification succeeds."""
    if name == "verify_customer":
        customer_id = str(args.get("customer_id", "")).strip().upper()
        pin = str(args.get("verification_pin", "")).strip()
        if not customer_id or not re.fullmatch(r"\d{4}", pin):
            state["access_token"] = None
            state["customer_id"] = None
            state["order_auth_prompted"] = False
            return {"verified": False, "message": "Please provide a valid customer ID and four-digit PIN."}

        result = await asyncio.to_thread(
            api_request, "/api/customers/verify", "POST",
            {"customer_id": customer_id, "verification_pin": pin}
        )
        if result.get("error") or not result.get("verified") or not result.get("access_token"):
            state["access_token"] = None
            state["customer_id"] = None
            state["order_auth_prompted"] = False
            return {"verified": False, "message": "Verification failed. Please check your customer ID and PIN."}

        state["access_token"] = result["access_token"]
        state["customer_id"] = result.get("customer_id", customer_id)
        return {
            "verified": True,
            "customer_name": result.get("customer_name", ""),
            "message": "Customer verified successfully. You may now look up this customer's orders."
        }

    if name in {"get_my_orders", "get_my_order_details"}:
        token = state.get("access_token")
        customer_id = state.get("customer_id")
        if not token or not customer_id:
            return {"success": False, "message": "Please verify your identity with your customer ID and PIN before looking up orders."}

        if name == "get_my_orders":
            result = await asyncio.to_thread(
                api_request, "/api/customers/me/orders", "GET", None, token
            )
            if result.get("status_code") == 401:
                state["access_token"] = None
                state["customer_id"] = None
                return {"success": False, "message": "Your verification session expired. Please verify again."}
            if result.get("error"):
                return {"success": False, "message": "I couldn't retrieve your orders right now."}
            return {"success": True, "orders": result.get("orders", [])}

        order_id = str(args.get("order_id", "")).strip().upper()
        if not order_id:
            return {"success": False, "message": "Please provide the order ID."}
        result = await asyncio.to_thread(
            api_request, f"/api/orders/{order_id}", "GET", None, token
        )
        if result.get("status_code") == 401:
            state["access_token"] = None
            state["customer_id"] = None
            return {"success": False, "message": "Your verification session expired. Please verify again."}
        if result.get("error"):
            return {"success": False, "message": "I couldn't find that order under your account or retrieve it right now."}
        # Return only customer-facing fields; never expose credentials.
        fields = ("order_id", "product", "amount", "status", "shipment_status", "order_date", "delivered_date")
        return {"success": True, "order": {k: result[k] for k in fields if k in result}}

    if name == "search_commerce_policies":
        if generate_policy_answer is None:
            return {"answer": "I couldn't retrieve the policy right now. Please contact customer support.", "sources": []}
        question = str(args.get("question", "")).strip()
        if not question:
            return {"answer": "Please tell me which policy you'd like to know about.", "sources": []}
        try:
            return await asyncio.to_thread(generate_policy_answer, question)
        except Exception as exc:
            logger.warning("Policy lookup failed: %s", type(exc).__name__)
            return {"answer": "I couldn't retrieve the policy right now. Please contact customer support.", "sources": []}

    return {"error": "Unknown tool."}


def mulaw_to_pcm(mulaw: bytes) -> bytes:
    """Decode G.711 mu-law bytes to signed 16-bit little-endian PCM."""
    output = bytearray()

    for value in mulaw:
        value = (~value) & 0xFF
        sign = value & 0x80
        exponent = (value >> 4) & 0x07
        mantissa = value & 0x0F

        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84

        if sign:
            sample = -sample

        output.extend(struct.pack("<h", sample))

    return bytes(output)

def pcm8k_to_pcm16k(pcm: bytes) -> bytes:
    """Resample mono 16-bit PCM from 8 kHz to 16 kHz."""
    sample_count = len(pcm) // 2
    if sample_count == 0:
        return b""

    samples = struct.unpack(
        "<" + "h" * sample_count,
        pcm[:sample_count * 2],
    )

    output = []
    for i, current in enumerate(samples):
        output.append(current)

        if i + 1 < sample_count:
            next_sample = samples[i + 1]
            output.append((current + next_sample) // 2)
        else:
            output.append(current)

    return struct.pack("<" + "h" * len(output), *output)

def pcm_sample_to_mulaw(sample: int) -> int:
    """Encode one signed 16-bit PCM sample as G.711 mu-law."""
    sample = max(-32768, min(32767, sample))
    mask = 0xFF

    if sample < 0:
        sample = -sample
        mask = 0x7F

    sample = min(sample, 32635) + 0x84
    segment_ends = (0xFF, 0x1FF, 0x3FF, 0x7FF,
                    0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)

    segment = 0
    while segment < len(segment_ends) and sample > segment_ends[segment]:
        segment += 1

    if segment >= 8:
        encoded = 0x7F
    else:
        encoded = (segment << 4) | ((sample >> (segment + 3)) & 0x0F)

    return encoded ^ mask


def pcm24k_to_mulaw8k(pcm: bytes) -> bytes:
    """Downsample 24 kHz PCM to 8 kHz and encode as mu-law."""
    sample_count = len(pcm) // 2
    sample_count -= sample_count % 3

    if sample_count == 0:
        return b""

    samples = struct.unpack("<" + "h" * sample_count, pcm[:sample_count * 2])
    output = bytearray()

    # Average each group of 3 samples to reduce aliasing slightly.
    for i in range(0, len(samples), 3):
        sample = round((samples[i] + samples[i + 1] + samples[i + 2]) / 3)
        output.append(pcm_sample_to_mulaw(sample))

    return bytes(output)


async def send_audio_to_twilio(websocket, stream_sid, mulaw_audio):
    """Send 8 kHz mu-law audio to Twilio in paced 20 ms frames."""
    frame_size = 160

    for i in range(0, len(mulaw_audio), frame_size):
        frame = mulaw_audio[i:i + frame_size]

        await websocket.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": base64.b64encode(frame).decode("ascii"),
            },
        })

        # Play at approximately real-time speed.
        await asyncio.sleep(len(frame) / 8000)



@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request):
    # Twilio calls this HTTP endpoint when the number receives a call.
    host = request.headers.get("host") or request.url.hostname
    stream_url = f"wss://{host}/media-stream"

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>"""

    logger.info("Returning TwiML for stream URL: %s", stream_url)
    return Response(content=xml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio WebSocket connected")

    stream_sid = None
    receiver_task = None
    playback_task = None
    # Decouple Gemini's receive loop from real-time Twilio playback. If the receive
    # loop waits while pacing audio to Twilio, Gemini's websocket can miss keepalives.
    audio_out_queue = asyncio.Queue(maxsize=1000)
    audio_frames_received = 0
    auth_state = {"access_token": None, "customer_id": None, "order_auth_prompted": False}

    try:
        async with client.aio.live.connect(
            model=MODEL,
            config=types.LiveConnectConfig(
                input_audio_transcription=types.AudioTranscriptionConfig(),
                response_modalities=["AUDIO"],
                tools=[types.Tool(function_declarations=live_tools())],
                system_instruction=(
                    "You are a friendly e-commerce customer service voice assistant. "
                    "Speak clearly and concisely in English. Greet the caller and ask how you can help. "
                    "For questions about returns, cancellations, shipping, delivery, or refunds, "
                    "call search_commerce_policies and base your answer on its result. "
                    "For any order-status, order-history, or order-detail request, first ask for "
                    "the customer's ID and four-digit PIN if you have not already collected them "
                    "during this call. Then call verify_customer. Do not access or disclose order "
                    "information unless the tool returns verified=true. After successful verification, "
                    "use get_my_orders for order history or get_my_order_details for a specific order. "
                    "If the caller asks where an order is but does not provide an order ID, verify "
                    "them first and then retrieve their orders; ask which order they mean if there "
                    "is more than one. Never invent order details, tracking information, or policies. "
                    "Never speak or reveal access tokens. Do not repeat or ask for the PIN again after "
                    "verification succeeds in this call. If verification or lookup fails, explain this "
                    "politely. Do not cancel orders or create return requests; explain policy only. "
                    "Never say you checked an order unless the corresponding tool succeeded."
                ),
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Aoede"
                        )
                    )
                ),
            ),
        ) as session:


            async def play_queued_audio():
                """Pace Gemini audio to Twilio without blocking Gemini's receive loop."""
                while True:
                    frame = await audio_out_queue.get()
                    try:
                        if frame is None:
                            return
                        if stream_sid:
                            await send_audio_to_twilio(websocket, stream_sid, frame)
                    finally:
                        audio_out_queue.task_done()


            async def send_gemini_audio():
                # Gemini Live's receive iterator can finish at the end of a model turn.
                # Keep the receiver task alive for subsequent turns while the session remains open.
                while True:
                    async for response in session.receive():
                        tool_call = getattr(response, "tool_call", None)
                        if tool_call and getattr(tool_call, "function_calls", None):
                            function_responses = []
                            for call in tool_call.function_calls:
                                args = dict(call.args or {})
                                logger.info("Gemini requested tool: %s", call.name)
                                result = await execute_live_tool(call.name, args, auth_state)
                                function_responses.append(
                                    types.FunctionResponse(
                                        id=call.id,
                                        name=call.name,
                                        response=result,
                                    )
                                )
                            await session.send_tool_response(function_responses=function_responses)
                            continue

                        content = response.server_content
                        if not content:
                            continue

                        logger.debug(
                            "Gemini response received; model_turn=%s",
                            bool(content.model_turn),
                        )

                        if content.input_transcription:
                            caller_text = (content.input_transcription.text or "").strip()
                            logger.info("Caller transcription: %s", caller_text)

                            # Keep the existing live-audio flow, but explicitly steer Gemini
                            # when the caller asks about an order before authenticating.
                            order_terms = (
                                "order", "orders", "tracking", "track my", "where is my",
                                "delivery status", "shipment", "shipped", "purchase history",
                            )
                            if (
                                caller_text
                                and any(term in caller_text.lower() for term in order_terms)
                                and not auth_state.get("access_token")
                                and not auth_state.get("order_auth_prompted")
                            ):
                                auth_state["order_auth_prompted"] = True
                                logger.info("Order-related request detected before verification; prompting for customer ID and PIN")
                                await session.send_client_content(
                                    turns=types.Content(
                                        role="user",
                                        parts=[types.Part(text=(
                                            "The caller has asked an order-related question and is not verified. "
                                            "Your next spoken response must ask them for their customer ID and "
                                            "four-digit verification PIN. Do not claim to check any order yet."
                                        ))],
                                    ),
                                    turn_complete=True,
                                )

                        if content.output_transcription:
                            logger.info(
                                "Assistant transcription: %s",
                                content.output_transcription.text,
                            )

                        if content.turn_complete:
                            logger.info("Gemini turn completed")

                        if not content.model_turn:
                            continue

                        for part in content.model_turn.parts or []:
                            if not part.inline_data or not part.inline_data.data:
                                continue

                            pcm24 = part.inline_data.data
                            mulaw8 = pcm24k_to_mulaw8k(pcm24)

                            if mulaw8 and stream_sid:
                                # Queue 20 ms Twilio frames. The receive task must not sleep
                                # while audio is played, otherwise Gemini websocket keepalives
                                # can time out during a long response.
                                for offset in range(0, len(mulaw8), 160):
                                    await audio_out_queue.put(mulaw8[offset:offset + 160])
                                logger.debug("Queued Gemini audio for Twilio: %d bytes", len(mulaw8))

                    logger.warning("Gemini receive iterator ended; reopening receive iterator")
                    await asyncio.sleep(0.05)  # Avoid a tight loop if the iterator repeatedly ends.



            async for raw_message in websocket.iter_text():
                import json
                event = json.loads(raw_message)
                event_type = event.get("event")

                if event_type == "connected":
                    logger.info("Twilio Media Streams connected")

                elif event_type == "start":
                    stream_sid = event["start"]["streamSid"]
                    logger.info(
                        "Call started: callSid=%s, streamSid=%s, format=%s",
                        event["start"].get("callSid"),
                        stream_sid,
                        event["start"].get("mediaFormat"),
                    )

                    if receiver_task is None:
                        receiver_task = asyncio.create_task(send_gemini_audio())
                    if playback_task is None:
                        playback_task = asyncio.create_task(play_queued_audio())

                    await session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    text="Greet the caller briefly and ask how you can help."
                                )
                            ],
                        ),
                        turn_complete=True,
                    )

                elif event_type == "media":
                    payload = event.get("media", {}).get("payload")
                    if payload:
                        mulaw = base64.b64decode(payload)
                        pcm = mulaw_to_pcm(mulaw)

                        # Twilio sends 20 ms frames. Avoid logging every frame because
                        # synchronous DEBUG console output can starve websocket keepalives.
                        audio_frames_received += 1
                        if audio_frames_received % 100 == 0:
                            logger.info("Caller audio stream active (%d frames received)", audio_frames_received)
                        pcm16k = pcm8k_to_pcm16k(pcm)

                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=pcm16k,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )

                elif event_type == "dtmf":
                    digit = event.get("dtmf", {}).get("digit")
                    logger.info("Caller pressed DTMF: %s", digit)

                elif event_type == "stop":
                    logger.info("Twilio call stream stopped")
                    break

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception:
        logger.exception("Twilio/Gemini bridge failed")
    finally:
        if playback_task:
            playback_task.cancel()
        if receiver_task:
            receiver_task.cancel()
        tasks = [task for task in (receiver_task, playback_task) if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    logger.info("Starting isolated Twilio bridge on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
