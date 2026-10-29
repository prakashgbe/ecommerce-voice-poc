
import asyncio
import base64
import json
import logging
import os
import struct

import websockets
from websockets.asyncio.server import serve
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exotel_voicebot")

HOST = "0.0.0.0"
PORT = int(os.getenv("EXOTEL_WS_PORT", "8765"))
MODEL = os.getenv("EXOTEL_GEMINI_MODEL", "gemini-3.8-live")

client = genai.Client()


def downsample_24k_to_8k(pcm: bytes) -> bytes:
    """Convert mono signed 16-bit PCM from 24 kHz to 8 kHz."""
    usable = len(pcm) - (len(pcm) % 6)
    if usable <= 0:
        return b""

    samples = struct.unpack("<" + "h" * (usable // 2), pcm[:usable])
    output = []

    # Average each group of three samples before decimating.
    for i in range(0, len(samples) - 2, 3):
        value = round((samples[i] + samples[i + 1] + samples[i + 2]) / 3)
        output.append(max(-32768, min(32767, value)))

    return struct.pack("<" + "h" * len(output), *output)


async def handle_exotel(websocket):
    logger.info("Exotel WebSocket connected")
    stream_sid = None
    audio_buffer = bytearray()

    try:
        async with client.aio.live.connect(
            model=MODEL,
            config=types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                system_instruction=(
                    "You are a friendly telephone assistant for an online store. "
                    "Greet the caller, speak clearly and briefly, and ask how you "
                    "can help. For this initial connectivity test, do not claim "
                    "to access orders, customer accounts, or perform transactions."
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
            logger.info("Gemini Live session connected")

            async def send_gemini_audio():
                async for response in session.receive():
                    content = response.server_content
                    if not content or not content.model_turn:
                        continue

                    for part in content.model_turn.parts or []:
                        if part.inline_data and part.inline_data.data:
                            pcm24 = part.inline_data.data
                            pcm8 = downsample_24k_to_8k(pcm24)
                            if not pcm8:
                                continue

                            audio_buffer.extend(pcm8)

                            # Send audio in 200 ms blocks at 8 kHz, 16-bit mono.
                            while len(audio_buffer) >= 3200:
                                chunk = bytes(audio_buffer[:3200])
                                del audio_buffer[:3200]

                                await websocket.send(json.dumps({
                                    "event": "media",
                                    "stream_sid": stream_sid,
                                    "media": {
                                        "payload": base64.b64encode(
                                            chunk
                                        ).decode("ascii")
                                    },
                                }))

            receiver = asyncio.create_task(send_gemini_audio())

            try:
                async for message in websocket:
                    if isinstance(message, bytes):
                        logger.warning("Ignoring unexpected binary message")
                        continue

                    try:
                        event = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning("Ignoring invalid JSON event")
                        continue

                    event_type = event.get("event")

                    if event_type == "connected":
                        logger.info("Exotel connection established")

                    elif event_type == "start":
                        start = event.get("start", {})
                        stream_sid = (
                            start.get("stream_sid")
                            or event.get("stream_sid")
                        )
                        logger.info(
                            "Call started; stream_sid=%s; audio=%s",
                            stream_sid,
                            start.get("media_format", {}),
                        )
                        await session.send_realtime_input(
                            text="Greet the caller briefly and ask how you can help."
                        )

                    elif event_type == "media":
                        payload = event.get("media", {}).get("payload")
                        if payload:
                            pcm8 = base64.b64decode(payload)
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=pcm8,
                                    mime_type="audio/pcm;rate=8000",
                                )
                            )

                    elif event_type == "dtmf":
                        digit = event.get("dtmf", {}).get("digit")
                        logger.info("Caller pressed DTMF key: %s", digit)

                    elif event_type == "stop":
                        logger.info("Call ended")
                        break

            finally:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

    except websockets.exceptions.ConnectionClosed:
        logger.info("Exotel WebSocket disconnected")
    except Exception:
        logger.exception("Telephony/Gemini session failed")


async def main():
    logger.info("Starting isolated Exotel voice service on %s:%s", HOST, PORT)
    async with serve(handle_exotel, HOST, PORT, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
