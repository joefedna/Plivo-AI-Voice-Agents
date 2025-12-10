# server.py
import os
import json
import sys
import asyncio
import base64
import io
import traceback
import wave
from typing import Optional

import numpy as np
import webrtcvad
import websockets

from deepgram import Deepgram
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# ------------ Configuration loader (env first, fallback to config.json) ------------
def load_config_from_env_or_file(file_path: str = "config.json"):
    cfg = {
        # Plivo
        "PLIVO_AUTH_ID": os.getenv("PLIVO_AUTH_ID"),
        "PLIVO_AUTH_TOKEN": os.getenv("PLIVO_AUTH_TOKEN"),
        # Third party APIs
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
        # Optional
        "VOICE_ID": os.getenv("VOICE_ID"),  # ElevenLabs voice id override
    }

    # If any required keys are missing and a config file exists, try loading it
    missing = [k for k, v in cfg.items() if v is None]
    if missing and os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                disk = json.load(f)
            # Look for lowercase keys in the file (repo convention)
            for k in cfg:
                if not cfg[k]:
                    val = disk.get(k.lower())
                    if val:
                        cfg[k] = val
        except Exception:
            # do not raise here, proceed with what we have
            traceback.print_exc()

    return cfg


CONFIG = load_config_from_env_or_file()

# Validate minimal keys (warn rather than crash so you can run locally with partial config)
if not CONFIG.get("DEEPGRAM_API_KEY"):
    print("Warning: DEEPGRAM_API_KEY not found in environment or config.json")
if not CONFIG.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in environment or config.json")
if not CONFIG.get("ELEVENLABS_API_KEY"):
    print("Warning: ELEVENLABS_API_KEY not found in environment or config.json")

# ------------ Clients ------------
dg_client = Deepgram(CONFIG.get("DEEPGRAM_API_KEY")) if CONFIG.get("DEEPGRAM_API_KEY") else None
openai_client = OpenAI(api_key=CONFIG.get("OPENAI_API_KEY")) if CONFIG.get("OPENAI_API_KEY") else None
tts_client = ElevenLabs(api_key=CONFIG.get("ELEVENLABS_API_KEY")) if CONFIG.get("ELEVENLABS_API_KEY") else None

# Default voice id if not provided
DEFAULT_VOICE_ID = CONFIG.get("VOICE_ID") or "XrExE9yKIg1WjnnlVkGX"

# ------------ Conversation memory and persona ------------
messages = []
system_msg = """Your name is Matilda. Matilda is a warm and friendly voicebot designed to have pleasant and engaging 
conversations with customers. Matilda's primary purpose is to greet customers in a cheerful and polite manner whenever 
they say 'hello' or any other greeting. She should respond with kindness, using a welcoming tone to make the customer 
feel valued and appreciated.

Matilda should always use positive language and maintain a light, conversational tone throughout the interaction. Her 
responses should be concise, friendly, and focused on making the customer feel comfortable and engaged. She should avoid 
overly complex language and strive to keep the conversation pleasant and easy-going."""
messages.append({"role": "system", "content": system_msg})

# ------------ Utility: write PCM16 bytes to WAV BytesIO ------------
def pcm16_to_wav_bytes(pcm_bytes: bytes, channels: int = 1, sampwidth: int = 2, framerate: int = 8000) -> io.BytesIO:
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(pcm_bytes)
    wav_io.seek(0)
    return wav_io

# ------------ Deepgram transcription (prerecorded) ------------
async def transcribe_audio_with_deepgram(pcm_bytes: bytes, sample_rate: int = 8000) -> str:
    if dg_client is None:
        print("Deepgram client not configured")
        return ""
    try:
        wav_io = pcm16_to_wav_bytes(pcm_bytes, channels=1, sampwidth=2, framerate=sample_rate)
        response = await dg_client.transcription.prerecorded(
            {"buffer": wav_io, "mimetype": "audio/wav"},
            {"punctuate": True},
        )
        # navigate response structure safely
        results = response.get("results", {})
        channels = results.get("channels", [])
        if channels and channels[0].get("alternatives"):
            transcript = channels[0]["alternatives"][0].get("transcript", "")
            return transcript
        return ""
    except Exception:
        traceback.print_exc()
        return ""

# ------------ OpenAI response generation ------------
async def generate_openai_response(user_text: str) -> str:
    if openai_client is None:
        print("OpenAI client not configured")
        return "Sorry, I am unable to respond right now."
    try:
        messages.append({"role": "user", "content": user_text})
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        # Extract reply safely
        reply = ""
        if hasattr(resp, "choices") and resp.choices:
            first = resp.choices[0]
            if getattr(first, "message", None):
                reply = first.message.content
        # Fallback extraction for dict-like responses
        if not reply and isinstance(resp, dict):
            try:
                reply = resp["choices"][0]["message"]["content"]
            except Exception:
                reply = ""
        if reply:
            messages.append({"role": "assistant", "content": reply})
        return reply or "Sorry, I did not understand that."
    except Exception:
        traceback.print_exc()
        return "Sorry, something went wrong while generating the response."

# ------------ ElevenLabs TTS and send back to Plivo ------------
async def synthesize_and_play(text: str, plivo_ws):
    if tts_client is None:
        print("ElevenLabs client not configured")
        return
    try:
        # ElevenLabs streaming response returns an iterator of bytes; this may vary by SDK version
        voice_id = DEFAULT_VOICE_ID
        stream = tts_client.text_to_speech.convert(
            voice_id=voice_id,
            output_format="ulaw_8000",
            text=text,
            model_id="eleven_turbo_v2_5",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        output = bytearray()
        # Stream may be sync iterator or async; handle both
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                if chunk:
                    output.extend(chunk)
        else:
            for chunk in stream:
                if chunk:
                    output.extend(chunk)

        # encode to base64 for Plivo
        encode = base64.b64encode(bytes(output)).decode("utf-8")
        payload = {
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
                "payload": encode,
            },
        }
        await plivo_ws.send(json.dumps(payload))
    except Exception:
        traceback.print_exc()

# ------------ WebSocket handler logic for Plivo connections ------------
# Replace the existing plivo_handler with this debug version
async def plivo_handler(plivo_ws, sample_rate: int = 8000, vad_mode: int = 1, silence_threshold_sec: float = 0.5):
    print("Plivo handler started for connection (verbose logging enabled)")
    vad = webrtcvad.Vad(vad_mode)

    inbuffer = bytearray()
    silence_accum = 0.0
    last_chunk_had_speech = False

    try:
        async for raw in plivo_ws:
            # Log raw incoming message length and a short preview
            try:
                preview = (raw[:200] + '...') if isinstance(raw, (bytes, bytearray)) else (str(raw)[:200] + '...')
            except Exception:
                preview = "<could not create preview>"
            try:
                print(f"[DEBUG] Received message length={len(raw) if hasattr(raw, '__len__') else 'unknown'} preview={preview}")
            except Exception:
                print("[DEBUG] Received a message (could not determine length)")

            try:
                data = json.loads(raw)
            except Exception as e:
                print(f"[DEBUG] Failed to parse JSON from incoming message: {e}")
                # continue to next message rather than aborting
                continue

            # Log event and media metadata
            event = data.get("event", "<no-event>")
            print(f"[DEBUG] Parsed event: {event}")

            if event == "media":
                media = data.get("media", {})
                payload_b64 = media.get("payload")
                if payload_b64 is None:
                    print("[DEBUG] media event received but no payload present")
                    continue

                try:
                    chunk = base64.b64decode(payload_b64)
                except Exception as e:
                    print(f"[DEBUG] Failed to base64-decode payload: {e}")
                    continue

                print(f"[DEBUG] media chunk decoded size={len(chunk)} bytes")
                inbuffer.extend(chunk)

                # VAD processing (20ms frames)
                frame_bytes = int(sample_rate * 2 * 0.02)  # 20ms frames
                has_speech = False
                for start in range(0, len(chunk), frame_bytes):
                    frame = chunk[start:start + frame_bytes]
                    if len(frame) < frame_bytes:
                        break
                    try:
                        if vad.is_speech(frame, sample_rate):
                            has_speech = True
                            break
                    except Exception as e:
                        print(f"[DEBUG] VAD raised exception on frame: {e}")
                        # ignore and proceed

                if has_speech:
                    last_chunk_had_speech = True
                    silence_accum = 0.0
                else:
                    silence_accum += 0.02
                    if last_chunk_had_speech and silence_accum >= silence_threshold_sec:
                        if len(inbuffer) > 2048:
                            print("[DEBUG] Silence threshold reached, transcribing buffered audio...")
                            try:
                                transcription = await transcribe_audio_with_deepgram(bytes(inbuffer), sample_rate=sample_rate)
                                print(f"[DEBUG] Transcription result: {transcription!r}")
                                if transcription:
                                    reply = await generate_openai_response(transcription)
                                    print(f"[DEBUG] OpenAI reply: {reply!r}")
                                    if reply:
                                        await synthesize_and_play(reply, plivo_ws)
                            except Exception as e:
                                print(f"[ERROR] Exception during transcription/response/synthesis: {e}")
                                traceback.print_exc()
                        inbuffer = bytearray()
                        silence_accum = 0.0
                        last_chunk_had_speech = False

            elif event == "stop":
                print("[DEBUG] Received stop event")
                if len(inbuffer) > 0:
                    try:
                        transcription = await transcribe_audio_with_deepgram(bytes(inbuffer), sample_rate=sample_rate)
                        print(f"[DEBUG] Final transcription before stop: {transcription!r}")
                        if transcription:
                            reply = await generate_openai_response(transcription)
                            if reply:
                                await synthesize_and_play(reply, plivo_ws)
                    except Exception as e:
                        print(f"[ERROR] Exception while handling stop event: {e}")
                        traceback.print_exc()
                break

            elif event == "start":
                print("[DEBUG] Received start event")

            else:
                print(f"[DEBUG] Unhandled event type: {event}")

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"[INFO] WebSocket connection closed: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected exception in plivo_handler: {e}")
        traceback.print_exc()
    finally:
        print("Plivo handler exiting for connection")

# ------------ Router and server bootstrap ------------
async def router(ws, path):
    if path == "/stream":
        print("Incoming connection on /stream")
        await plivo_handler(ws)
    else:
        # For any other path, optionally respond or close
        print(f"Unknown path {path}, closing connection")
        await ws.close()

def start_server(host: str, port: int):
    print(f"Starting websocket server on {host}:{port}")
    return websockets.serve(router, host, port)

# ------------ Entrypoint ------------
if __name__ == "__main__":
    # Use Render's PORT or fallback to 5000 for local dev
    try:
        PORT = int(os.getenv("PORT", os.getenv("port", "5000")))
    except Exception:
        PORT = 5000
    HOST = "0.0.0.0"

    loop = asyncio.get_event_loop()
    server = start_server(HOST, PORT)
    loop.run_until_complete(server)
    print("Server running. Awaiting Plivo connections...")
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Shutting down server")
    finally:
        loop.stop()
