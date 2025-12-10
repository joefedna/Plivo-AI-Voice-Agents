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
import audioop

from deepgram import Deepgram
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

# ---------------- Config loader (env first, fallback to config.json) ----------------
def load_config(file_path: str = "config.json"):
    cfg = {
        "PLIVO_AUTH_ID": os.getenv("PLIVO_AUTH_ID"),
        "PLIVO_AUTH_TOKEN": os.getenv("PLIVO_AUTH_TOKEN"),
        "DEEPGRAM_API_KEY": os.getenv("DEEPGRAM_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
        "VOICE_ID": os.getenv("VOICE_ID"),
    }

    missing = [k for k, v in cfg.items() if v is None]
    if missing and os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                disk = json.load(f)
            for k in cfg:
                if not cfg[k]:
                    val = disk.get(k.lower())
                    if val:
                        cfg[k] = val
        except Exception:
            traceback.print_exc()

    return cfg

CONFIG = load_config()

if not CONFIG.get("DEEPGRAM_API_KEY"):
    print("Warning: DEEPGRAM_API_KEY not found")
if not CONFIG.get("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found")
if not CONFIG.get("ELEVENLABS_API_KEY"):
    print("Warning: ELEVENLABS_API_KEY not found")

# ---------------- Clients ----------------
dg_client = Deepgram(CONFIG.get("DEEPGRAM_API_KEY")) if CONFIG.get("DEEPGRAM_API_KEY") else None
openai_client = OpenAI(api_key=CONFIG.get("OPENAI_API_KEY")) if CONFIG.get("OPENAI_API_KEY") else None
tts_client = ElevenLabs(api_key=CONFIG.get("ELEVENLABS_API_KEY")) if CONFIG.get("ELEVENLABS_API_KEY") else None

DEFAULT_VOICE_ID = CONFIG.get("VOICE_ID") or "XrExE9yKIg1WjnnlVkGX"

# ---------------- Conversation memory ----------------
messages = []
system_msg = (
    "Your name is Matilda. Matilda is a warm and friendly voicebot designed to have pleasant and engaging "
    "conversations with customers. Matilda's primary purpose is to greet customers in a cheerful and polite manner "
    "whenever they say 'hello' or any other greeting. She should respond with kindness, using a welcoming tone to make "
    "the customer feel valued and appreciated. Keep responses concise and conversational."
)
messages.append({"role": "system", "content": system_msg})

# ---------------- Utilities ----------------
def pcm16_to_wav_bytes(pcm_bytes: bytes, channels: int = 1, sampwidth: int = 2, framerate: int = 8000) -> io.BytesIO:
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(pcm_bytes)
    wav_io.seek(0)
    return wav_io

def log_api_error(api_name: str, exc: Exception):
    print(f"[API ERROR] {api_name} failed: {exc}")
    traceback.print_exc()

# ---------------- API health checks ----------------
async def run_api_health_checks(timeout_sec: float = 6.0):
    results = {}
    # Deepgram health
    try:
        if dg_client:
            silent_pcm = (b"\x00\x00" * 160)  # 20ms silence at 8kHz
            wav_io = pcm16_to_wav_bytes(silent_pcm, channels=1, sampwidth=2, framerate=8000)
            resp = await dg_client.transcription.prerecorded({"buffer": wav_io, "mimetype": "audio/wav"}, {"punctuate": False})
            results["deepgram"] = "ok" if resp else "no_response"
        else:
            results["deepgram"] = "not_configured"
    except Exception as e:
        results["deepgram"] = f"error: {str(e)}"
        log_api_error("Deepgram", e)

    # OpenAI health
    try:
        if openai_client:
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "healthcheck"}, {"role": "user", "content": "hi"}],
                    max_tokens=1
                )
                results["openai"] = "ok" if resp else "no_response"
            except Exception as e:
                results["openai"] = f"error: {str(e)}"
                log_api_error("OpenAI", e)
        else:
            results["openai"] = "not_configured"
    except Exception as e:
        results["openai"] = f"error: {str(e)}"
        log_api_error("OpenAI", e)

    # ElevenLabs health
    try:
        if tts_client:
            try:
                stream = tts_client.text_to_speech.convert(
                    voice_id=DEFAULT_VOICE_ID,
                    output_format="ulaw_8000",
                    text="test",
                    model_id="eleven_turbo_v2_5",
                    voice_settings=VoiceSettings(stability=0.0, similarity_boost=0.0, style=0.0)
                )
                first_chunk = None
                if hasattr(stream, "__aiter__"):
                    async for c in stream:
                        first_chunk = c
                        break
                else:
                    for c in stream:
                        first_chunk = c
                        break
                results["elevenlabs"] = "ok" if first_chunk else "no_audio"
            except Exception as e:
                results["elevenlabs"] = f"error: {str(e)}"
                log_api_error("ElevenLabs", e)
        else:
            results["elevenlabs"] = "not_configured"
    except Exception as e:
        results["elevenlabs"] = f"error: {str(e)}"
        log_api_error("ElevenLabs", e)

    print(f"[API HEALTH] {results}")
    return results

# ---------------- Transcription ----------------
async def transcribe_audio_with_deepgram(pcm_bytes: bytes, sample_rate: int = 8000) -> str:
    if dg_client is None:
        print("[Deepgram] client not configured")
        return ""
    try:
        wav_io = pcm16_to_wav_bytes(pcm_bytes, channels=1, sampwidth=2, framerate=sample_rate)
        response = await dg_client.transcription.prerecorded(
            {"buffer": wav_io, "mimetype": "audio/wav"},
            {"punctuate": True},
        )
        try:
            results = response.get("results", {})
            channels = results.get("channels", [])
            if channels and channels[0].get("alternatives"):
                transcript = channels[0]["alternatives"][0].get("transcript", "")
                # If empty transcript, log the full response for debugging
                if not transcript:
                    print("[Deepgram] Empty transcript returned. Full response:")
                    print(json.dumps(response)[:4000])  # avoid ultra-long logs; prints first 4k chars
                return transcript
        except Exception:
            print("[Deepgram] Unexpected response structure:")
            print(response)
            return ""
        return ""
    except Exception as e:
        log_api_error("Deepgram", e)
        return ""

# ---------------- OpenAI generation ----------------
async def generate_openai_response(user_text: str) -> str:
    if openai_client is None:
        print("[OpenAI] client not configured")
        return "Sorry, I am unable to respond right now."
    try:
        messages.append({"role": "user", "content": user_text})
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = ""
        if hasattr(resp, "choices") and resp.choices:
            first = resp.choices[0]
            if getattr(first, "message", None):
                reply = first.message.content
        if not reply and isinstance(resp, dict):
            try:
                reply = resp["choices"][0]["message"]["content"]
            except Exception:
                reply = ""
        if reply:
            messages.append({"role": "assistant", "content": reply})
        return reply or "Sorry, I did not understand that."
    except Exception as e:
        log_api_error("OpenAI", e)
        return "Sorry, something went wrong while generating the response."

# ---------------- ElevenLabs TTS ----------------
async def synthesize_and_play(text: str, plivo_ws):
    """
    Convert text to ulaw_8000 via ElevenLabs (or configured TTS), then send to Plivo in
    multiple playAudio events with smaller payloads to avoid large single-message failures.
    """
    if tts_client is None:
        print("[ElevenLabs] client not configured")
        return

    try:
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

        # collect the audio bytes (stream may be async or sync iterator)
        output = bytearray()
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                if chunk:
                    output.extend(chunk)
        else:
            for chunk in stream:
                if chunk:
                    output.extend(chunk)

        if not output:
            print("[ElevenLabs] TTS produced empty audio")
            return

        # Base64 encode once, then chunk the base64 string for safe sending
        b64 = base64.b64encode(bytes(output)).decode("utf-8")

        # Determine chunk size: pick small enough to avoid websocket frame issues.
        # 4096 base64 chars ≈ 3 KB binary. This is conservative.
        CHUNK_SIZE = 4096

        total = len(b64)
        sent_bytes = 0
        idx = 0

        # Send chunks sequentially as separate playAudio events.
        while idx < total:
            part = b64[idx: idx + CHUNK_SIZE]
            payload = {
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-mulaw",
                    "sampleRate": 8000,
                    "payload": part,
                    # Optional: explicit outbound track hint
                    "track": "outbound"
                }
            }

            try:
                await plivo_ws.send(json.dumps(payload))
                sent_bytes += len(part)
            except websockets.exceptions.ConnectionClosedError as e:
                # Connection closed while sending. Log and abort playback.
                print(f"[API ERROR] ElevenLabs/playAudio send failed: {e}")
                traceback.print_exc()
                return
            except Exception as e:
                # Other send errors
                print(f"[API ERROR] Unexpected error while sending playAudio chunk: {e}")
                traceback.print_exc()
                return

            # small yield to avoid flooding and allow peer to process
            await asyncio.sleep(0.04)
            idx += CHUNK_SIZE

        print(f"[PLAYBACK] Sent audio payload total_base64_chars={total} sent_parts={(total + CHUNK_SIZE - 1)//CHUNK_SIZE} bytes_unencoded={len(output)}")

    except Exception as e:
        log_api_error("ElevenLabs", e)

# ---------------- plivo_handler (aggregated logging, keepalive, health checks) ----------------
async def plivo_handler(plivo_ws, sample_rate: int = 8000, vad_mode: int = 1, silence_threshold_sec: float = 0.5):
    print("Plivo handler started for connection (mu-law -> PCM16 conversion + aggregated logging)")
    # quick health checks (non-blocking)
    try:
        task = asyncio.create_task(run_api_health_checks())
        await asyncio.wait_for(task, timeout=6.0)
    except asyncio.TimeoutError:
        print("[API HEALTH] health check timed out")
    except Exception as e:
        print(f"[API HEALTH] health check error: {e}")

    vad = webrtcvad.Vad(vad_mode)
    inbuffer = bytearray()          # will hold PCM16 bytes
    silence_accum = 0.0
    last_chunk_had_speech = False

    media_counter = 0
    last_log_time = asyncio.get_event_loop().time()

    async def ping_loop():
        try:
            while True:
                await asyncio.sleep(10)
                try:
                    await plivo_ws.ping()
                except Exception:
                    break
        except asyncio.CancelledError:
            pass

    ping_task = asyncio.create_task(ping_loop())

    try:
        async for raw in plivo_ws:
            try:
                data = json.loads(raw)
            except Exception:
                continue

            event = data.get("event", "")
            if event == "media":
                media_counter += 1
                media = data.get("media", {})
                payload_b64 = media.get("payload")
                if not payload_b64:
                    continue

                try:
                    mulaw_chunk = base64.b64decode(payload_b64)
                except Exception:
                    continue

                # convert mu-law (u-law) bytes to 16-bit PCM (linear) for VAD and Deepgram
                try:
                    # audioop.ulaw2lin(input_bytes, width) -> linear bytes where width is target sample width in bytes (2 for 16-bit)
                    pcm16_chunk = audioop.ulaw2lin(mulaw_chunk, 2)
                except Exception as e:
                    print(f"[ERROR] mu-law -> PCM conversion failed: {e}")
                    continue

                # append PCM16 bytes into buffer (consistent format for VAD and Deepgram)
                inbuffer.extend(pcm16_chunk)

                # VAD expects 16-bit frames; compute 20ms frame size in bytes: sample_rate * bytes_per_sample * 0.02
                frame_bytes = int(sample_rate * 2 * 0.02)  # 2 bytes per sample (16-bit), 20ms frames
                has_speech = False
                # loop frames inside the PCM chunk (not mu-law)
                for start in range(0, len(pcm16_chunk), frame_bytes):
                    frame = pcm16_chunk[start:start + frame_bytes]
                    if len(frame) < frame_bytes:
                        break
                    try:
                        if vad.is_speech(frame, sample_rate):
                            has_speech = True
                            break
                    except Exception:
                        # ignore VAD frame errors
                        pass

                if has_speech:
                    # log first detection event so it's visible in logs
                    if not last_chunk_had_speech:
                        print(f"[VAD] speech detected (media_chunks={media_counter})")
                    last_chunk_had_speech = True
                    silence_accum = 0.0
                else:
                    silence_accum += 0.02
                    if last_chunk_had_speech and silence_accum >= silence_threshold_sec:
                        # end of utterance — process PCM16 buffer
                        if len(inbuffer) > 2048:
                            print(f"[MEDIA] End of utterance. media_chunks={media_counter} buffer_bytes={len(inbuffer)}")
                            try:
                                transcription = await transcribe_audio_with_deepgram(bytes(inbuffer), sample_rate=sample_rate)
                                print(f"[TRANSCRIBE] {transcription!r}")
                                if transcription:
                                    reply = await generate_openai_response(transcription)
                                    print(f"[OPENAI] reply_len={len(reply) if reply else 0}")
                                    if reply:
                                        await synthesize_and_play(reply, plivo_ws)
                            except Exception as e:
                                print(f"[PROCESS ERROR] {e}")
                                traceback.print_exc()
                        # reset
                        inbuffer = bytearray()
                        silence_accum = 0.0
                        last_chunk_had_speech = False
                        media_counter = 0

                # periodic compact log: once per second
                now = asyncio.get_event_loop().time()
                if now - last_log_time >= 1.0:
                    print(f"[MEDIA STATS] chunks={media_counter} buffer_bytes={len(inbuffer)} last_has_speech={last_chunk_had_speech}")
                    last_log_time = now

            elif event == "stop":
                print("[EVENT] stop received")
                if len(inbuffer) > 0:
                    try:
                        transcription = await transcribe_audio_with_deepgram(bytes(inbuffer), sample_rate=sample_rate)
                        print(f"[TRANSCRIBE-final] {transcription!r}")
                        if transcription:
                            reply = await generate_openai_response(transcription)
                            if reply:
                                await synthesize_and_play(reply, plivo_ws)
                    except Exception as e:
                        print(f"[PROCESS ERROR] {e}")
                        traceback.print_exc()
                break

            elif event == "start":
                print("[EVENT] start received")

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"[INFO] WebSocket closed with error: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected exception in plivo_handler: {e}")
        traceback.print_exc()
    finally:
        if not ping_task.done():
            ping_task.cancel()
            try:
                await ping_task
            except Exception:
                pass
        close_code = getattr(plivo_ws, "close_code", None)
        close_reason = getattr(plivo_ws, "close_reason", None)
        print(f"[INFO] plivo_ws close_code={close_code} close_reason={close_reason}")
        print("Plivo handler exiting for connection")


# ---------------- Router and server (with /health HTTP endpoint) ----------------
async def router(ws, path):
    if path == "/stream":
        print("Incoming connection on /stream")
        await plivo_handler(ws)
    else:
        print(f"Unknown path {path}, closing")
        await ws.close()

# HTTP handler for simple health checks on the same port
async def process_request(path, request_headers):
    """
    Handle simple HTTP requests on the same port as the WebSocket server.
    Return (status, headers, body) for HTTP responses, or None to continue websocket handshake.
    """
    if path == "/health":
        try:
            results = await run_api_health_checks()
        except Exception as e:
            results = {"error": str(e)}
        body = json.dumps(results).encode("utf-8")
        headers = [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ]
        return 200, headers, body
    return None

def start_server(host: str, port: int):
    """
    Start websockets server that also responds to HTTP /health via process_request.
    """
    print(f"Starting websocket server on {host}:{port} (with /health HTTP endpoint)")
    return websockets.serve(router, host, port, process_request=process_request)

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
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
