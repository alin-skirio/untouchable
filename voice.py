"""Voice loop: listen for 'hey grok', then run the same computer tools as grok_agent.

Prefers Grok Voice realtime. If that endpoint rejects the key, falls back to
Grok speech-to-text plus the text tool loop that already works.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import signal
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import websockets
from dotenv import load_dotenv
from websockets.exceptions import InvalidStatus

from grok_agent import INSTRUCTIONS, TOOLS, _run_tool, run_instruction

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)

WAKE_PHRASE = "hey grok"
SAMPLE_RATE = 24000
STT_RATE = 16000
BLOCK_FRAMES = 2400
MODEL = "grok-voice-latest"
URL = f"wss://api.x.ai/v1/realtime?model={MODEL}"

VOICE_INSTRUCTIONS = (
    INSTRUCTIONS
    + "\nOnly call tools if this user turn includes the wake phrase "
    + '"hey grok". If it does not, do not call tools. '
    + "If they ask to simplify this page or make it easier to read, call simplify_page. "
    + "If the user says hey grok shut down, do not control the Mac; the listener will exit."
)


def contains_wake(text: str) -> bool:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return WAKE_PHRASE in cleaned


def strip_wake(text: str) -> str:
    cleaned = re.sub(r"(?i)hey\s+grok[,.!]?", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_command(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^(please|now)\s+", "", cleaned)
    return cleaned


def is_simplify_command(text: str) -> bool:
    """True for 'hey grok, simplify this' and close variants."""
    if not contains_wake(text):
        return False
    rest = _normalize_command(strip_wake(text))
    needles = (
        "simplify this",
        "simplify the page",
        "simplify this page",
        "make this easier",
        "easier to read",
        "easier to navigate",
        "simplify",
    )
    return any(needle in rest for needle in needles)


def is_exit_command(text: str) -> bool:
    """True for 'hey grok, shut down' — stops this program, not the Mac."""
    if not contains_wake(text):
        return False
    rest = _normalize_command(strip_wake(text))
    return rest in {
        "shut down",
        "shutdown",
        "shut down the program",
        "stop listening",
        "quit",
        "exit",
        "stop",
    }


def _api_key() -> str:
    key = os.getenv("XAI_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("No XAI_API_KEY found. Paste your key into .env.")
    return key


def _pcm16_b64(frames: np.ndarray) -> str:
    clipped = np.clip(frames, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode("ascii")


def _ws_connect(headers: dict):
    kwargs = {"max_size": None, "open_timeout": 20}
    params = inspect.signature(websockets.connect).parameters
    if "additional_headers" in params:
        kwargs["additional_headers"] = headers
    else:
        kwargs["extra_headers"] = headers
    return websockets.connect(URL, **kwargs)


def _error_body(exc: InvalidStatus) -> str:
    body = getattr(exc.response, "body", None)
    if isinstance(body, (bytes, bytearray)):
        return bytes(body).decode("utf-8", "replace")
    return str(exc)


async def run_realtime(key: str) -> None:
    audio_q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=50)
    turn_text = ""
    flush_task: asyncio.Task | None = None

    def on_audio(indata, _frames, _time, status) -> None:
        if status:
            print(status)
        try:
            audio_q.put_nowait(indata[:, 0].copy())
        except asyncio.QueueFull:
            pass

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_FRAMES,
        callback=on_audio,
    )
    headers = {"Authorization": f"Bearer {key}"}
    print('Listening on Grok Voice. Say "hey grok" then a command. Say "hey grok, shut down" to quit.')

    async with _ws_connect(headers) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "voice": "eve",
                        "instructions": VOICE_INSTRUCTIONS,
                        "turn_detection": {"type": "server_vad"},
                        "tools": TOOLS,
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                                "transcription": {
                                    "model": "grok-transcribe",
                                    "language_hint": "en",
                                    "keyterms": ["hey Grok", "Grok", "shut down", "simplify"],
                                },
                            },
                            "output": {
                                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                            },
                        },
                    },
                }
            )
        )

        stream.start()
        stop = asyncio.Event()

        def request_stop(_signum=None, _frame=None) -> None:
            stop.set()

        signal.signal(signal.SIGINT, request_stop)

        async def sender() -> None:
            while not stop.is_set():
                try:
                    frames = await asyncio.wait_for(audio_q.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": _pcm16_b64(frames),
                        }
                    )
                )

        async def wait_for_wake() -> bool:
            for _ in range(12):
                if contains_wake(turn_text):
                    return True
                await asyncio.sleep(0.08)
            return contains_wake(turn_text)

        async def handle_function_call(event: dict) -> None:
            nonlocal flush_task
            name = event.get("name") or ""
            call_id = event.get("call_id")
            raw = event.get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            armed = await wait_for_wake()
            if not armed:
                result = {
                    "ok": False,
                    "error": 'Wake phrase "hey grok" was not heard. Ignore this request.',
                }
                print(f"Ignored tool (no wake): {name}")
            else:
                print(f"Tool: {name} {json.dumps(args)}")
                result = _run_tool(name, args)
                print(f"Result: {json.dumps(result)}")

            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result),
                        },
                    }
                )
            )

            if flush_task is not None:
                flush_task.cancel()

            async def flush() -> None:
                await asyncio.sleep(0.2)
                await ws.send(json.dumps({"type": "response.create"}))

            flush_task = asyncio.create_task(flush())

        async def receiver() -> None:
            nonlocal turn_text
            async for message in ws:
                if stop.is_set():
                    break
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                kind = event.get("type") or ""

                if kind == "input_audio_buffer.speech_started":
                    turn_text = ""
                    print("\nHeard speech…")

                if kind in (
                    "conversation.item.input_audio_transcription.updated",
                    "conversation.item.input_audio_transcription.completed",
                    "conversation.item.input_audio_transcription.delta",
                ):
                    piece = event.get("transcript") or event.get("text") or ""
                    if piece:
                        if kind.endswith("delta"):
                            turn_text += piece
                        else:
                            turn_text = piece
                        mark = "wake" if contains_wake(turn_text) else "no wake"
                        print(f"You ({mark}): {turn_text}")
                        if is_exit_command(turn_text):
                            print("Shutting down the voice listener.")
                            stop.set()
                            break

                if kind == "response.function_call_arguments.done":
                    await handle_function_call(event)

                if kind in (
                    "response.output_audio_transcript.delta",
                    "response.audio_transcript.delta",
                ):
                    delta = event.get("delta") or event.get("text") or ""
                    if delta:
                        sys.stdout.write(delta)
                        sys.stdout.flush()

                if kind == "response.done":
                    print()

                if kind == "error":
                    print(f"Voice API error: {event}")

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        await stop.wait()
        sender_task.cancel()
        receiver_task.cancel()

    stream.stop()
    stream.close()


def _record_utterance(seconds_silence: float = 1.1) -> np.ndarray:
    print("Speak now…")
    chunks: list[np.ndarray] = []
    started = False
    quiet_for = 0.0
    block = 0.1
    with sd.InputStream(samplerate=STT_RATE, channels=1, dtype="float32", blocksize=int(STT_RATE * block)) as stream:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            frames, _overflow = stream.read(int(STT_RATE * block))
            sample = frames[:, 0]
            level = float(np.sqrt(np.mean(np.square(sample))))
            if level > 0.02:
                started = True
                quiet_for = 0.0
                chunks.append(sample.copy())
            elif started:
                quiet_for += block
                chunks.append(sample.copy())
                if quiet_for >= seconds_silence:
                    break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def _transcribe(key: str, audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(STT_RATE)
        wav.writeframes(pcm_bytes)
    try:
        import httpx

        with open(path, "rb") as handle:
            response = httpx.post(
                "https://api.x.ai/v1/stt",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("speech.wav", handle, "audio/wav")},
                timeout=60.0,
            )
        if response.status_code >= 400:
            raise RuntimeError(response.text)
        payload = response.json()
        return str(payload.get("text") or payload.get("transcript") or "").strip()
    finally:
        os.unlink(path)


def run_stt_fallback(key: str) -> None:
    print(
        "Grok Voice realtime is not available on this key. "
        'Falling back to Grok speech-to-text plus tools. Say "hey grok" then a command.'
    )
    print('Say "hey grok, shut down" to quit.')
    while True:
        audio = _record_utterance()
        if audio.size == 0:
            print("Did not hear speech. Try again.")
            continue
        text = _transcribe(key, audio)
        print(f"You: {text or '(empty)'}")
        if not contains_wake(text):
            print('Need the wake phrase "hey grok" first.')
            continue
        if is_exit_command(text):
            print("Shutting down the voice listener.")
            return
        if is_simplify_command(text):
            from simplify import simplify_current_page

            command = strip_wake(text) or "Simplify this page."
            print("Simplifying the current browser page…")
            result = simplify_current_page(command)
            print(f"Grok said: {result}")
            continue
        command = strip_wake(text) or text
        reply = run_instruction(command)
        print(f"Grok said: {reply}")


def main() -> None:
    key = _api_key()
    try:
        asyncio.run(run_realtime(key))
    except InvalidStatus as exc:
        detail = _error_body(exc)
        print(f"Grok Voice realtime rejected the connection: {detail}")
        run_stt_fallback(key)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped listening.")
    except Exception as exc:
        print(f"Voice loop failed: {exc}")
        sys.exit(1)
