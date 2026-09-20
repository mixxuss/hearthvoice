"""Talking to the three models on the DGX Spark.

The agent reaches these through pipecat. Everything else, meaning the
evaluation harness and the live test, calls the functions here, so there is one
definition of how each model is called and how each call is timed.
"""

from __future__ import annotations

import array
import json
import time
from typing import Callable

import riva.client
from openai import OpenAI

from .config import ASR_SERVER, LLM_BASE_URL, LLM_MODEL, TTS_SERVER, TTS_VOICE

TTS_RATE = 44100
ASR_RATE = 16000


def resample(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linear resample of 16-bit mono audio.

    Magpie speaks at 44.1 kHz and Parakeet only accepts 16 kHz, so audio going
    from one to the other passes through here.
    """
    if source_rate == target_rate:
        return pcm

    samples = array.array("h")
    samples.frombytes(pcm)
    step = source_rate / target_rate
    out = array.array("h")

    for i in range(int(len(samples) / step)):
        position = i * step
        low = int(position)
        high = min(low + 1, len(samples) - 1)
        weight = position - low
        out.append(int(samples[low] * (1 - weight) + samples[high] * weight))

    return out.tobytes()


def _auth(server: str) -> riva.client.Auth:
    """Self-hosted Riva wants no TLS, no cloud function id and no API key."""
    return riva.client.Auth(None, False, server, [])


def speak(text: str) -> tuple[bytes, float]:
    """Synthesise speech. Returns the audio and the time to the first chunk."""
    service = riva.client.SpeechSynthesisService(_auth(TTS_SERVER))
    started = time.perf_counter()
    first_chunk = None
    audio = bytearray()

    for response in service.synthesize_online(
        text=text,
        voice_name=TTS_VOICE,
        language_code="en-US",
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=TTS_RATE,
    ):
        if response.audio:
            if first_chunk is None:
                first_chunk = time.perf_counter()
            audio.extend(response.audio)

    return bytes(audio), round((first_chunk or started) - started, 3)


def transcribe(audio_44k: bytes) -> tuple[str, float]:
    """Recognise speech. Returns the transcript and how long it took.

    Parakeet on this server is streaming only, so the audio goes in as 100 ms
    chunks rather than as one blob.
    """
    pcm = resample(audio_44k, TTS_RATE, ASR_RATE)
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            sample_rate_hertz=ASR_RATE,
            audio_channel_count=1,
        ),
        interim_results=False,
    )

    chunk_size = ASR_RATE * 2 // 10
    chunks = [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]

    service = riva.client.ASRService(_auth(ASR_SERVER))
    started = time.perf_counter()
    heard = []

    for response in service.streaming_response_generator(
        audio_chunks=iter(chunks), streaming_config=config
    ):
        for result in response.results:
            if result.alternatives and result.is_final:
                heard.append(result.alternatives[0].transcript)

    return "".join(heard).strip(), round(time.perf_counter() - started, 3)


def decide(
    system_prompt: str, transcript: str, tools: list[dict]
) -> tuple[list[dict], str, float]:
    """Ask the model what to do.

    Returns the tool calls it wants, anything it said in plain text, and the
    time to the first token. Temperature is zero so that re-running the
    evaluation gives the same answer and the report's numbers can be reproduced.
    """
    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    started = time.perf_counter()
    first_token = None
    partial: dict[int, dict] = {}
    said = ""

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        tools=tools,
        temperature=0.0,
        stream=True,
    )

    for chunk in stream:
        if first_token is None:
            first_token = time.perf_counter()
        delta = chunk.choices[0].delta
        if delta.content:
            said += delta.content
        # Tool calls arrive split across chunks and have to be stitched back
        # together by their index before the arguments will parse as JSON.
        for call in delta.tool_calls or []:
            slot = partial.setdefault(call.index, {"name": "", "args": ""})
            if call.function and call.function.name:
                slot["name"] = call.function.name
            if call.function and call.function.arguments:
                slot["args"] += call.function.arguments

    calls = [
        {"name": slot["name"], "arguments": _parse_args(slot["args"])}
        for slot in partial.values()
    ]
    return calls, said, round((first_token or started) - started, 3)


def respond(system_prompt: str, transcript: str, tools: list[dict],
            calls: list[dict], results: list[str],
            run: Callable[[str, dict], str] | None = None,
            max_rounds: int = 3) -> tuple[str, list[dict], list[str], float]:
    """The reply the user actually hears, once the tools have run.

    The model's first response is produced before it knows whether a tool
    succeeded, so scoring that text cannot catch a reply that contradicts what
    happened. The live agent always makes this second call; an earlier version
    of the harness did not, which is how a scenario that left a door unlocked
    while announcing it had locked it scored as a success.

    This keeps going while the model asks for more tools. A sentence with two
    actions in it often arrives as two rounds rather than two calls: the model
    turns the lights off, sees that it worked, and only then reaches for the
    lock. The first version of this function collected text and dropped any
    further tool calls on the floor, so the second action never ran and, since
    the model had produced no text either, the agent said nothing at all. A
    half-executed command delivered in silence is the worst outcome available
    to a user who cannot see the lights, and it was two of the three
    severity-4 findings in section 5.6.

    `run` executes a tool and returns what to tell the model. Leave it out and
    the function behaves as it used to, one round and no follow-up.

    Returns the reply, every tool call made here, their results, and the time
    to the first token of the final answer.
    """
    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]
    messages += _tool_exchange(calls, results, offset=0)

    extra_calls: list[dict] = []
    extra_results: list[str] = []
    started = time.perf_counter()
    first_token = None

    for _ in range(max_rounds):
        said, more, first = _one_round(client, messages, tools)
        if first_token is None:
            first_token = first
        if said or not more or run is None:
            return (said.strip(), extra_calls, extra_results,
                    round((first_token or started) - started, 3))

        # The model wants another tool before it will say anything.
        outcomes = [run(call["name"], call["arguments"]) for call in more]
        extra_calls += more
        extra_results += outcomes
        messages += _tool_exchange(more, outcomes, offset=len(messages))

    # Out of rounds. Saying nothing is not an option, so describe what was done.
    done = "; ".join(results + extra_results) or "nothing"
    return (f"I did this: {done}.", extra_calls, extra_results,
            round((first_token or started) - started, 3))


def _tool_exchange(calls: list[dict], results: list[str],
                   offset: int) -> list[dict]:
    """One assistant turn of tool calls, followed by each result."""
    ids = [f"c{offset}_{i}" for i in range(len(calls))]
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": cid, "type": "function",
             "function": {"name": c["name"],
                          "arguments": json.dumps(c["arguments"])}}
            for cid, c in zip(ids, calls)
        ],
    }
    return [assistant] + [
        {"role": "tool", "tool_call_id": cid, "content": result}
        for cid, result in zip(ids, results)
    ]


def _one_round(client, messages: list[dict],
               tools: list[dict]) -> tuple[str, list[dict], float | None]:
    """One streamed completion. Returns its text, its tool calls, and the
    moment the first token arrived."""
    first_token = None
    said = ""
    partial: dict[int, dict] = {}

    stream = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, tools=tools,
        temperature=0.0, stream=True,
    )
    for chunk in stream:
        if first_token is None:
            first_token = time.perf_counter()
        delta = chunk.choices[0].delta
        if delta.content:
            said += delta.content
        for call in delta.tool_calls or []:
            slot = partial.setdefault(call.index, {"name": "", "args": ""})
            if call.function and call.function.name:
                slot["name"] = call.function.name
            if call.function and call.function.arguments:
                slot["args"] += call.function.arguments

    calls = [{"name": slot["name"], "arguments": _parse_args(slot["args"])}
             for slot in partial.values() if slot["name"]]
    return said, calls, first_token


def _parse_args(raw: str) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
