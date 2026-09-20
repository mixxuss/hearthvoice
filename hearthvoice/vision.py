"""Letting the assistant look at something.

This is the third data type the project needs: audio in, text in the middle,
and an image here. It is also an accessibility feature in its own right. A
visually impaired user can hold a package up to the camera and ask what it is,
which is something no amount of speech recognition will do for them.

The same Qwen3.5 instance on the Spark that handles the conversation handles
the image, so sight costs no extra model and no extra hardware, and nothing
leaves the local network.

The earlier version of this project had a vision stage that was never local at
all. It built its own OpenAI client and ignored the configured endpoint, so it
had only ever run against a vendor while the report said otherwise. This one
goes through the same base URL as every other model call, so if the
configuration is local then the vision is local.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

from openai import OpenAI

from .config import LLM_BASE_URL, LLM_MODEL

log = logging.getLogger(__name__)

CAPTURE_PATH = Path(os.getenv("EVAL_DIR", "eval")) / "vision_capture.jpg"
FALLBACK_IMAGE = os.getenv("VISION_IMAGE")

# Everything this returns is spoken aloud, so the model has to be told to
# write for the ear. Left to itself it answers with bold text, bullet points
# and a warning emoji, all of which a speech synthesiser reads out literally.
PROMPT = (
    "You are the eyes of someone who cannot see this. Answer in one or two "
    "short plain sentences, as if speaking to them. If there is readable text "
    "such as a label, a dosage or an expiry date, say that first, because it "
    "is usually why they asked. Never use markdown, bullet points, asterisks, "
    "headings or emoji: every character you write will be read aloud. Do not "
    "describe the lighting or the background."
)


class NoImage(RuntimeError):
    """Nothing to look at: no camera, and no fallback image configured."""


def capture() -> bytes:
    """A frame from the camera, or the configured fallback image.

    macOS grants camera permission per application, and a terminal that has not
    been granted it fails rather than prompting. The fallback keeps the feature
    testable on a machine where permission has not been given.
    """
    frame = _from_camera()
    if frame is not None:
        CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CAPTURE_PATH.write_bytes(frame)
        return frame

    if FALLBACK_IMAGE and Path(FALLBACK_IMAGE).is_file():
        log.info("no camera, using %s", FALLBACK_IMAGE)
        return Path(FALLBACK_IMAGE).read_bytes()

    raise NoImage(
        "I cannot see anything at the moment. The camera is unavailable, which "
        "on this machine usually means the terminal has not been given camera "
        "permission."
    )


def _from_camera() -> bytes | None:
    try:
        import cv2
    except ImportError:
        return None

    # macOS lists every camera it knows about, and on a Mac with an iPhone
    # nearby the phone's Continuity Camera often sits at index 0 whether or
    # not it is actually available. Opening it "succeeds" and returns black
    # frames, so a fixed VideoCapture(0) reported no camera on a machine with
    # a perfectly good one built in. Try each index in turn and keep the first
    # that produces a frame with something in it. VISION_CAMERA pins one.
    pinned = os.getenv("VISION_CAMERA")
    candidates = [int(pinned)] if pinned else [0, 1, 2]

    for index in candidates:
        camera = cv2.VideoCapture(index)
        try:
            if not camera.isOpened():
                continue
            # The first frame from a cold camera is often black while the
            # exposure settles, so take several and keep the last.
            image = None
            for _ in range(6):
                ok, frame = camera.read()
                if ok:
                    image = frame
            if image is None or float(image.mean()) < 8.0:
                log.info("camera %d gave no usable frame, trying the next", index)
                continue
            ok, encoded = cv2.imencode(".jpg", image)
            if ok:
                log.info("using camera %d", index)
                return encoded.tobytes()
        finally:
            camera.release()
    return None


def _speakable(text: str) -> str:
    """Strip anything a speech synthesiser would read out as punctuation."""
    import re

    text = re.sub(r"[*_#`]+", "", text)
    text = re.sub(r"^\s*[-\u2022]\s*", "", text, flags=re.MULTILINE)
    text = "".join(c for c in text if c.isascii())
    return " ".join(text.split()).strip()


def _ask(question: str | None) -> str:
    if not question:
        return PROMPT
    return f"{PROMPT}\n\nThey asked: {question}"


def describe(image: bytes | None = None, question: str | None = None) -> str:
    """Ask the local model what it is looking at."""
    data = image if image is not None else capture()
    encoded = base64.b64encode(data).decode()

    client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": [
                # The style rules always apply. An earlier version passed the
                # user's question INSTEAD of the prompt, which dropped them and
                # brought the bullet points back.
                {"type": "text", "text": _ask(question)},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }],
        max_tokens=120,
        temperature=0.0,
    )
    answer = _speakable(response.choices[0].message.content or "")
    log.info("vision: %s", answer)
    return answer or "I could not make out what that is."
