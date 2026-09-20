"""Acoustic echo cancellation, so the agent stops hearing itself.

On a laptop the speakers sit centimetres from the microphone, so the agent's
own reply comes straight back in. The recogniser turns that into a transcript,
the transcript counts as a new user turn, and a new turn cancels the reply in
progress, so the agent appears to stop talking part way through a sentence.

Ignoring input while the agent speaks avoids the problem by refusing to listen.
Echo cancellation solves it: subtract the audio being played from the audio
being heard, and what remains is the room.

This uses SpeexDSP through ctypes, which needs only the shared library rather
than a compiled Python extension. The approach is adapted from the Olegos audio
kit's `algorithms/speex.py`.

SpeexDSP reference: https://www.speex.org/docs/manual/speex-manual/node7.html

The awkward part is delay. SpeexDSP assumes the reference and the microphone
are aligned and has no delay estimator of its own. Between writing a frame to
the speaker and the microphone hearing it there is a real gap, so the reference
is buffered and the filter is given a tail long enough to cover the
misalignment. Getting this wrong makes the canceller do nothing rather than
fail loudly, which is the sort of quiet failure this project has hit before, so
`echo_reduction_db` reports whether it is actually doing anything.
"""

from __future__ import annotations

import array
import ctypes
import ctypes.util
import logging
import math
import os
from collections import deque
from pathlib import Path

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter

log = logging.getLogger(__name__)

# SPEEX_ECHO_SET_SAMPLING_RATE, from speex_echo.h
SET_SAMPLING_RATE = 24

FRAME_MS = 10
DEFAULT_TAIL_MS = 300


class SpeexUnavailable(RuntimeError):
    """The SpeexDSP shared library could not be loaded."""


def _load_library() -> tuple[ctypes.CDLL, str]:
    """Find libspeexdsp, preferring an explicitly configured path."""
    explicit = os.environ.get("SPEEX_LIB")
    candidates = [explicit] if explicit else [
        str(Path.home() / ".local/hearthvoice/speexdsp/lib/libspeexdsp.dylib"),
        ctypes.util.find_library("speexdsp"),
        "/opt/homebrew/lib/libspeexdsp.dylib",
        "/usr/local/lib/libspeexdsp.dylib",
        "libspeexdsp.so.1",
        "libspeexdsp.dylib",
    ]

    for path in dict.fromkeys(p for p in candidates if p):
        try:
            library = ctypes.CDLL(path)
            # libspeex, the codec, will load happily but has none of these.
            for symbol in ("speex_echo_state_init", "speex_echo_cancellation",
                           "speex_echo_state_destroy", "speex_echo_ctl"):
                getattr(library, symbol)
            return library, path
        except (OSError, AttributeError):
            continue

    raise SpeexUnavailable(
        "libspeexdsp not found. Build it, or set SPEEX_LIB to its path."
    )


class ReferenceTap:
    """Holds recently played audio for the canceller to subtract.

    The agent's voice reaches the microphone a little after it is written to
    the speaker, so the reference has to be buffered rather than used the
    instant it is produced. When nothing is playing this hands back silence,
    which makes the canceller a no-op, which is correct when there is no echo.
    """

    def __init__(self, max_frames: int = 100) -> None:
        self._frames: deque[bytes] = deque(maxlen=max_frames)

    def add(self, pcm: bytes) -> None:
        self._frames.append(pcm)

    def take(self, size: int) -> bytes:
        """The oldest played audio not yet used as a reference."""
        if not self._frames:
            return b"\x00" * size

        chunk = self._frames.popleft()
        if len(chunk) < size:
            return chunk + b"\x00" * (size - len(chunk))
        if len(chunk) > size:
            self._frames.appendleft(chunk[size:])
            return chunk[:size]
        return chunk

    def clear(self) -> None:
        self._frames.clear()

    @property
    def waiting(self) -> int:
        return len(self._frames)


class SpeexEchoFilter(BaseAudioFilter):
    """Subtracts the agent's own voice from the microphone signal."""

    def __init__(self, reference: ReferenceTap,
                 tail_ms: int = DEFAULT_TAIL_MS) -> None:
        self._reference = reference
        self._tail_ms = tail_ms
        self._library: ctypes.CDLL | None = None
        self._state: int | None = None
        self._frame_bytes = 0
        self._enabled = True
        self._mic_energy = 0.0
        self._out_energy = 0.0
        self._frames_seen = 0

    async def start(self, sample_rate: int) -> None:
        try:
            self._library, path = _load_library()
        except SpeexUnavailable as e:
            log.warning("echo cancellation off: %s", e)
            self._enabled = False
            return

        frame_size = int(sample_rate * FRAME_MS / 1000)
        tail = int(sample_rate * self._tail_ms / 1000)
        self._frame_bytes = frame_size * 2

        pointer = ctypes.POINTER(ctypes.c_int16)
        self._library.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        self._library.speex_echo_state_init.restype = ctypes.c_void_p
        self._library.speex_echo_cancellation.argtypes = [
            ctypes.c_void_p, pointer, pointer, pointer
        ]
        self._library.speex_echo_cancellation.restype = None
        self._library.speex_echo_ctl.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        ]
        self._library.speex_echo_ctl.restype = ctypes.c_int
        self._library.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]

        self._state = self._library.speex_echo_state_init(frame_size, tail)
        if not self._state:
            log.warning("echo cancellation off: speex_echo_state_init failed")
            self._enabled = False
            return

        rate = ctypes.c_int(sample_rate)
        self._library.speex_echo_ctl(self._state, SET_SAMPLING_RATE,
                                     ctypes.byref(rate))
        log.info("echo cancellation on: %s, %d ms tail at %d Hz",
                 Path(path).name, self._tail_ms, sample_rate)

    async def stop(self) -> None:
        if self._state and self._library:
            self._library.speex_echo_state_destroy(self._state)
            self._state = None
        log.info("echo cancellation: %s", self.summary())

    async def process_frame(self, frame) -> None:
        """No runtime controls. The filter is either on or it is not."""

    async def filter(self, audio: bytes) -> bytes:
        if not self._enabled or not self._state:
            return audio

        cleaned = bytearray()
        for start in range(0, len(audio), self._frame_bytes):
            chunk = audio[start : start + self._frame_bytes]
            if len(chunk) < self._frame_bytes:
                cleaned.extend(chunk)
                break
            cleaned.extend(self._cancel(chunk))

        self._mic_energy += _energy(audio)
        self._out_energy += _energy(bytes(cleaned))
        self._frames_seen += 1
        return bytes(cleaned)

    def _cancel(self, mic: bytes) -> bytes:
        reference = self._reference.take(self._frame_bytes)
        out = (ctypes.c_int16 * (self._frame_bytes // 2))()
        self._library.speex_echo_cancellation(
            self._state,
            ctypes.cast(ctypes.create_string_buffer(mic),
                        ctypes.POINTER(ctypes.c_int16)),
            ctypes.cast(ctypes.create_string_buffer(reference),
                        ctypes.POINTER(ctypes.c_int16)),
            out,
        )
        return bytes(out)

    @property
    def echo_reduction_db(self) -> float:
        """How much quieter the microphone got. Zero means it did nothing."""
        if not self._frames_seen or self._out_energy <= 0:
            return 0.0
        return round(10 * math.log10(self._mic_energy / self._out_energy), 1)

    def summary(self) -> str:
        if not self._enabled:
            return "not running"
        return (f"{self.echo_reduction_db} dB average reduction over "
                f"{self._frames_seen} blocks")


def _energy(pcm: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if not samples:
        return 0.0
    return sum(float(s) * s for s in samples) / len(samples)
