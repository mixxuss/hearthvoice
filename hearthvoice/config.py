"""Where everything lives, and the check that it is all local.

Every address the system talks to is declared here. That way the claim that
nothing leaves the local network can be checked in one place rather than being
spread through the code.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

VIP = os.getenv("SPARK_VIP", "192.168.60.49")
ASR_SERVER = os.getenv("NVIDIA_STT_SERVER", f"{VIP}:50051")
TTS_SERVER = os.getenv("NVIDIA_TTS_SERVER", f"{VIP}:50053")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", f"http://{VIP}:19080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5")
TTS_VOICE = os.getenv("NVIDIA_TTS_VOICE", "Magpie-Multilingual.EN-US.Brian")
ASR_MODEL = "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"

MQTT_HOST = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
HA_URL = os.getenv("HA_URL", "http://127.0.0.1:8123")

EVAL_DIR = Path(os.getenv("EVAL_DIR", "eval"))


class OfflineViolation(RuntimeError):
    """Something is set up to talk to a machine we do not control."""


def _is_local(host: str) -> bool:
    """True only for an address on a network the user controls.

    An earlier version of this check built its allowlist out of the same
    environment variables it was checking, so pointing `SPARK_VIP` at a vendor
    made the guard approve of it. A check that can be satisfied by changing its
    own input is not a check. This one asks the address itself: loopback, or
    RFC 1918 private, or nothing.
    """
    if host in {"localhost", ""}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A hostname that is not an IP literal cannot be shown to be local, and
        # the report's claim is that every destination is known and local.
        return False
    return address.is_loopback or address.is_private


def check_offline() -> list[str]:
    """Refuse to start if any stage would leave the local network.

    The earlier version of this project claimed to be local and was not. A
    framework default was sending microphone audio to a vendor, and nothing in
    the code would have noticed. Checking beats assuming.

    Every address the system uses is listed here, including Home Assistant,
    which an earlier version of this function omitted.
    """
    targets = {
        "LLM": urlparse(LLM_BASE_URL).hostname or "",
        "ASR": ASR_SERVER.split(":")[0],
        "TTS": TTS_SERVER.split(":")[0],
        "MQTT": MQTT_HOST,
        "Home Assistant": urlparse(HA_URL).hostname or "",
    }

    leaks = [f"{name} points at {host}"
             for name, host in targets.items() if not _is_local(host)]
    if leaks:
        raise OfflineViolation(
            "These would leave the local network: " + "; ".join(leaks)
        )

    return [f"{name}: {host}" for name, host in targets.items()]
