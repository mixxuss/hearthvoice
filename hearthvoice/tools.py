"""What the model is allowed to do, and how it does it.

The tools are described once, in `TOOLS`. The agent needs them as pipecat
schemas and the evaluation harness needs them as plain OpenAI dicts, so both
shapes are generated from that one list. They used to be written out twice,
which let the harness drift away from the thing it was meant to measure.

The handlers are thin. Every rule about what counts as a valid command lives in
`devices.py`, so the rules can be tested without a model in the loop.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

from . import vision
from .devices import DeviceError, Home

log = logging.getLogger(__name__)

ON_OFF = {"type": "string", "enum": ["on", "off"]}
DEVICE = {
    "type": "string",
    "description": "What the user called it, such as 'living room lights' or 'kettle'.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "set_power",
        "description": "Turn a light or switch on or off.",
        "properties": {"device": DEVICE, "state": ON_OFF},
        "required": ["device", "state"],
    },
    {
        "name": "set_brightness",
        "description": "Set how bright a light is, as a percentage.",
        "properties": {
            "device": DEVICE,
            "percent": {"type": "integer", "description": "0 to 100."},
        },
        "required": ["device", "percent"],
    },
    {
        "name": "set_temperature",
        "description": "Set the thermostat target in Celsius.",
        "properties": {
            "celsius": {"type": "number", "description": "Between 5 and 30."}
        },
        "required": ["celsius"],
    },
    {
        "name": "set_lock",
        "description": (
            "Lock or unlock a door. Unlocking always needs the user to confirm "
            "first, so call this once, say what it tells you, and only call "
            "confirm_action if the user then agrees."
        ),
        "properties": {
            "device": DEVICE,
            "action": {"type": "string", "enum": ["lock", "unlock"]},
        },
        "required": ["device", "action"],
    },
    {
        "name": "confirm_action",
        "description": (
            "Carry out whatever is waiting for confirmation. Only after the "
            "user has said yes."
        ),
        "properties": {},
        "required": [],
    },
    {
        "name": "all_lights",
        "description": (
            "Turn EVERY light in the house on or off at once. Only when the "
            "user says all, every, or the whole house. If they name a room use "
            "set_power, even if they say 'lights' in the plural."
        ),
        "properties": {"state": ON_OFF},
        "required": ["state"],
    },
    {
        "name": "get_state",
        "description": "Read the current state of one device or sensor.",
        "properties": {"device": DEVICE},
        "required": ["device"],
    },
    {
        "name": "describe_surroundings",
        "description": (
            "Look through the camera and say what is there. Use whenever the "
            "user asks what something is, what a label or packet says, what "
            "they are holding, what is in front of them, or asks you to read, "
            "look at, or check something. You can see: never say you cannot."
        ),
        "properties": {
            "question": {
                "type": "string",
                "description": "What they want to know, if they were specific.",
            }
        },
        "required": [],
    },
    {
        "name": "list_devices",
        "description": "List everything in the house.",
        "properties": {},
        "required": [],
    },
]


def openai_schema() -> list[dict]:
    """The tools in the shape the OpenAI API expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool["properties"],
                    "required": tool["required"],
                },
            },
        }
        for tool in TOOLS
    ]


def pipecat_schema() -> ToolsSchema:
    """The same tools in the shape pipecat expects."""
    return ToolsSchema(standard_tools=[FunctionSchema(**tool) for tool in TOOLS])


def run(home: Home, name: str, args: dict) -> tuple[str, bool]:
    """Carry out one tool call. Returns what to say, and whether it refused.

    A refusal is an ordinary outcome, not an error to hide. The model gets the
    text so it can say it out loud, because in a voice interface the message is
    the whole error surface.
    """
    actions: dict[str, Callable[[], str]] = {
        "set_power": lambda: home.set_power(args["device"], args["state"] == "on"),
        "set_brightness": lambda: home.set_brightness(
            args["device"], int(args["percent"])
        ),
        "set_temperature": lambda: home.set_temperature(float(args["celsius"])),
        "set_lock": lambda: home.set_lock(args["device"], args["action"] == "lock"),
        "confirm_action": home.confirm_pending,
        "all_lights": lambda: home.all_lights(args["state"] == "on"),
        "get_state": lambda: home.resolve(args["device"]).describe(),
        "list_devices": home.summary,
        "describe_surroundings": lambda: vision.describe(
            question=args.get("question") or None),
    }

    action = actions.get(name)
    if action is None:
        return f"I do not have a tool called {name}.", True

    try:
        result = action()
        log.info("%s -> %s", name, result)
        return result, False
    except vision.NoImage as e:
        # Not being able to see is something to say out loud, not an error
        # to swallow: the user may be relying on it.
        log.info("%s could not see: %s", name, e)
        return str(e), True
    except DeviceError as e:
        log.info("%s refused: %s", name, e)
        return str(e), True
    except (KeyError, ValueError) as e:
        log.warning("%s got bad arguments: %s", name, e)
        return "I could not make sense of that request.", True


def register(home: Home, llm: Any) -> None:
    """Wire the tools into a pipecat LLM service."""
    import asyncio
    from pipecat.frames.frames import TTSSpeakFrame

    def handler_for(name: str):
        async def handler(params) -> None:
            if name == "describe_surroundings":
                # Opening the camera and asking the vision model takes three or
                # four seconds, and the reply that follows takes a couple more.
                # Seven seconds of silence reads as a system that has not heard
                # you, and a second attempt during that silence starts a new
                # turn and cancels this one. Say something first. This is the
                # visibility heuristic from the evaluation, applied to the one
                # tool slow enough to need it.
                await params.llm.push_frame(
                    TTSSpeakFrame("Let me look.", append_to_context=False)
                )
            # The tools block: the camera, the vision call, MQTT publishes.
            # Running them on the event loop froze the pipeline for the
            # duration, so audio frames piled up and were released in a burst.
            message, refused = await asyncio.to_thread(
                run, home, name, params.arguments
            )
            await params.result_callback(
                {"error" if refused else "result": message}
            )

        return handler

    for tool in TOOLS:
        llm.register_function(tool["name"], handler_for(tool["name"]))
