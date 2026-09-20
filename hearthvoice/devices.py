"""The simulated home.

Ten entities with no I/O of their own. Everything here is pure state, so the
behaviour a voice command produces can be tested without a broker, a microphone
or a model. `mqtt_bridge` is what makes these visible to Home Assistant.

Entity ids follow Home Assistant's `<domain>.<object_id>` convention so the
discovery payloads in `mqtt_bridge` map onto real HA domains rather than a
private scheme.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Iterable


class Domain(StrEnum):
    """The Home Assistant domains this project uses."""

    LIGHT = "light"
    CLIMATE = "climate"
    LOCK = "lock"
    SWITCH = "switch"
    BINARY_SENSOR = "binary_sensor"
    SENSOR = "sensor"


class DeviceError(Exception):
    """Raised when a command cannot be carried out.

    Carries a sentence meant to be spoken to the user rather than logged, because
    in a voice interface the error message is the entire error surface. There is
    no screen to put a code on.
    """


@dataclass
class Entity:
    """One thing in the home.

    `state` holds whatever that domain needs. A light has `on` and `brightness`,
    a thermostat has `current` and `target`. Keeping it loose avoids a class per
    domain for what is ultimately a demonstration home.
    """

    entity_id: str
    name: str
    domain: Domain
    state: dict[str, Any]
    area: str = ""
    writable: bool = True
    unit: str | None = None
    device_class: str | None = None

    @property
    def object_id(self) -> str:
        return self.entity_id.split(".", 1)[1]

    def describe(self) -> str:
        """A short spoken description of the current state."""
        match self.domain:
            case Domain.LIGHT:
                if not self.state["on"]:
                    return f"the {self.name} is off"
                return f"the {self.name} is on at {self.state['brightness']} percent"
            case Domain.CLIMATE:
                return (
                    f"the {self.name} is set to {self.state['target']} degrees "
                    f"and currently reads {self.state['current']} degrees"
                )
            case Domain.LOCK:
                return f"the {self.name} is {'locked' if self.state['locked'] else 'unlocked'}"
            case Domain.SWITCH:
                return f"the {self.name} is {'on' if self.state['on'] else 'off'}"
            case Domain.BINARY_SENSOR:
                on = self.state["on"]
                if self.device_class == "motion":
                    return f"{'motion detected' if on else 'no motion'} in the {self.area}"
                return f"the {self.name} is {'open' if on else 'closed'}"
            case Domain.SENSOR:
                return f"the {self.name} reads {self.state['value']}{self.unit or ''}"
        return f"{self.name} state unknown"


def _default_entities() -> list[Entity]:
    return [
        Entity(
            "light.living_room", "living room light", Domain.LIGHT,
            {"on": False, "brightness": 80}, area="living room",
        ),
        Entity(
            "light.kitchen", "kitchen light", Domain.LIGHT,
            {"on": False, "brightness": 100}, area="kitchen",
        ),
        Entity(
            "light.bedroom", "bedroom light", Domain.LIGHT,
            {"on": False, "brightness": 40}, area="bedroom",
        ),
        Entity(
            "climate.thermostat", "thermostat", Domain.CLIMATE,
            {"target": 20.0, "current": 21.4}, area="hallway", unit="°C",
        ),
        Entity(
            "lock.front_door", "front door lock", Domain.LOCK,
            {"locked": True}, area="hallway",
        ),
        Entity(
            "switch.kettle", "kettle", Domain.SWITCH,
            {"on": False}, area="kitchen",
        ),
        Entity(
            "binary_sensor.motion_hall", "hall motion sensor", Domain.BINARY_SENSOR,
            {"on": False}, area="hallway", writable=False, device_class="motion",
        ),
        Entity(
            "binary_sensor.front_door_contact", "front door sensor", Domain.BINARY_SENSOR,
            {"on": False}, area="hallway", writable=False, device_class="door",
        ),
        Entity(
            "sensor.temperature_living", "living room temperature", Domain.SENSOR,
            {"value": 21.4}, area="living room", writable=False,
            unit="°C", device_class="temperature",
        ),
        Entity(
            "sensor.last_command", "last voice command", Domain.SENSOR,
            {"value": "none yet"}, writable=False,
        ),
    ]


# Spoken names people actually use, mapped to entity ids. The recogniser returns
# "lounge" or "front door" rather than "light.living_room", and refusing to
# understand a synonym is a usability failure, not a safety feature.
ALIASES: dict[str, str] = {
    "living room": "light.living_room",
    "living room light": "light.living_room",
    "living room lights": "light.living_room",
    "lounge": "light.living_room",
    "sitting room": "light.living_room",
    "kitchen": "light.kitchen",
    "kitchen light": "light.kitchen",
    "kitchen lights": "light.kitchen",
    "bedroom": "light.bedroom",
    "bedroom light": "light.bedroom",
    "bedroom lights": "light.bedroom",
    "thermostat": "climate.thermostat",
    "heating": "climate.thermostat",
    "temperature": "climate.thermostat",
    "front door": "lock.front_door",
    "front door lock": "lock.front_door",
    "door": "lock.front_door",
    "kettle": "switch.kettle",
    "motion": "binary_sensor.motion_hall",
    "hall motion": "binary_sensor.motion_hall",
    "door sensor": "binary_sensor.front_door_contact",
    "living room temperature": "sensor.temperature_living",
}

# Actions that change something a burglar would care about. These get confirmed
# rather than executed on a first-pass recognition (heuristic H5).
CONFIRM_REQUIRED = {"unlock"}


@dataclass
class Home:
    """The house. Owns the entities and every state change to them."""

    entities: dict[str, Entity] = field(default_factory=dict)
    on_change: Callable[[Entity], None] | None = None
    _pending_confirmation: tuple[str, str] | None = field(default=None, init=False)
    # Which user turn the confirmation was asked for in. A confirmation that
    # arrives in the same turn as the request did not come from the user: the
    # model emitted both tool calls at once. Chapter 4 called this guard
    # model-independent while it was defeated by two calls in one response.
    _pending_turn: int = field(default=-1, init=False)
    turn: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.entities:
            self.entities = {e.entity_id: e for e in _default_entities()}

    # ------------------------------------------------------------------ lookup

    def resolve(self, spoken: str) -> Entity:
        """Turn something a person said into an entity, or explain the failure."""
        key = spoken.strip().lower().removeprefix("the ")
        if key in self.entities:
            return self.entities[key]
        if key in ALIASES:
            return self.entities[ALIASES[key]]

        # ALIASES maps a spoken phrase to an entity id, so the values here are
        # strings; look each one up before treating it as an entity.
        partial = [
            self.entities[eid]
            for phrase, eid in ALIASES.items()
            if (key in phrase or phrase in key) and eid in self.entities
        ]
        unique = {e.entity_id: e for e in partial}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if len(unique) > 1:
            names = ", ".join(sorted(e.name for e in unique.values()))
            raise DeviceError(f"I know several: {names}. Which one did you mean?")
        raise DeviceError(f"I do not have anything called {spoken}.")

    def all(self) -> Iterable[Entity]:
        return self.entities.values()

    def summary(self) -> str:
        """Everything in the house, spoken. Leaves out the transcript sensor."""
        return "; ".join(
            e.describe() for e in self.all()
            if e.entity_id != "sensor.last_command"
        )

    # ----------------------------------------------------------------- mutate

    def _changed(self, entity: Entity) -> None:
        if self.on_change:
            self.on_change(entity)

    def set_power(self, spoken: str, on: bool) -> str:
        entity = self.resolve(spoken)
        if entity.domain not in (Domain.LIGHT, Domain.SWITCH):
            raise DeviceError(f"I cannot switch {entity.name} on or off.")
        entity.state["on"] = on
        self._changed(entity)
        return f"{entity.name} {'on' if on else 'off'}"

    def set_brightness(self, spoken: str, percent: int) -> str:
        entity = self.resolve(spoken)
        if entity.domain is not Domain.LIGHT:
            raise DeviceError(f"{entity.name} does not have a brightness.")
        if not 0 <= percent <= 100:
            raise DeviceError("Brightness has to be between 0 and 100 percent.")
        entity.state["brightness"] = percent
        entity.state["on"] = percent > 0
        self._changed(entity)
        return f"{entity.name} at {percent} percent"

    def set_temperature(self, celsius: float) -> str:
        if not 5 <= celsius <= 30:
            raise DeviceError(
                "I can only set the thermostat between 5 and 30 degrees."
            )
        entity = self.entities["climate.thermostat"]
        entity.state["target"] = float(celsius)
        self._changed(entity)
        return f"thermostat set to {celsius:g} degrees"

    def begin_turn(self) -> None:
        """A new thing was said. Called once per user utterance."""
        self.turn += 1

    def set_lock(self, spoken: str, locked: bool, confirmed: bool = False) -> str:
        entity = self.resolve(spoken)
        if entity.domain is not Domain.LOCK:
            raise DeviceError(f"{entity.name} is not a lock.")
        if not locked and not confirmed:
            self._pending_confirmation = (entity.entity_id, "unlock")
            self._pending_turn = self.turn
            raise DeviceError(
                f"Unlocking the {entity.name} is not something I will do without "
                "a confirmation. Say yes to confirm."
            )
        entity.state["locked"] = locked
        self._pending_confirmation = None
        self._changed(entity)
        return f"{entity.name} {'locked' if locked else 'unlocked'}"

    def confirm_pending(self) -> str:
        """Carry out whatever was waiting on a yes.

        The yes has to arrive in a later turn than the request. Otherwise the
        model can ask for confirmation and grant it in the same breath, which
        is what happens if it emits set_lock and confirm_action together, and
        which makes the guard theatre.
        """
        if not self._pending_confirmation:
            raise DeviceError("There is nothing waiting for a confirmation.")
        if self._pending_turn == self.turn:
            self._pending_confirmation = None
            raise DeviceError(
                "I asked for that confirmation just now and have not heard you "
                "answer yet. Say yes if you want me to go ahead."
            )
        entity_id, action = self._pending_confirmation
        self._pending_confirmation = None
        if action == "unlock":
            return self.set_lock(entity_id, locked=False, confirmed=True)
        raise DeviceError(f"I do not know how to confirm {action}.")

    def all_lights(self, on: bool) -> str:
        for entity in self.entities.values():
            if entity.domain is Domain.LIGHT:
                entity.state["on"] = on
                self._changed(entity)
        return f"all lights {'on' if on else 'off'}"

    def note_command(self, text: str) -> None:
        """Publish what was heard, so the dashboard shows it next to the devices."""
        entity = self.entities["sensor.last_command"]
        entity.state["value"] = text
        self._changed(entity)

    # ---------------------------------------------------------------- ambient

    def tick_sensors(self, rng: random.Random | None = None) -> list[Entity]:
        """Nudge the read-only sensors so the dashboard is not frozen.

        A demonstration home with permanently static sensors looks broken on
        video. The drift is small and bounded; nothing here pretends to model
        real thermal behaviour.
        """
        rng = rng or random.Random()
        touched = []

        temp = self.entities["sensor.temperature_living"]
        drift = round(rng.uniform(-0.2, 0.2), 1)
        temp.state["value"] = round(
            min(26.0, max(17.0, temp.state["value"] + drift)), 1
        )
        touched.append(temp)

        thermostat = self.entities["climate.thermostat"]
        thermostat.state["current"] = temp.state["value"]
        touched.append(thermostat)

        motion = self.entities["binary_sensor.motion_hall"]
        if rng.random() < 0.15:
            motion.state["on"] = not motion.state["on"]
            touched.append(motion)

        for entity in touched:
            self._changed(entity)
        return touched
