"""Publishes the simulated home to Home Assistant over MQTT.

Home Assistant picks entities up automatically if you publish a discovery config
to `homeassistant/<component>/<node>/<object_id>/config`. That is why there is no
YAML to edit and no UI to click through: the agent announces the house and Home
Assistant builds the dashboard from it.

Command topics run the other way, so toggling a tile in Lovelace changes the same
state the voice agent sees. Without that the dashboard would be a read-only
picture rather than a shared model of the house.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable

import paho.mqtt.client as mqtt

from .devices import Domain, Entity, Home

log = logging.getLogger(__name__)

DISCOVERY_PREFIX = "homeassistant"
NODE_ID = "hearthvoice"
STATE_PREFIX = f"{NODE_ID}/state"
COMMAND_PREFIX = f"{NODE_ID}/set"
AVAILABILITY_TOPIC = f"{NODE_ID}/status"

DEVICE_INFO = {
    "identifiers": [NODE_ID],
    "name": "HearthVoice",
    "manufacturer": "CM3070 final project",
    "model": "Offline voice assistant",
}


def state_topic(entity: Entity) -> str:
    return f"{STATE_PREFIX}/{entity.object_id}"


def command_topic(entity: Entity) -> str:
    return f"{COMMAND_PREFIX}/{entity.object_id}"


def discovery_topic(entity: Entity) -> str:
    return f"{DISCOVERY_PREFIX}/{entity.domain}/{NODE_ID}/{entity.object_id}/config"


def discovery_payload(entity: Entity) -> dict:
    """Build the Home Assistant MQTT discovery config for one entity."""
    payload: dict = {
        "name": entity.name,
        "unique_id": f"{NODE_ID}_{entity.object_id}",
        "state_topic": state_topic(entity),
        "availability_topic": AVAILABILITY_TOPIC,
        "device": DEVICE_INFO,
    }
    if entity.area:
        payload["suggested_area"] = entity.area.title()
    if entity.writable:
        payload["command_topic"] = command_topic(entity)

    match entity.domain:
        case Domain.LIGHT:
            payload.update(
                schema="json",
                brightness=True,
                brightness_scale=100,
            )
        case Domain.CLIMATE:
            payload.update(
                modes=["heat"],
                mode_state_topic=state_topic(entity),
                mode_state_template="{{ 'heat' }}",
                temperature_state_topic=state_topic(entity),
                temperature_state_template="{{ value_json.target }}",
                temperature_command_topic=command_topic(entity),
                current_temperature_topic=state_topic(entity),
                current_temperature_template="{{ value_json.current }}",
                min_temp=5,
                max_temp=30,
                temp_step=0.5,
                temperature_unit="C",
            )
        case Domain.LOCK:
            payload.update(
                payload_lock="LOCK",
                payload_unlock="UNLOCK",
                state_locked="LOCKED",
                state_unlocked="UNLOCKED",
            )
        case Domain.SWITCH:
            payload.update(payload_on="ON", payload_off="OFF")
        case Domain.BINARY_SENSOR:
            payload.update(payload_on="ON", payload_off="OFF")
            if entity.device_class:
                payload["device_class"] = entity.device_class
        case Domain.SENSOR:
            if entity.unit:
                payload["unit_of_measurement"] = entity.unit
            if entity.device_class:
                payload["device_class"] = entity.device_class

    return payload


def state_payload(entity: Entity) -> str:
    """Serialise an entity's state in the shape its discovery config promised."""
    match entity.domain:
        case Domain.LIGHT:
            return json.dumps(
                {
                    "state": "ON" if entity.state["on"] else "OFF",
                    "brightness": entity.state["brightness"],
                }
            )
        case Domain.CLIMATE:
            return json.dumps(
                {"target": entity.state["target"], "current": entity.state["current"]}
            )
        case Domain.LOCK:
            return "LOCKED" if entity.state["locked"] else "UNLOCKED"
        case Domain.SWITCH | Domain.BINARY_SENSOR:
            return "ON" if entity.state["on"] else "OFF"
        case Domain.SENSOR:
            return str(entity.state["value"])
    return ""


class MqttBridge:
    """Connects a `Home` to a local broker.

    Deliberately fails loudly rather than degrading: if the broker is not there,
    the agent should say so out loud, not carry on pretending the lights changed.
    """

    def __init__(
        self,
        home: Home,
        host: str = "127.0.0.1",
        port: int = 1883,
        client_id: str = "hearthvoice-agent",
        on_external_change: Callable[[Entity], None] | None = None,
    ) -> None:
        self.home = home
        self.host = host
        self.port = port
        self.on_external_change = on_external_change
        self._connected = threading.Event()

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=client_id
        )
        self.client.will_set(AVAILABILITY_TOPIC, "offline", retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        home.on_change = self.publish_state

    # ------------------------------------------------------------- lifecycle

    def start(self, timeout: float = 10.0) -> None:
        self.client.connect(self.host, self.port, keepalive=30)
        self.client.loop_start()
        if not self._connected.wait(timeout):
            raise ConnectionError(
                f"No MQTT broker at {self.host}:{self.port}. "
                "Start it with: docker compose -f deploy/docker-compose.yaml up -d"
            )

    def stop(self) -> None:
        self.client.publish(AVAILABILITY_TOPIC, "offline", retain=True)
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if reason_code != 0:
            log.error("MQTT connect refused: %s", reason_code)
            return
        log.info("MQTT connected to %s:%s", self.host, self.port)
        self.announce()
        client.subscribe(f"{COMMAND_PREFIX}/#")
        client.publish(AVAILABILITY_TOPIC, "online", retain=True)
        self._connected.set()

    # ------------------------------------------------------------- publishing

    def announce(self) -> None:
        """Publish discovery for every entity, then its current state."""
        for entity in self.home.all():
            self.client.publish(
                discovery_topic(entity),
                json.dumps(discovery_payload(entity)),
                retain=True,
            )
            self.publish_state(entity)

    def publish_state(self, entity: Entity) -> None:
        self.client.publish(state_topic(entity), state_payload(entity), retain=True)

    # -------------------------------------------------------------- incoming

    def _on_message(self, client, userdata, msg) -> None:
        """Apply a change made from the Home Assistant dashboard."""
        object_id = msg.topic.rsplit("/", 1)[-1]
        entity = next(
            (e for e in self.home.all() if e.object_id == object_id), None
        )
        if entity is None:
            return
        raw = msg.payload.decode()

        try:
            match entity.domain:
                case Domain.LIGHT:
                    body = json.loads(raw)
                    entity.state["on"] = body.get("state", "OFF").upper() == "ON"
                    if "brightness" in body:
                        entity.state["brightness"] = int(body["brightness"])
                case Domain.CLIMATE:
                    entity.state["target"] = float(raw)
                case Domain.LOCK:
                    entity.state["locked"] = raw.upper() == "LOCK"
                case Domain.SWITCH:
                    entity.state["on"] = raw.upper() == "ON"
                case _:
                    return
        except (ValueError, json.JSONDecodeError):
            log.warning("Unparseable command on %s: %r", msg.topic, raw)
            return

        self.publish_state(entity)
        if self.on_external_change:
            self.on_external_change(entity)
