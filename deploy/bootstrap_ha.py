"""Bring a fresh Home Assistant from first boot to a working MQTT dashboard.

Home Assistant normally wants a person to click through onboarding and then add
the MQTT integration by hand. That would make the demo unreproducible, and
reproducibility is something the project is judged on, so this does the whole
thing over the REST API instead.

Idempotent: safe to run again against an instance that is already set up, as long
as the credentials match.

    python deploy/bootstrap_ha.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HA = os.getenv("HA_URL", "http://127.0.0.1:8123")
USERNAME = os.getenv("HA_USER", "hearthvoice")
PASSWORD = os.getenv("HA_PASSWORD", "hearthvoice-cm3070")
NAME = os.getenv("HA_NAME", "HearthVoice")
BROKER_HOST = os.getenv("MQTT_HOST_FOR_HA", "mosquitto")
BROKER_PORT = int(os.getenv("MQTT_PORT_FOR_HA", "1883"))
CLIENT_ID = "http://hearthvoice.local/"
TOKEN_PATH = os.path.join(os.path.dirname(__file__), ".ha_token")
REFRESH_PATH = os.path.join(os.path.dirname(__file__), ".ha_refresh")


def _request(path: str, payload=None, token: str | None = None, method="GET"):
    url = f"{HA}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def wait_for_ha(timeout: float = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{HA}/", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"Home Assistant did not answer on {HA}")


def onboard() -> str | None:
    """Create the owner account and finish onboarding. Returns an auth code."""
    status, steps = _request("/api/onboarding")
    if isinstance(steps, list) and all(s["done"] for s in steps):
        print("onboarding already complete")
        return None

    status, body = _request(
        "/api/onboarding/users",
        {
            "client_id": CLIENT_ID,
            "name": NAME,
            "username": USERNAME,
            "password": PASSWORD,
            "language": "en",
        },
    )
    if status == 404:
        # Onboarding endpoints disappear once it is finished, which is the
        # normal case on a re-run.
        print("onboarding already complete")
        return None
    if status != 200:
        raise RuntimeError(f"user creation failed: {status} {body}")
    auth_code = body["auth_code"]
    print("owner account created")

    token = exchange_code(auth_code)
    for step in ("core_config", "analytics"):
        st, bd = _request(f"/api/onboarding/{step}", {}, token=token)
        print(f"onboarding {step}: {st}")
    _request("/api/onboarding/integration",
             {"client_id": CLIENT_ID, "redirect_uri": CLIENT_ID}, token=token)
    return token


def exchange_code(auth_code: str) -> str:
    """Swap an auth code for an access token, keeping the refresh token.

    Access tokens expire in about half an hour, which is shorter than a test
    run. The refresh token is what makes `access_token()` below able to hand
    out a fresh one whenever something needs to query Home Assistant.
    """
    data = urllib.parse.urlencode(
        {"client_id": CLIENT_ID, "grant_type": "authorization_code", "code": auth_code}
    ).encode()
    req = urllib.request.Request(f"{HA}/auth/token", data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    if body.get("refresh_token"):
        with open(REFRESH_PATH, "w") as f:
            f.write(body["refresh_token"])
    return body["access_token"]


def access_token() -> str:
    """A currently valid access token, minted from the stored refresh token."""
    if not os.path.exists(REFRESH_PATH):
        return login()
    with open(REFRESH_PATH) as f:
        refresh = f.read().strip()
    data = urllib.parse.urlencode(
        {"client_id": CLIENT_ID, "grant_type": "refresh_token",
         "refresh_token": refresh}
    ).encode()
    req = urllib.request.Request(f"{HA}/auth/token", data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())["access_token"]
    except urllib.error.HTTPError:
        return login()


def login() -> str:
    """Password login for an instance that is already onboarded."""
    step = _login_flow_start()
    flow_id = step["flow_id"]
    status, body = _request(
        f"/auth/login_flow/{flow_id}",
        {"client_id": CLIENT_ID, "username": USERNAME, "password": PASSWORD},
    )
    if status != 200 or not isinstance(body, dict) or "result" not in body:
        raise RuntimeError(f"login failed: {status} {body}")
    return exchange_code(body["result"])


def _login_flow_start() -> dict:
    status, body = _request(
        "/auth/login_flow",
        {
            "client_id": CLIENT_ID,
            "handler": ["homeassistant", None],
            "redirect_uri": CLIENT_ID,
        },
    )
    if status != 200:
        raise RuntimeError(f"login flow refused: {status} {body}")
    return body


def add_mqtt(token: str) -> None:
    """Create the MQTT config entry pointing at the local broker."""
    status, entries = _request("/api/config/config_entries/entry", token=token)
    if isinstance(entries, list) and any(e.get("domain") == "mqtt" for e in entries):
        print("MQTT integration already configured")
        return

    status, flow = _request(
        "/api/config/config_entries/flow",
        {"handler": "mqtt", "show_advanced_options": True},
        token=token,
    )
    if status != 200:
        raise RuntimeError(f"mqtt flow start failed: {status} {flow}")

    status, result = _request(
        f"/api/config/config_entries/flow/{flow['flow_id']}",
        {
            "broker": BROKER_HOST,
            "port": BROKER_PORT,
            "protocol": "5",
            # HA 2026.x nests the advanced fields; set_client_cert, set_ca_cert
            # and transport are required even when you want plain unencrypted TCP.
            "other_settings": {
                "set_client_cert": False,
                "set_ca_cert": "off",
                "transport": "tcp",
            },
        },
        token=token,
    )
    if status != 200 or (isinstance(result, dict) and result.get("errors")):
        raise RuntimeError(f"mqtt setup failed: {status} {result}")
    print(f"MQTT integration added ({BROKER_HOST}:{BROKER_PORT})")


def main() -> None:
    print(f"waiting for {HA} ...")
    wait_for_ha()

    token = onboard()
    if token is None:
        token = login()

    add_mqtt(token)

    with open(TOKEN_PATH, "w") as f:
        f.write(token)
    print(f"token written to {TOKEN_PATH}")
    print(f"\nDashboard: {HA}   user: {USERNAME}   password: {PASSWORD}")


if __name__ == "__main__":
    main()
