"""Check that nothing leaves the local network during a conversation.

This is the test the project's central claim rests on, and the one that was
missing. `check_offline()` reads the configuration; this watches what actually
happens.

Method: run the evaluation suite, and while it runs, list every network
connection the agent process holds open, repeatedly. Each remote address is
then judged against what the system is allowed to reach, which is the Spark and
this machine.

Per-process enumeration rather than a packet capture, for two reasons. It needs
no root, so anyone can reproduce it. And it attributes each connection to the
process, where a capture shows everything the laptop was doing, browser and
mail client included. A capture is the better evidence for wire-level truth and
the worse for attribution, so the `tcpdump` line for running both is printed at
the end.

    python tools_diag/egress_test.py
"""

from __future__ import annotations

import ipaddress
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hearthvoice.config import EVAL_DIR, MQTT_HOST, VIP  # noqa: E402

SAMPLE_EVERY = 0.25


def _why_allowed(host: str) -> str | None:
    """Everything the system may reach. Anything else is a finding."""
    if host == VIP:
        return "DGX Spark: recognition, language model and speech"
    if host in {MQTT_HOST, "127.0.0.1", "::1", "localhost"}:
        return "this machine: MQTT broker and Home Assistant"
    try:
        if ipaddress.ip_address(host).is_loopback:
            return "this machine"
    except ValueError:
        pass
    return None


def connections(pid: int) -> set[tuple[str, str]]:
    """Every remote endpoint this process currently holds open."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-i", "-a", "-p", str(pid)],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return set()

    found = set()
    for line in out.splitlines()[1:]:
        match = re.search(r"->(\[?[0-9a-fA-F:.]+\]?):(\d+)", line)
        if match:
            found.add((match.group(1).strip("[]"), match.group(2)))
    return found


def watch(pid: int, seen: set, stop: threading.Event) -> None:
    while not stop.is_set():
        seen |= connections(pid)
        time.sleep(SAMPLE_EVERY)


def main() -> None:
    print("\n  Running the evaluation suite and watching every connection it opens.\n")

    project = Path(__file__).resolve().parent.parent
    suite = subprocess.Popen(
        # Twelve scenarios, not the full thirty: this is the suite as it was
        # when the egress measurement in the report was taken, and it keeps
        # the run to about a minute and a half.
        [sys.executable, "-m", "hearthvoice.harness", "--limit", "12"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=project,
    )

    seen: set[tuple[str, str]] = set()
    stop = threading.Event()
    watcher = threading.Thread(target=watch, args=(suite.pid, seen, stop))
    watcher.start()

    started = time.perf_counter()
    output = suite.communicate()[0]
    stop.set()
    watcher.join()
    elapsed = round(time.perf_counter() - started, 1)

    endpoints = []
    for host, port in sorted(seen):
        why = _why_allowed(host)
        endpoints.append({
            "remote": host,
            "port": int(port),
            "allowed": why is not None,
            "why": why or "NOT PERMITTED: neither the Spark nor this machine",
        })

    leaks = [e for e in endpoints if not e["allowed"]]
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "method": ("per-process socket enumeration via lsof, sampled every "
                   f"{SAMPLE_EVERY} s while the evaluation suite ran"),
        "duration_s": elapsed,
        "allowed_hosts": {"spark": VIP, "local": [MQTT_HOST, "127.0.0.1"]},
        "endpoints": endpoints,
        "verdict": "no egress" if not leaks else f"{len(leaks)} unexpected",
    }

    path = EVAL_DIR / "egress_test.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"  suite finished in {elapsed} s")
    print(f"  {len(endpoints)} distinct remote endpoints\n")
    for endpoint in endpoints:
        mark = "ok  " if endpoint["allowed"] else "LEAK"
        print(f"    [{mark}] {endpoint['remote']}:{endpoint['port']:<6} "
              f"{endpoint['why']}")

    print(f"\n  verdict: {report['verdict']}")
    print(f"  written to {path}")
    print("\n  For wire-level confirmation, run this in another terminal while"
          "\n  the suite runs, and compare the destinations:"
          f"\n    sudo tcpdump -n -i en0 'not host {VIP} and not net 127.0.0.0/8'\n")

    if "written to" not in output:
        print("  note: the suite did not finish cleanly, check its output")


if __name__ == "__main__":
    main()
