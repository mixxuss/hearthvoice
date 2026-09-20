"""Is the stall driven by turns, or just by time?

The evaluation names two suspects for the live-path stall. One of them, the
sensor-drift task, is ruled out by running the live test with it off and
getting the identical failure. The other is pipecat's audio callback, which
schedules a coroutine for every 20 ms buffer without awaiting it.

If that is the cause, the event loop should starve on its own, with nobody
speaking at all. This runs the real pipeline against a silent microphone and
measures how late a task that asks to wake every 0.5 s actually wakes. No
speech, no turns, so anything it finds is not about turn count.
"""

from __future__ import annotations

import asyncio
import sys
import time

from hearthvoice import agent


async def heartbeat(seconds: float) -> None:
    started = time.perf_counter()
    worst = 0.0
    while time.perf_counter() - started < seconds:
        due = time.perf_counter() + 0.5
        await asyncio.sleep(0.5)
        lag = time.perf_counter() - due
        worst = max(worst, lag)
        elapsed = time.perf_counter() - started
        if lag > 0.1 or int(elapsed) % 5 == 0:
            print(f"  {elapsed:6.1f}s  loop lag {lag * 1000:7.1f} ms"
                  f"   worst so far {worst * 1000:.1f} ms", flush=True)
    print(f"\n  survived {seconds:.0f}s, worst lag {worst * 1000:.1f} ms", flush=True)
    import os
    os._exit(0)
    raise SystemExit(0)


async def main(in_device: int, seconds: float) -> None:
    asyncio.create_task(heartbeat(seconds))
    await agent.run(in_device=in_device, out_device=None, drift=False, aec=True)


if __name__ == "__main__":
    device = int(sys.argv[1])
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
    try:
        asyncio.run(main(device, duration))
    except SystemExit:
        pass
