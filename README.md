# HearthVoice

A voice assistant for a smart home that runs its models on hardware you own.
You speak, it works out what you meant, it changes a device, and it tells you
what it did. Nothing in the conversation reaches a third party.

Built for the CM3070 final project, against the CM3020 Artificial Intelligence
template (orchestrating pre-trained models across different data types).

## What it runs on

Three models, all on an NVIDIA DGX Spark reached over the local network:

| Stage | Model | Port |
|---|---|---|
| Speech recognition | `parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer` | 50051 |
| Dialogue, tool calling, vision | `Qwen3.5-35B-A3B` via vLLM | 19080 |
| Speech synthesis | Magpie Multilingual, `EN-US.Brian` | 50053 |

Plus Mosquitto and Home Assistant in Docker on the machine you run the agent
from. The agent refuses to start if any of these is configured to point
somewhere other than the Spark or localhost.

## What you need

- A machine with a microphone and speakers. Developed on an Apple Silicon Mac.
- Python 3.12.
- Docker, for the broker and Home Assistant.
- Network access to a Riva speech server and a vLLM endpoint. Without those
  the agent will not start; there is no cloud fallback by design.
- `portaudio` and `libspeexdsp`. On a Mac where Homebrew is x86 and Python is
  arm64, both need building from source; see "Audio libraries" below.

## Getting it running

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then point it at your own servers

docker compose -f deploy/docker-compose.yaml up -d
python deploy/bootstrap_ha.py # creates the HA account and wires up MQTT

python -m hearthvoice.agent --check     # confirm every target is local
python -m hearthvoice.agent             # talk to it
```

The dashboard is at http://127.0.0.1:8123 with the credentials in
`deploy/bootstrap_ha.py`. Ten devices appear automatically through MQTT
discovery; there is no dashboard to configure.

**Use headphones.** Without them the agent hears its own voice through the
speakers and cuts its replies short. Echo cancellation is implemented
(`hearthvoice/aec.py`) and only partly effective: there is no delay
estimator, so it subtracts a reference it assumes is aligned.

## Evaluation

```bash
python -m hearthvoice.harness --repeat 3   # thirty commands, intent + latency
python tools_diag/egress_test.py           # prove nothing leaves the network
python tools_diag/model_comparison.py      # Parakeet vs Whisper, Magpie vs Piper
python -m pytest tests/ -q                 # 30 unit tests
```

Each writes JSON to `eval/`, and the report's figures are generated from those
files rather than typed in by hand. The results are not committed here; run the
scripts against your own stack and you get your own.

## Audio libraries

PyAudio's wheel links against a portaudio that may not match your Python's
architecture. If `import pyaudio` fails with a missing symbol:

```bash
curl -LO https://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz
tar xzf pa_stable_v190700_20210406.tgz && cd portaudio
./configure --prefix="$HOME/.local/hearthvoice/portaudio" --disable-mac-universal \
  CFLAGS="-arch arm64" LDFLAGS="-arch arm64" && make && make install
CFLAGS="-I$HOME/.local/hearthvoice/portaudio/include" \
LDFLAGS="-L$HOME/.local/hearthvoice/portaudio/lib" \
  pip install --no-cache-dir --no-binary=pyaudio pyaudio
```

Build the prefix somewhere without spaces in the path; libtool fails otherwise.

Echo cancellation needs `libspeexdsp`, found via `SPEEX_LIB` or
`~/.local/hearthvoice/speexdsp/lib`. Built the same way from
https://downloads.xiph.org/releases/speex/speexdsp-1.2.1.tar.gz. Without it the
agent runs and logs that echo cancellation is off.

## Layout

```
hearthvoice/
  agent.py        the pipecat pipeline: microphone to speaker
  config.py       every address the system uses, and the check that they are local
  devices.py      ten entities and the rules about them, no I/O
  engines.py      one place for calling the three Spark models
  harness.py      the evaluation suite
  live_test.py    drives the real microphone path through a loopback device
  metrics.py      per-stage timing, with the averaging bug fixed
  mqtt_bridge.py  publishes the house to Home Assistant
  prompt.py       what the model is told before it hears anything
  tools.py        the nine tools the model may call
  vision.py       the camera, through the same local model
  aec.py          SpeexDSP echo cancellation
deploy/           Mosquitto and Home Assistant, plus automated setup
tests/            unit tests for the rules that must hold
tools_diag/       measurement scripts used for the report
```

## Known issues

The agent still hears itself on laptop speakers, so use headphones. Echo
cancellation is integrated but has no delay estimator, so it subtracts a
reference it assumes is aligned.

The live loopback test used to stop after three turns, and that turned out to
be the harness rather than the agent: it captured the agent's stdout with a
pipe it only read at the end, so three turns of verbose context logging filled
the 64 KB buffer and the agent blocked inside `logging` with its event loop
stopped. Output now goes to a file and the suite runs to the end, passing six
or seven of seven cases.

Per-stage timings are not reported on the live path. The transcript arrives
before the frame marking the end of the user's turn, so the recognition stage
has no interval to be measured over, and those figures are left empty rather
than filled with something wrong. The wait itself is measured directly: across
eight sessions, a per-session mean of 1.27 to 2.20 s, with 39 of 52 timed turns
inside the two-second target.

## Camera

The vision tool tries camera indices 0, 1 and 2 and keeps the first that
returns a frame with something in it, because a Mac with an iPhone nearby can
list the phone's Continuity Camera first and it returns black. Pin a device
with `VISION_CAMERA=1`. On macOS the terminal running the agent needs Camera
permission; the first use prompts for it.
