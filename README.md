# Turbulence-aware flight ranking

An agent that ranks flights by expected turbulence, and says so when it
doesn't know.

Live: **[turbulence.adeptsecurity.net](https://turbulence.adeptsecurity.net)**

---

## The problem

I'm an anxious flier. All I want to know before a flight is whether it's
going to be bumpy.

All the inputs to that question are public and free:

- pilots file turbulence reports en route
- the Aviation Weather Center makes turbulence forecasts
- filed routes and flown tracks are published

None of that answers the question. Taking three sparse, disagreeing inputs
and turning them into one answer requires judgement, and on most routes the
honest answer is that nobody knows.

That's why no product exists. A consumer product that often answers "we
don't know" is a tricky thing to ship. Also the right thing. This system is
built to give that answer.

### Intended audience

A nervous flier who's technical enough to want to see the reasoning. Not a
dispatcher. That choice permeates the design: the system reports rather than
recommends, and takes the worse of two disagreeing readings because a
passenger and a dispatcher would have opposite defaults.

---

## The one rule the system follows

> **Absence of data is never smooth air.**

There is no default severity anywhere in the system. A route on which nobody
filed a report reads `unresolved`, and the interface reports which kind of
silence it encountered: no pilot reports, no forecast, or a forecast that
covered the route on the ground but not at cruise altitude.

Every other design choice follows from that first one:

- coverage can only lower a corridor's score, never prune it, because the
  true corridor may be the one nobody observed
- disagreeing sources are both presented and the worse is used, because an
  average would match neither
- a failed data source is reported as such, with the cause named, never as
  quiet weather
- turbulence data has an explicit TTL and is never held for longer than that

---

## Architecture

Six roles. Two are language models, and both are at the edges. Everything
that decides anything is deterministic Python.

One of those two language models is specified but not implemented - a form
handles that today - so only one is running.

```
              Origin, destination, time
                        |
              +---------v----------+
              :    Trip parser     :  language model - NOT BUILT
              : free text -> search:  a form does this today
              +---------+----------+
                        |
  +---------------------v----------------------+
  |  Deterministic core - no model reaches this |
  |                                             |
  |   Corridor search  ->  Critic  ->  Reading  |
  |   four candidates      score       no       |
  |                        & prune     default  |
  +---------------------+----------------------+
                        |
              +---------v----------+
              |     Explainer      |  language model
              | restates, cannot   |
              | decide             |
              +---------+----------+
                        |
              +---------v----------+
              |     Validator      |  guardrail
              | rejected, never    |
              | repaired           |
              +---------+----------+
                        |
                     Reader
```

**The model doesn't touch the number.** The explainer is called after the
reading already exists; it writes prose about a value it cannot change.

### The reasoning layer

Corridor search is a bounded **Tree-of-Thought**. The path which an aircraft
actually flies is unknown at takeoff, so the system thinks through four
competing hypotheses rather than committing to one:

| Depth | Splits by | Candidates |
|---|---|---|
| 1 | corridor source | flown track, filed route, published airway, great circle |
| 2 | cruise altitude band | high / low, per surviving corridor |

A deterministic critic calculates a score for every candidate and stores a
decision and reason for each, **including the ones it rejects**:

```
score = 0.40*provenance + 0.25*geometry + 0.20*agreement + 0.15*coverage
```

Provenance matters more than geometry because how a corridor was derived
matters more than how it looks: flown track 1.00, filed route 0.75,
published airway 0.50, great circle 0.25. Coverage is given the lowest
weight on purpose, so thin observation only lowers confidence, never prunes
a candidate.

Beam width 2, depth limit 2. Both fit within a cost cap: every branch kept
alive at one depth costs metered API calls at the next.

### The guardrail

The explainer takes twelve structured facts and is only allowed to repeat
them. Those twelve are also where prompt injection would have to arrive,
and eleven of them are computed here: enums, integers, floats, and strings
built from validated airport codes. **There is no free-text field, so a
caller cannot reach the model with anything they wrote.** The two that can
carry outside text are the aircraft variant, passed through from the flight
data provider, and the plain summary, written here from external weather
readings. Both are shape-checked before the prompt is assembled: field
allowlist, enum membership, type and range, format patterns, and a
structural check for markup and role markers. The exposure is the data
providers, not the caller.

Even if that failed entirely, the reading was produced before the model was
called and is not an output of it. The worst case is a bad paragraph.

Its output is then validated against four rules: no invented severity, no
reassurance, caveats must survive, and **any failure discards the whole
paragraph rather than editing it** - because if the model believed the air
was smooth, that belief informs every sentence.

A discarded paragraph falls back to a deterministic summary. That's what a
reader sees when the explainer is off anyway.

---

## Repository layout

```
app/
  api.py                 FastAPI routes, SSE streaming
  runs.py                per-search record, metrics, origin resolution
  logging_setup.py       syslog handler, request ids, credential redaction
  web/
    service.py           orchestration: resolve -> search -> evidence -> explain
    static/              the search page and the status page
  reasoning/
    generator.py         corridor candidates from flight data
    critic.py            scoring, beam and dominance pruning
    controller.py        the expand/assess loop
    graph.py             the same loop as a LangGraph StateGraph
    explainer.py         the one language model call, and its validator
                         (no trip_parser.py - the role is specified in the
                          design documents and not implemented)
    fact_checks.py       shape checks on every field before the prompt
  retrieval/
    airports.py          IATA -> ICAO resolution
    aircraft_types.py    aircraft variant normalisation
    embedding.py         NTSB corpus embeddings and search
  sources/
    aeroapi.py           FlightAware: schedules, filed routes, flown tracks
    awc.py               Aviation Weather Center: PIREPs, G-AIRMETs
    gtg.py               NOAA gridded turbulence nowcast

scripts/
  serve.sh               run locally
  deploy.sh              test-gated rsync deploy
  ingest_ntsb.py         bulk NTSB ingest into SQLite
  embed_chunks.py        build the retrieval index
  beam_analysis.py       was the beam ever the deciding factor?
  dominance_analysis.py  is the 0.80 threshold load-bearing?
  check_edge.sh          hourly edge health check

tests/                   ~1,269 tests
```

---

## Setup

Python 3.12.

```bash
git clone https://github.com/anubisdar/turbulence-agent.git
cd turbulence-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configuration

Copy the example environment file and populate it with what you have:

```bash
cp turbulence-agent.env.example turbulence-agent.env
```

| Variable | Needed for | Without it |
|---|---|---|
| `AEROAPI_KEY` | FlightAware flight data | corridors fall back to great circle only |
| `ANTHROPIC_API_KEY` | the explainer | deterministic summaries only |
| `TURBULENCE_LOG_TRIP_CONTENT` | logging routes | off by default, deliberately |

The Aviation Weather Center API requires no key. **The system can run with
no key at all**, and it does: it produces the same readings, with worse
provenance and no model-written prose.

### Running

```bash
./scripts/serve.sh              # http://127.0.0.1:8000
```

Or directly:

```bash
uvicorn app.api:app --reload
```

### Testing

```bash
python3 -m pytest tests/ -q
```

The deploy script runs the full test suite first and **refuses to sync if
any test fails**.

---

## Usage

Type two airport codes. Three or four letters both work: `PIT` resolves to
`KPIT`, `ANC` to `PANC`, `NRT` to `RJAA`. The interface tells you what it
resolved to.

The result page has six tabs:

- **Agent Processing View** - every corridor considered, kept or pruned,
  with the reason
- **Trace** - the search step by step
- **Notes** - what each source said along the winning corridor
- **Overlap** - pairwise airspace overlap between candidates
- **Safety record** - NTSB history for the aircraft type
- **LLM debug** - the rules the model was given, the facts it was given,
  what it wrote, and whether the validator kept it

There is also a public
[status page](https://turbulence.adeptsecurity.net/status) that shows 30
days of behavior: resolution rates, timings, where the money goes, guardrail
rejections, and edge traffic.

### API

```bash
curl -X POST http://127.0.0.1:8000/api/search/corridors \
  -H 'Content-Type: application/json' \
  -d '{"origin": "PIT", "dest": "BOS", "include_explanation": true}'
```

---

## Evaluation

Everything evaluated here is **internal consistency, not correctness**.
Whether the winning corridor is the path the aircraft actually flew has not
been validated against held-out flown tracks, and that's an honest gap.

What has been evaluated:

| | |
|---|---|
| Resolution rate | 30-58% across 20-route runs - the variance is weather, not the agent |
| Median search | 14.2s, of which 11.5s is waiting on the flight data provider |
| Cost | ~9¢ per search, dominated by one 5¢ endpoint |
| Explainer acceptance | 80-100%, and the rejections were the interesting part |
| Beam width | the deciding factor in ~3% of prune decisions |
| Dominance threshold | load-bearing - smallest triggering overlap was 0.8040 |

The last two were already being logged by the system at zero marginal cost.
`scripts/beam_analysis.py` and `scripts/dominance_analysis.py` reproduce
them.

### The result worth reading about

In one build the explainer's measured acceptance rate was **91%** and its
true acceptance rate was **100%**. Every rejection was the validator being
wrong - four distinct classes of false positive, each an accurate paragraph
discarded for naming a severity in order to deny it:

> "There is no basis in the available data to characterize conditions as
> light, moderate, or severe."

The project's own argument, in the model's mouth, thrown away for using the
words to say it. A rejection and a wrong rejection are treated the same, so
no automated check could find them. Reading the discarded text could - which
is why rejected output is kept and shown rather than silently replaced.

---

## Safety and human oversight

The agent has no actuators. It doesn't book, cancel, or recommend. **Every
`unresolved` reading is a deferral**: handing the judgement back instead of
making up an answer to pad the output.

Beyond that, a person is required in five cases: a rejected model output,
two sources disagreeing, a source failing, a threshold being changed, and a
fact failing its shape check.

Deployed with a web application firewall that watched for five days before
it was given permission to reject anything, a CAPTCHA in front of every
metered search, split rate limits, and a spend cap that lives outside the
system so no bug inside it can raise the ceiling.

---

## Known limitations

- **The corridor is inferred, not known.** Every turbulence lookup is
  conditioned on a path the aircraft may not fly, and nothing downstream can
  fix a wrong corridor. This is the deepest limitation and the reason the
  system reports rather than recommends.
- **No held-out validation** against flown tracks.
- **Beam width and the dominance threshold** are evaluated but not tuned.
- **Departures more than ~6 hours ahead** cannot be forecast; the agent says
  so rather than answering about now.
- **Nonstop routes only** - a pair with no nonstop service reports that the
  geometric path is not a real route.
- **The trip parser is specified but not written.** It is a designed role
  with no code behind it; the search form covers the same ground. Building
  it would place a second language model at the input edge, where a wrong
  parse degrades an explanation rather than corrupting a score.

---

## Built with

Python 3.12 · FastAPI · LangGraph · SQLite · pyproj + shapely ·
sentence-transformers (BAAI/bge-small-en-v1.5) · Claude Sonnet 5 ·
FlightAware AeroAPI · Aviation Weather Center API · NOAA GTG · NTSB CAROL

---

## License

MIT - see [LICENSE](LICENSE).

The code is MIT. The data it reads is not mine to license: FlightAware
AeroAPI and MaxMind GeoLite2 carry their own terms, and neither is
redistributed here.

© 2026 Adept Security LLC
