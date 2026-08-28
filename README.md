# Turbulence-aware flight ranking

An agent that ranks flights by expected turbulence - and says plainly when
nothing is known.

Live: **[turbulence.adeptsecurity.net](https://turbulence.adeptsecurity.net)**

---

## The problem

I am an anxious flier. Before a flight I want to know one thing: is this
going to be bumpy?

Every input to that question is public and free. Pilots file turbulence
reports in flight. The Aviation Weather Center issues turbulence forecasts.
Filed routes and flown tracks are published. None of it answers the
question, because turning three sparse, disagreeing sources into a single
answer is judgement work - and on most routes the honest answer is that
nobody knows.

That last part is why no product does this. A consumer product that
frequently returns "we don't know" is a hard thing to ship. It is also the
correct answer, and this system is built to give it.

**Intended user:** a nervous passenger who is technical enough to want to
see the reasoning, not a dispatcher. That choice shows up throughout - the
system reports rather than recommends, and it takes the worse of two
disagreeing readings because a passenger and a dispatcher would want
opposite defaults.

---

## The one rule everything follows

> **Absence of data is never smooth air.**

No severity default exists anywhere in the system. A route nobody has
reported on reads `unresolved`, and the interface says *which kind* of
silence it hit - no pilot reports, no forecast, or a forecast that covered
the route on the ground but not at cruise altitude.

Every other design decision follows from this one:

- Coverage can **lower** a corridor's score but can never **prune** it,
  because the true corridor may be the one nobody observed.
- Disagreeing sources are both shown and the worse is used, because an
  average would match neither.
- A failed data source is reported as `degraded` with the cause named,
  never as quiet weather.
- Turbulence data carries an explicit TTL and is never cached beyond it.

---

## Architecture

Six roles. **Two are language models, and both sit at the edges.**
Everything that decides anything is deterministic Python.

One of those two, the trip parser, is **specified but not implemented** - a
form does that job today. So one language model runs.

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

**The model never touches the number.** By the time the explainer is
called, the reading already exists - it writes prose about a value it
cannot change.

### The reasoning layer

The corridor search is a bounded **Tree-of-Thought**. Which path an
aircraft actually flies is unknown before departure, so the system reasons
over four competing hypotheses rather than committing to one:

| Depth | Splits by | Candidates |
|---|---|---|
| 1 | corridor source | flown track, filed route, published airway, great circle |
| 2 | cruise altitude band | high / low, per surviving corridor |

A deterministic critic scores every candidate and records a decision and a
reason for each, **including the ones it rejects**:

```
score = 0.40*provenance + 0.25*geometry + 0.20*agreement + 0.15*coverage
```

Provenance dominates because how a corridor was derived matters more than
how it looks: flown track 1.00, filed route 0.75, published airway 0.50,
great circle 0.25. Coverage is weighted lowest deliberately, so thin
observation lowers confidence without ever eliminating a candidate.

Beam width 2, depth limit 2. Both fit a cost cap - every branch kept alive
at one depth costs metered API calls at the next.

### The guardrail

The explainer receives twelve structured facts and may only restate them.
Its output is checked afterwards against four rules: no invented severity,
no reassurance, caveats must survive, and **any failure discards the whole
paragraph rather than editing it** - because if the model wrongly believed
the air was smooth, that belief shapes every sentence.

A discarded paragraph falls back to a deterministic summary, which is what
a reader sees with the explainer switched off anyway.

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

tests/                   ~1,068 tests
```

---

## Setup

Python 3.12.

```bash
git clone https://github.com/<user>/turbulence-agent.git
cd turbulence-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configuration

Copy the example environment file and fill in what you have:

```bash
cp turbulence-agent.env.example turbulence-agent.env
```

| Variable | Needed for | Without it |
|---|---|---|
| `AEROAPI_KEY` | FlightAware flight data | corridors fall back to great circle only |
| `ANTHROPIC_API_KEY` | the explainer | deterministic summaries only |
| `TURBULENCE_LOG_TRIP_CONTENT` | logging routes | off by default, deliberately |

The Aviation Weather Center API needs no key. **The system runs without
any key at all** - it produces the same readings, with worse provenance and
no model-written prose.

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

The deploy script runs the full suite first and **refuses to sync if any
test fails**.

---

## Usage

Enter two airport codes. Three letters or four both work - `PIT` resolves
to `KPIT`, `ANC` to `PANC`, `NRT` to `RJAA` - and the interface shows what
it resolved to.

The result page has six tabs:

- **Agent Processing View** - every corridor considered, kept or pruned,
  with the reason
- **Trace** - the search step by step
- **Notes** - what each source said along the winning corridor
- **Overlap** - pairwise airspace overlap between candidates
- **Safety record** - NTSB history for the aircraft type
- **LLM debug** - the rules the model was given, the facts it was given,
  what it wrote, and whether the validator kept it

There is also a public [status page](https://turbulence.adeptsecurity.net/status)
showing thirty days of behaviour: resolution rates, timings, where the
money goes, guardrail rejections, and edge traffic.

### API

```bash
curl -X POST http://127.0.0.1:8000/api/search/corridors \
  -H 'Content-Type: application/json' \
  -d '{"origin": "PIT", "dest": "BOS", "include_explanation": true}'
```

---

## Evaluation

Everything measured here is **internal consistency, not correctness**.
Whether the winning corridor is the path the aircraft actually flew has not
been validated against held-out flown tracks. That is the honest gap.

What has been measured:

| | |
|---|---|
| Resolution rate | 30-58% across twenty-route runs - the variance is weather, not the agent |
| Median search | 14.2s, of which 11.5s is waiting on the flight data provider |
| Cost | ~9¢ per search, dominated by one 5¢ endpoint |
| Explainer acceptance | 80-100%, and the rejections were the interesting part |
| Beam width | the deciding factor in ~3% of prune decisions |
| Dominance threshold | load-bearing - smallest triggering overlap was 0.8040 |

The last two came from logs the system was already keeping, at no cost.
`scripts/beam_analysis.py` and `scripts/dominance_analysis.py` reproduce
them.

### The result worth reading about

In one build the explainer's measured acceptance rate was **91%** and its
true acceptance rate was **100%**. Every rejection was the validator being
wrong - four distinct classes of false positive, each an accurate paragraph
discarded for naming a severity in order to deny it:

> "There is no basis in the available data to characterize conditions as
> light, moderate, or severe."

The project's own argument, in the model's words, thrown away for using the
words to make it. A rejection and a wrong rejection are counted the same,
so no automated check could have found them. Reading the discarded text
did - which is why rejected output is retained and shown rather than
silently replaced.

---

## Safety and human oversight

The agent has no actuators. It does not book, cancel, or recommend. **Every
`unresolved` reading is a deferral** - handing the judgement back rather
than manufacturing an answer to fill the gap.

Beyond that, five conditions require a person: a rejected model output, two
sources disagreeing, a source failing, a threshold being changed, and a
fact failing its shape check.

Deployed with a web application firewall that spent five days observing
before it was allowed to refuse anything, a CAPTCHA in front of every
metered search, split rate limits, and a spend cap that lives outside the
system - so no bug inside it can raise the ceiling.

---

## Known limitations

- **The corridor is inferred, not known.** Every turbulence lookup is
  conditioned on a path the aircraft may not fly, and nothing downstream
  can fix a wrong corridor. This is the deepest limitation and the reason
  the system reports rather than recommends.
- **No held-out validation** against flown tracks.
- **Beam width and the dominance threshold** are measured but not tuned.
- **Departures beyond about six hours** cannot be forecast; the agent says
  so rather than answering about now.
- **Nonstop routes only** - a pair with no nonstop service reports that the
  geometric path is not a real route.

---

## Built with

Python 3.12 * FastAPI * LangGraph * SQLite * pyproj + shapely *
sentence-transformers (BAAI/bge-small-en-v1.5) * Claude Sonnet 5 *
FlightAware AeroAPI * Aviation Weather Center API * NOAA GTG * NTSB CAROL

---

## License

MIT - see [LICENSE](LICENSE).

The code is MIT. The data it reads is not mine to license: FlightAware
AeroAPI and MaxMind GeoLite2 carry their own terms, and neither is
redistributed here.

© 2026 Adept Security LLC
