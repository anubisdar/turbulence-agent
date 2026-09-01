# Roadmap

This started as a graduate capstone and is maintained by one person. What
follows is what I intend to work on and what I know is missing, not a
schedule. Nothing here carries a date.

The ordering principle is the same one the system uses: things that make
the project's own claims more accurate come before things that make it
do more.

---

## Known gaps

These are places where the code is currently less good than the README
implies. They are listed first because that is the order I plan to fix
them in.

### The output validator has a fifth false-positive class

The README describes four classes of accurate paragraph wrongly discarded
by the explainer's validator. There is a fifth, found in production on
31 August 2026, and it is the same construction as the first:

> when sources disagree, the more severe reading is used rather than an
> average

That is a comparative describing the conflict rule, not an FAA severity
level. The clause-scoping fix that resolved the earlier cases handled the
phrasings that existed rather than the pattern behind them.

The interesting part is that it appeared *after* the fix, from the model
correctly restating the project's own policy. A guardrail firing and a
guardrail being right stay different facts even once you have repaired it
once.

**Planned:** extend the comparative exemption to cover a severity word
used adjectivally on `reading`, `source` or `value`, add the sentence as
a regression test, and correct the count in the README.

### Part of the test suite reaches the network

`tests/conftest.py` blocks live calls by patching httpx transports.
`app/sources/aeroapi.py`, `app/sources/gairmet.py` and
`app/web/turnstile.py` use `urllib.request.urlopen` directly, so tests
touching those paths are not covered by the guard.

Measured: the same test passed four times and failed once in five
consecutive runs, with `URLError: [Errno 104] Connection reset by peer`.
Arming a `urllib` guard turns two intermittent failures into sixty-two
deterministic ones, because those sixty-two genuinely depend on the
calls.

**Planned:** record fixtures for the three paths, then arm the guard.
Until that lands, the reported pass count is conditional on network
conditions - which is a weaker claim than it reads as.

### The API and the service disagree on four defaults

`CorridorSearchBody` sets every field explicitly, so its values override
`SearchRequest`'s. Four have drifted: `depth_limit` maximum, `max_tool_calls`,
`use_graph` and `include_explanation`. A caller who omits them gets
different behaviour from someone using the search form.

**Planned:** align them, add a test that diffs every shared field between
the two models programmatically, and generate `docs/openapi.json` at
build time with a test that fails when it drifts from the code.

---

## Planned features

### Departure times in the airport's local time

Everything is UTC today. That is correct internally - aviation runs on
UTC and so do the data sources - and wrong at the interface, where a
passenger has a boarding pass that says 4:15 PM and the form wants 20:15.

The design decision that matters: a departure time is local to the
*origin airport*, not to the person searching. Someone in Pittsburgh
looking up a Tokyo departure wants JST. A picker defaulting to the
browser's timezone would be wrong for most searches.

**Shape:** a toggle above the time field switching between UTC and the
origin airport's local time. The field converts; the form still sends
UTC; the server keeps its single unambiguous representation.

**The cost:** 208 timezone mappings added to
`app/retrieval/airports.py`, and they have to be right. Arizona has no
DST, Indiana is split, and the international entries are where mistakes
hide. A wrong mapping silently shifts a departure by an hour, which is
exactly the class of quiet wrong answer this project exists to avoid.

### The trip parser

Specified since Module 3 and never built. The README and the architecture
diagram both say so, and this is the role that would make them agree.

Free text in - *"Pittsburgh to Boston next Tuesday morning"* - and a
structured search out. It sits at the input edge deliberately: a wrong
parse degrades an explanation rather than corrupting a score, which is
what makes it safe to be a language model at all.

What it needs beyond a prompt:

- A structured output contract, so a parse either produces a valid search
  or fails visibly. No partial parses.
- Resolution shown to the reader, the way airport codes already are. If
  it read "next Tuesday" as a date, say which date.
- The same shape checks the explainer's facts receive, because the parse
  output becomes API input.
- A refusal path. "Somewhere warm in March" is not a trip, and the honest
  answer is to say so rather than guess.

### Connection-aware corridor search

Nonstop routes only today. A pair with no nonstop service is told that
the geometric path is not a real route, which is honest and unhelpful.

The useful version searches the long-haul leg: San Diego to Tokyo has no
nonstop, but KLAX to RJTT is where the turbulence matters.

The code change is small. The design problem is reporting on a different
route than the one asked about, clearly enough that nobody misreads it.

### Search by flight number

You know your flight number. You do not know your corridor.

Three pieces: `flight_by_ident()` on the AeroAPI client, an optional
ident in `_get_flight` that skips list-and-pick-by-time, and interface
changes, since origin and destination then come from the flight.

The open question is what it does to the reasoning. With a known flight
number the corridor search largely collapses - from four competing
hypotheses to one known path. That is probably the more useful product
and definitely the less interesting one to look at.

### A third search depth, conditionally

Designed and not built. It would split a corridor longitudinally, and
only when the evidence says the route is not uniform: partial coverage,
reports disagreeing inside one corridor, or a forecast overlapping only
part of it. All three signals are already computed.

Conditional matters. Splitting a uniformly observed route produces two
children with the same answer, so an unconditional third level would burn
reasoning to restate the second.

It costs nothing in API calls - it splits a corridor whose data is
already fetched. What it buys is the ability to say *the first half is
rough and the rest is unknown* rather than averaging the two.

---

## Measurement work

None of this changes what the system does. It changes how much anyone is
entitled to claim about it, which is why it is on the roadmap at all.

### Held-out validation against flown tracks

The most valuable single item here.

Everything currently measured is internal consistency: the code does what
its tests say. Nothing tests whether the winning corridor is the path the
aircraft actually flew.

The experiment is to withhold the flown track from the generator, run the
search, and compare the winner against the withheld track. What makes it
awkward is that the flown track is also the highest-provenance input, so
removing it changes what the search can find - the honest comparison is
between the best non-track corridor and the track, which answers a
slightly narrower question than "is the agent right."

### Re-calibrating two thresholds

Both are measured, neither is tuned.

**Beam width** is the deciding factor in about 3% of prune decisions: one
of 31 separated two corridors by 0.0255.

**The dominance threshold** is load-bearing. The smallest overlap that
ever triggered it was 0.8040 against a line at 0.80. Move the line to
0.81 and that corridor survives.

`scripts/beam_analysis.py` and `scripts/dominance_analysis.py` reproduce
both from decision logs the system already keeps. What is missing is more
searches to run them against.

### Caching route geometry

Cost does not survive scale: about nine cents a search, dominated by one
five-cent endpoint called first every time. The fix cache is the start of
this - eight calls cold against four to six warm - but it caches
waypoints rather than corridors.

Turbulence cannot be cached, because a stale reading presented as current
is a confident wrong answer. Route geometry can, because waypoint
positions do not change.

---

## Not planned

**A recommendation.** The system reports and does not recommend, because
a recommendation compounds an inferred corridor and sparse data into one
confident output. That is a design position rather than a missing
feature.

**A severity default.** There is none anywhere in the code and there will
not be. A route with no evidence reads `unresolved`.

**More turbulence sources to raise the resolution rate.** The 30-58%
range measures how much weather data existed, not how well the agent
performed. Adding sources to make the number look better would be
optimising the wrong thing.

---

## Contributing

Issues and pull requests are welcome, particularly on the known gaps
above. There is no contribution process beyond opening one, and I make no
promises about response times - see [SECURITY.md](SECURITY.md) for the
same caveat applied to vulnerability reports.

If you are reporting a bug, the most useful thing you can include is what
you expected the system to say rather than what it said. Most of the
interesting failures in this project have been cases where it produced a
plausible answer that was quietly wrong, and those are hard to spot from
the outside.
