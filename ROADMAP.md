# Roadmap

This started as an executive education program capstone and is maintained by one person. What
follows is what I intend to work on and what I know is missing, not a
schedule. Nothing here carries a date.

---

## Planned features

Nothing queued right now. Everything below "Recently shipped" was still
open the last time this list was written; see "Measurement work" for what
is actually left.

---

## Recently shipped

### Weekly health check, automated and emailed

Built. `scripts/weekly_check.sh` runs the full pytest suite, refreshes the
G-AIRMET probe and re-runs the upstream contract tests against it,
validates end-to-end against fixtures at no API cost, and runs a
dependency vulnerability audit - then emails one pass/fail report via
`scripts/send_weekly_report.py`. This is what actually catches an
upstream API quietly changing its response shape while the fixtures keep
encoding the old belief (see `test_upstream_contracts.py`'s own
docstring), which nothing in the per-commit suite can, since that suite
is deliberately network-blocked. Optional tiers that cost real money
against metered APIs - a live AeroAPI probe, a live validation run, a
20-route load test, a red-team pass - stay opt-in; the free tier is what
runs on the cron.

It runs from the deployed EC2 instance now, against itself, rather than
from the workstation that pushes code to it. The cron sources a narrow,
SMTP-only secrets file with no access to the app's own API keys, and
audits dependencies through an isolated venv (`.venv-checks`) built
independently of the app's own (`AUDIT_VENV`) - so a test-only package
landing in the checker never gets mistaken for one running in
production, and the audit checks what production actually has installed
rather than what the checker does.

### Deploy-time safety fixes

Three real bugs, all found by running the documented recipes end to end
rather than by reading them. `deploy.sh`'s own local pytest gate
reported false failures whenever the calling shell had ever sourced the
app's secrets, for the same reason two other scripts already had to be
fixed the same way: real Turnstile keys sitting in the environment make
the suite's in-process test client start demanding a challenge no test
sends. A remote `rsync --delete` would have silently deleted
`.venv-checks/` and `reports/` on the very next ordinary deploy, since
both are created by the weekly checker after a deploy, never by one, and
neither was excluded. And `pip-audit` would have quietly audited the
checker's own isolated dependencies instead of the application's real
ones. None of the three showed up from reading the scripts - only from
running the exact command the documentation told someone to run.

### Connection-aware corridor search

Built. A pair with no nonstop service used to get told exactly that and
nothing more - honest, and unhelpful, since the geometric line between
the two airports isn't a route anyone flies. San Diego to Tokyo has no
nonstop, but KLAX to RJTT, the long-haul leg of a real SAN-LAX-RJTT
itinerary, is where the turbulence actually is.

`AeroAPIClient.route_or_long_haul_leg` answers both questions - is there
a nonstop, and if not, is there a connecting itinerary worth searching
instead - from the single AeroAPI call the nonstop check already made.
The routing endpoint returns full itineraries broken into segments, so a
connection's other legs are sitting in the same response that said there
was no nonstop; a second call would have asked the same question twice.

The design problem this item flagged - reporting on a different route
than the one asked about, clearly enough that nobody misreads it - turned
out to be the real work. `CorridorGenerator._get_flight` records the swap
as `route_substitution` (requested pair, searched pair), which reaches
the response as its own field, not just as prose in a notes list, and a
deterministic note (`_route_substitution_note` in `app/web/service.py`,
same pattern as `_partial_route_note`) says so independently of whether
the explainer runs. The route label on the page itself shows both pairs
side by side rather than silently swapping one for the other.

The harder problem underneath that one was picking the right leg without
data to rank it. AeroAPI's own itineraries commonly omit a leg's filed
distance, and a connection's segments are typically listed short-leg
first - so "just pick a segment" would systematically return the
regional feeder into the hub, not the long-haul leg, which is exactly the
"Dash 8 turboprop returned as the reference for Seattle to Tokyo" bug the
nonstop filter already exists to prevent (see `flights_between`'s own
docstring). Rather than repeat that mistake with a different data source,
a leg is only ever picked when at least one candidate reports a real
distance to rank against the others; with no distance data at all, the
search declines to guess and falls back to the pair exactly as it behaved
before this feature existed - the honest gap, not a confident wrong
answer.

### Local departure times

Built, though not the shape originally planned here - that plan predated
the trip chat, and described a toggle above a time field that no longer
exists. The interface takes departure time through conversation now, so
the redesign moved the whole thing into that layer instead.

The model (`app/web/tripchat.py`) is told to record a departure time
exactly as the traveler stated it, on their own clock, and never to
convert it itself - "4:15 PM" is recorded as "4:15 PM," not silently
turned into a UTC guess sitting in front of the one that's actually
checked. `respond()` does the real conversion, once the origin airport
has resolved, using a new IANA timezone table in `app/retrieval/airports.py`
(`timezone_for`, `local_to_utc`) keyed to the same codes `resolve_airport`
already produces - a guessed (ASSUMED) airport code never gets a timezone
entry, on purpose, so one uncertain guess is never compounded with a
second. `zoneinfo` (stdlib) resolves the DST offset from the date itself,
so Arizona's no-DST `America/Phoenix`, Indiana's
`America/Indiana/Indianapolis`, and the international entries that don't
follow their country's biggest-city zone (Hanoi is `Asia/Ho_Chi_Minh`;
Bali is `Asia/Makassar`) all convert correctly rather than by a Mountain
or Eastern approximation.

The resolution is shown back to the reader as a note - "4:15 PM local
time at KPIT on 2026-10-01 is 20:15 UTC" - the same transparency
`Airport.note()` already gives a guessed airport code. A departure time
given before the origin resolves is carried through unconverted rather
than guessed at, and picked up once the origin is known. An airport with
no timezone entry (an ASSUMED guess, or a real gap in the table) falls
back to treating the stated time as UTC, with a note saying so plainly -
an honest degradation, not a silent one. A bare clock time with no date
is left alone entirely; `_target_time` in `app/web/service.py` already
rolls a dateless UTC time to its next occurrence, and guessing a
reference day just to resolve a timezone offset would be a new source of
the exact error this feature exists to remove.

A direct API caller (not going through the chat) is unaffected:
`SearchRequest.departure_time` is still UTC by contract, same as always.

### A third search depth, conditionally

Built. `CorridorGenerator._longitudinal_branches` splits a surviving
depth-2 corridor lengthwise, but only when the evidence already gathered
for it says the route is not uniform: partial pilot-report coverage,
reports that disagree with each other, or a forecast polygon that covers
only one half. Any one signal alone is enough; a uniformly observed route
never splits, so a search never burns a level restating the same answer
twice.

It costs nothing in API calls. A split child's evidence is re-derived
from its parent's already-fetched reports and advisories
(`evidence_from_raw()` in `app/reasoning/evidence.py`), filtered against
the smaller shape - never a second fetch. The `_split_children` set on
the generator is what routes a child into that free reuse path instead of
a real gather.

A split corridor winning the search is flagged explicitly - the reading
is prefixed with which half it covers - so it is never presented as
covering the whole trip. That note (`_partial_route_note` in
`app/web/service.py`) is deterministic, not dependent on the explainer
running.

`MAX_IMPLEMENTED_DEPTH` and the interface's depth control both moved from
2 to 3 to match.

### The trip parser, as a conversational intake layer

Built, though not quite as originally specified. Rather than a one-shot
parse of a single free-text message, it's a back-and-forth
(`app/web/tripchat.py`) that resolves the same origin/destination/date/
time fields the form always took, plus an optional flight number. Forced
tool-use rather than a parsed free-text response, so there's no partial
parse to degrade gracefully from. Every airport code the model proposes
is checked against `resolve_airport()` before it can reach a search -
the refusal path this item asked for, just enforced per-field rather than
on the whole trip at once. Completion is decided in code, never trusted
from a model-returned boolean.

Not yet done: a held-out measurement of how often a real conversation
resolves to the trip a human reader would have picked. See "Measurement
work" below.

### Search by flight number

Built. `flight_by_ident()` on the AeroAPI client, plus three situations
handled in the intake layer rather than the form: a number with no
airport known yet is looked up to find its route; a number alongside
known airports is looked up to check the real route and only warns on a
mismatch rather than unpinning it; a trip that completes with no number
given gets a short list of real upcoming flights on the resolved route
offered back. All three are optional and never block a trip on their
own.

The reasoning question this item raised turned out not to need an
answer: a pinned flight number doesn't collapse the corridor search to
one hypothesis. It's a hint the reference-flight matcher uses when
picking among segments (`_match_flight_number`), and the matcher already
falls back safely if the pin doesn't actually match - the four-hypothesis
search still runs underneath it.

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

### Held-out evaluation of the trip chat

No labelled set exists yet for how often the conversational intake layer
(`app/web/tripchat.py`) resolves a real or realistic conversation to the
trip a human reader would have picked. The extraction contract is
enforced - forced tool-use, every proposed airport code checked against
`resolve_airport()` - but enforcement of the contract isn't the same
claim as accuracy of the extraction, and nobody has measured the second
one. Needs a labelling pass and a definition of "close enough" for a
fuzzy city name before it can run.

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

Issues and pull requests are welcome, particularly on anything above.
There is no contribution process beyond opening one, and I make no
promises about response times - see [SECURITY.md](SECURITY.md) for the
same caveat applied to vulnerability reports.

If you are reporting a bug, the most useful thing you can include is what
you expected the system to say rather than what it said. Most of the
interesting failures in this project have been cases where it produced a
plausible answer that was quietly wrong, and those are hard to spot from
the outside.
