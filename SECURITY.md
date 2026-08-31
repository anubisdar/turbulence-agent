# Security policy

This is a personal capstone project by me, not a funded product. 
The policy below is an honest reflection of that, not some unrealistic response time that nobody can be held to.

## Reporting a vulnerability

Please use **GitHub's private vulnerability reporting** on this repository
(Security tab -> Report a vulnerability). This will keep your report
private until I have a fix, and I only check that channel for
vulnerability reports.

What helps:

- What you did, what happened, and what you expected instead
- Whether you tested against the live deployment or a local checkout
- Any request or response that shows the behaviour

What I will do:

- Acknowledge within about a week. This is a side project, not something
  I'm on call for.
- Tell you honestly whether I consider it a vulnerability or not, with an
  explanation if not.
- Credit you in the fix commit unless you'd prefer I didn't.

There is no bug bounty.

## Please do not test against the live deployment

`turbulence.adeptsecurity.net` is a real deployment using a metered API.
**Every search costs about nine cents of my own money**, and an automated
tool making a few hundred requests would cost more than this project has
spent in a month.

That's why there's a CAPTCHA in front of every search, split rate limits,
and a spend cap at the provider. Those are all there to bound this
exactly, and I would rather you not test whether they hold by spending the
budget.

**Run it locally instead.** The system works with no API keys at all - the
readings still resolve, with worse route provenance and deterministic
prose instead of model-written prose:

```bash
git clone https://github.com/anubisdar/turbulence-agent.git
cd turbulence-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
./scripts/serve.sh
```

If a finding genuinely needs the live instance to demonstrate, please say
so in the report and I will arrange it.

## Scope

**In scope**

- The application in `app/`, including the API, the search page and the
  status page
- The guardrails: input shape checks in `app/reasoning/fact_checks.py` and
  output validation in `app/reasoning/explainer.py`
- Credential handling and log redaction in `app/logging_setup.py`
- The deployment scripts in `scripts/`

**Out of scope**

- Vulnerabilities in the upstream data providers (FlightAware, the
  Aviation Weather Center, NOAA, NTSB, MaxMind, Cloudflare). Please report
  those to them.
- Anything requiring physical or shell access to the instance
- Denial of service. Volumetric attacks at the network layer are absorbed by AWS Shield Standard, which is on by default. Application- layer floods from one address are bounded by per-IP rate limits and the CAPTCHA. Neither helps against a distributed flood: the rate limiter keys on the remote address, so a thousand addresses each get their full budget. The origin is reachable directly - There is no Akamai or Cloudflare proxy - and a single instance with one worker would fall over. I know, and that risk is absolutley accepted rather than mitigated.
- Missing headers or configuration that carry no exploitable consequence

## Known and accepted

Stated here so nobody wastes time on something I've already thought
through and decided on.

**The GeoIP filter is not a security boundary.** It's a filter that
reduces noise from foreign scanners. A hosted scanner within the allowed
region walks straight through it, and treating it as protection would be
overclaiming.

**The web application firewall runs a baseline rule set.** Coraza with
OWASP CRS 4.25, in blocking mode after five days of profiling.
Detection logs showed no rule matching a legitimate search.

**The turbulence reading is not safety-of-life information.** It's a
consumer-grade summary of public weather data. Nobody should use it to
make an operational aviation decision, and the interface says so.

**The corridor is inferred, not known.** Every reading is conditioned on a
path the aircraft may not fly. This is a correctness limitation, not a
security one, but it's the deepest limitation in the system and it's
documented in the README.

**Proof-of-work challenges were considered and rejected.** Anubis and
similar tools make sense when an attacker's gain per request is small.
Here the gain per request is nine cents of metered budget, so a CPU toll
is trivial compared to the prize, and native solvers can compute those
challenges faster than the JavaScript that a real visitor's browser runs,
so the asymmetry is reversed.

## What is deliberately not in this repository

No API keys, no `.env` file, no TLS keys, no Caddy configuration, no
MaxMind databases (licensed and not redistributable), and no built
retrieval database. `turbulence-agent.env.example` lists every variable
the system reads.

The git history has been checked for credentials. If you find one I
missed, that is exactly the kind of report I'm looking for.

## Supported versions

The `main` branch is the only supported version. There are no releases and
no backports.

---

© 2026 Adept Security LLC
