# Security policy

This is a personal capstone project maintained by one person, not a funded
product with a security team. The policy below reflects that honestly
rather than promising a response time nobody is on call to meet.

## Reporting a vulnerability

Use **GitHub's private vulnerability reporting** on this repository
(Security tab -> Report a vulnerability). That keeps the report private
until there is a fix, and it is the only channel I monitor for this.

What helps:

- What you did, what happened, and what you expected instead
- Whether you tested against the live deployment or a local checkout
- Any request or response that shows the behaviour

What I will do:

- Acknowledge within about a week. This is a side project; I am not on
  call.
- Tell you plainly whether I consider it a vulnerability, and why if I do
  not.
- Credit you in the fix commit unless you would rather I did not.

There is no bug bounty. I have no budget for one.

## Please do not test against the live deployment

`turbulence.adeptsecurity.net` is a real deployment on a metered API.
**Each search costs about nine cents of my money**, and an automated tool
making a few hundred requests would cost more than this project has spent
in a month.

That is why there is a CAPTCHA in front of every search, split rate limits,
and a spend cap at the provider. Those controls exist to bound exactly this,
and I would rather you did not test whether they hold by spending the
budget.

**Run it locally instead.** The system works with no API keys at all - the
readings still resolve, with worse route provenance and deterministic prose
instead of model-written prose:

```bash
git clone https://github.com/anubisdar/turbulence-agent.git
cd turbulence-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
./scripts/serve.sh
```

If a finding genuinely requires the live instance to demonstrate, say so in
the report and I will arrange it.

## Scope

**In scope**

- The application in `app/`, including the API, the search page and the
  status page
- The guardrails: input shape checks in `app/reasoning/fact_checks.py` and
  output validation in `app/reasoning/explainer.py`
- Credential handling and log redaction in `app/logging_setup.py`
- The deployment scripts in `scripts/`

**Out of scope**

- Vulnerabilities in the upstream data providers (FlightAware, the Aviation
  Weather Center, NOAA, NTSB, MaxMind, Cloudflare). Report those to them.
- Anything requiring physical or shell access to the instance
- Denial of service by volume. It is a single instance behind a rate limit
  and I already know it would fall over.
- Missing headers or configuration that carry no exploitable consequence

## Known and accepted

Stated here so nobody spends time on something I have already decided
about.

**The GeoIP filter is not a security boundary.** It reduces noise from
foreign scanners. A hosted scanner inside the allowed region walks straight
past it, and treating it as protection would be overclaiming.

**The web application firewall runs a generic rule set.** Coraza with OWASP
CRS 4.25, in blocking mode since 27 August 2026 after five days of
detection logs showed no rule matching a legitimate search. It will not
stop something written for this application specifically.

**The turbulence reading is not safety-of-life information.** It is a
consumer-grade summary of public weather data. Nobody should use it to make
an operational aviation decision, and the interface says so.

**The corridor is inferred, not known.** Every reading is conditioned on a
path the aircraft may not fly. This is a correctness limitation rather than
a security one, but it is the deepest limitation in the system and it is
documented in the README.

**Proof-of-work challenges were considered and rejected.** Anubis and
similar tools make sense when an attacker's gain per request is small. Here
the gain per request is nine cents of metered budget, so a CPU toll is
trivial against the prize - and native solvers compute those challenges
faster than the JavaScript a real visitor's browser runs, so the asymmetry
runs the wrong way.

## What is deliberately not in this repository

No API keys, no `.env` file, no TLS keys, no Caddy configuration, no
MaxMind databases (licensed and not redistributable), and no built
retrieval database. `turbulence-agent.env.example` lists every variable the
system reads.

The git history has been checked for credentials. If you find one I missed,
that is exactly the kind of report I want.

## Supported versions

The `main` branch is the only supported version. There are no releases and
no backports.

---

© 2026 Adept Security LLC
