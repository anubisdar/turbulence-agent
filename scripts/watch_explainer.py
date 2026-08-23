#!/usr/bin/env python3
# install-to: scripts
"""
Watch the language model's exchanges as they happen.

There is exactly one place a model speaks in this system, and it is the only
component whose output cannot be predicted from its input. Everything else
can be reasoned about by reading the code; this can only be observed.

What this shows, per search: the facts the model was given, the paragraph it
wrote, whether the validator accepted it, and if not, which rule it broke and
what the discarded text said. Plus a running acceptance rate, because the
interesting question is not whether a single response was rejected but
whether the rejections have a pattern.

They do. In production the model reaches for reassurance almost exclusively
on routes where nothing is known - it writes "smooth" or "light" when the
evidence holds neither. Watching twenty searches makes that visible in a way
that reading twenty log lines does not.

REQUIRES the prompt and response to be logged, which is off by default
because the facts carry the route:

    sudo systemctl edit turbulence-agent      # or edit the env file
        TURBULENCE_LOG_EXPLAINER_IO=1
    sudo systemctl restart turbulence-agent

Usage:
    python3 scripts/watch_explainer.py                  # follow live
    python3 scripts/watch_explainer.py --since "1 hour ago"
    python3 scripts/watch_explainer.py --rejected-only  # study the failures
    python3 scripts/watch_explainer.py --no-facts       # verdicts only
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
AMBER = "\033[33m"
RED = "\033[31m"
BLUE = "\033[36m"
PURPLE = "\033[35m"

#: Anchored on `req=` rather than on the start of the line, because the
#: same logger emits two different shapes. The syslog handler omits the
#: timestamp - journald supplies its own - while the stderr fallback
#: includes it:
#:
#:   INFO req=7e4f turbulence-agent.explainer explainer prompt facts="..."
#:   21:11:51 INFO    req=7e4f turbulence-agent.explainer explainer prompt ...
#:
#: An earlier version required the leading timestamp and therefore matched
#: nothing at all on a real deployment, silently.
LINE = re.compile(
    r"req=(?P<req>\S+)\s+\S*explainer\s+(?P<message>explainer\s.*)$")

#: journald's own timestamp, when reading with -o short-iso.
STAMP = re.compile(r"^(?:\S*?T)?(?P<time>\d\d:\d\d:\d\d)")

#: `key=value` or `key="value with spaces"`, the format kv() emits. Quoted
#: values may contain escaped quotes and backslashes - a JSON payload does,
#: and so does any model response using quotation marks.
FIELD = re.compile(r'(\w+)=("((?:[^"\\]|\\.)*)"|\S+)')


def fields(message: str) -> dict:
    out = {}
    for key, raw, quoted in FIELD.findall(message):
        if raw.startswith('"'):
            out[key] = quoted.replace('\\"', '"').replace("\\\\", "\\")
        else:
            out[key] = raw
    return out


def width() -> int:
    return min(shutil.get_terminal_size((100, 24)).columns, 100)


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width() - len(indent),
                         initial_indent=indent, subsequent_indent=indent)


def render_facts(raw: str) -> list[str]:
    """The prompt, as the model received it."""
    try:
        facts = json.loads(raw)
    except (ValueError, TypeError):
        return [wrap(raw)]

    lines = []
    order = ["route", "reading", "pilot_reports", "forecast",
             "sources_disagree", "route_coverage_fraction",
             "corridors_considered", "corridors_kept", "search_was_truncated",
             "aircraft", "cruise_band", "plain_summary"]
    for key in order + [k for k in facts if k not in order]:
        if key not in facts:
            continue
        value = facts[key]
        if key == "plain_summary":
            lines.append(f"    {DIM}{key}{RESET}")
            lines.append(wrap(str(value), "      "))
        else:
            lines.append(f"    {DIM}{key:<24}{RESET} {value}")
    return lines


class Watcher:
    def __init__(self, args):
        self.args = args
        self.pending: dict[str, dict] = {}
        self.accepted = 0
        self.rejected = 0
        self.reasons: Counter = Counter()

    def handle(self, req: str, message: str, stamp: str) -> None:
        data = fields(message)

        # A new request means any earlier one is finished. Without this a
        # rejection whose discarded text never arrived would be held
        # forever and never printed.
        for other in [r for r in self.pending if r != req]:
            if self.pending[other].get("verdict"):
                self.emit(other)

        state = self.pending.setdefault(req, {"time": stamp})

        if message.startswith("explainer prompt"):
            state["facts"] = data.get("facts")
        elif message.startswith("explainer response"):
            state["text"] = data.get("text")
        elif message.startswith("explainer discarded text"):
            state["discarded"] = data.get("text")
            if state.get("verdict") == "rejected":
                self.emit(req)
        elif message.startswith("explainer output accepted"):
            state.update(verdict="accepted", **data)
            self.emit(req)
        elif message.startswith("explainer output rejected"):
            # Held rather than emitted: the discarded text is logged on the
            # next line, and it is the most interesting part of a rejection.
            state.update(verdict="rejected", **data)
        elif message.startswith("explainer call failed"):
            state.update(verdict="failed", **data)
            self.emit(req)

    def emit(self, req: str) -> None:
        state = self.pending.pop(req, {})
        verdict = state.get("verdict", "?")

        if verdict == "accepted":
            self.accepted += 1
            if self.args.rejected_only:
                return
            colour, label = GREEN, "ACCEPTED"
        elif verdict == "rejected":
            self.rejected += 1
            raw = str(state.get("reasons") or state.get("reason") or "?")
            for one in raw.split("; "):
                self.reasons[one.split(":")[0][:52]] += 1
            colour, label = AMBER, "REJECTED"
        else:
            colour, label = RED, "CALL FAILED"
            if self.args.rejected_only:
                return

        bar = "\u2500" * (width() - 2)
        print(f"\n{DIM}{bar}{RESET}")
        head = (f"{colour}{BOLD}{label}{RESET}  "
                f"{DIM}{state.get('time','')}  req={req}{RESET}")
        total = self.accepted + self.rejected
        if total:
            head += (f"  {DIM}|  {self.accepted}/{total} accepted "
                     f"({self.accepted / total:.0%}){RESET}")
        print(head)

        if state.get("reading"):
            print(f"  {DIM}reading{RESET} {BOLD}{state['reading']}{RESET}", end="")
            for key in ("words", "tokens_in", "tokens_out"):
                if state.get(key):
                    print(f"   {DIM}{key}{RESET} {state[key]}", end="")
            print()

        if state.get("facts") and not self.args.no_facts:
            print(f"\n  {BLUE}{BOLD}\u2192 given to the model{RESET}")
            for line in render_facts(state["facts"]):
                print(line)

        text = state.get("text") or state.get("discarded")
        if text:
            arrow = ("\u2190 written" if verdict == "accepted"
                     else "\u2190 written, then discarded")
            print(f"\n  {PURPLE}{BOLD}{arrow}{RESET}")
            print(wrap(text))

        if verdict == "rejected":
            raw = str(state.get("reasons") or state.get("reason") or "?")
            plural = "s" if "; " in raw else ""
            print(f"\n  {AMBER}{BOLD}\u2717 rule{plural} broken{RESET}")
            for one in raw.split("; "):
                print(wrap(one))
        elif verdict == "failed":
            print(f"\n  {RED}{BOLD}\u2717 the call itself failed{RESET}")
            print(wrap(str(state.get("error", "?"))))

    def summary(self) -> None:
        total = self.accepted + self.rejected
        if not total:
            print(f"\n{DIM}No explainer activity seen.{RESET}")
            print(f"{DIM}A search only calls the model when the explanation "
                  f"switch is on.{RESET}")
            return
        bar = "\u2550" * (width() - 2)
        print(f"\n{DIM}{bar}{RESET}")
        print(f"{BOLD}{self.accepted} of {total} accepted "
              f"({self.accepted / total:.0%}){RESET}")
        if self.reasons:
            print(f"\n{DIM}rules broken:{RESET}")
            for reason, count in self.reasons.most_common():
                print(f"  {AMBER}{count:>3}\u00D7{RESET}  {reason}")
            print(f"\n{DIM}Rejections cluster on routes where nothing is "
                  f"known: the model reaches for reassurance{RESET}")
            print(f"{DIM}precisely where there is least to say.{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="e.g. '1 hour ago'. Omit to follow live")
    ap.add_argument("--rejected-only", action="store_true",
                    help="only show what the validator caught")
    ap.add_argument("--no-facts", action="store_true",
                    help="hide the prompt, show verdicts only")
    ap.add_argument("--unit", default="turbulence-agent",
                    help="journald syslog identifier")
    args = ap.parse_args()

    # short-iso rather than cat: the syslog formatter omits the time, so
    # journald's own stamp is the only one available.
    cmd = ["journalctl", "-t", args.unit, "-o", "short-iso", "--no-pager"]
    cmd += ["--since", args.since] if args.since else ["-f", "-n", "0"]

    watcher = Watcher(args)
    following = not args.since

    print(f"{DIM}Watching the model's exchanges"
          f"{' (live)' if following else f' since {args.since}'}."
          f"{' Ctrl-C to stop.' if following else ''}{RESET}")
    if not args.no_facts:
        print(f"{DIM}Prompts appear only if TURBULENCE_LOG_EXPLAINER_IO=1 "
              f"is set on the service.{RESET}")

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True,
                                bufsize=1)
    except FileNotFoundError:
        print("journalctl not found. This tool reads the systemd journal.",
              file=sys.stderr)
        return 2

    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            match = LINE.search(line)
            if not match:
                continue
            stamp = STAMP.match(line)
            watcher.handle(match.group("req"), match.group("message"),
                           stamp.group("time") if stamp else "")
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        watcher.summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
