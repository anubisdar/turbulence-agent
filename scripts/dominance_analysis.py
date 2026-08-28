#!/usr/bin/env python3
# install-to: scripts
"""
Is the dominance threshold doing any work?

0.80 was calibrated from a single observed search. That is one data point
carrying a rule that discards corridors, and the honest thing is to find out
whether it matters before defending the number.

The logs already answer it. Every dominance prune records the overlap that
triggered it, so the distribution of those overlaps says which of three
worlds you are in:

  nothing near the line   every overlap is 0.95 or above, so the threshold
                          could be 0.90 and nothing would change. The number
                          is arbitrary and harmless.

  clustered on the line   overlaps sit just above 0.80, so the one search
                          that set it is deciding what gets discarded today.
                          That is worth re-calibrating.

  never fires at all      dominance has pruned nothing, and the rule is dead
                          code with a threshold nobody needs to defend.

Runs against the journal. Costs nothing.

Usage:
    python3 scripts/dominance_analysis.py --days 30
    python3 scripts/dominance_analysis.py --threshold 0.80
"""

import argparse
import re
import subprocess
import sys
from collections import Counter

#: Matches the decision name however it is spelled. The first version of
#: this script looked for "dominan", which does not appear in
#: "prune_dominated" - the substring ends one letter short. It reported
#: zero while the rule had fired 303 times, and only the decision-type
#: listing above showed why.
_IS_DOMINANCE = "domina"

#: The configured rule. Passed in rather than imported so the script can be
#: pointed at a log from a build that used a different one.
DEFAULT_THRESHOLD = 0.80

#: Any decision line, so the script can report what it actually found
#: rather than silently matching nothing.
_DECISION = re.compile(
    r"critic decision .*?decision=(?P<decision>\S+).*?score=(?P<score>[\d.]+)")

#: The reason string, isolated first. Searching the whole line found
#: score=0.61 and reported it as the overlap, because the score appears
#: before the reason and the pattern is not anchored.
_REASON = re.compile(r'reason="(?P<reason>[^"]*)"')

#: The overlap inside that reason. Several phrasings have been used, so
#: this looks for a fraction near the word rather than one exact sentence.
_OVERLAP = re.compile(
    r"overlap[^0-9]{0,24}(?P<frac>[01]?\.\d+|\d{1,3})\s*%?", re.I)


def journal(days: int) -> list[str]:
    try:
        return subprocess.run(
            ["journalctl", "-t", "turbulence-agent",
             "--since", f"{days} days ago", "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=120).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as e:
        print(f"could not read the journal: {type(e).__name__}",
              file=sys.stderr)
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--near", type=float, default=0.05,
                    help="how close to the threshold counts as near it")
    args = ap.parse_args()

    lines = journal(args.days)
    if not lines:
        print("Nothing in the journal for that window.")
        return 1

    kinds = Counter()
    overlaps = []
    unparsed = 0

    for line in lines:
        found = _DECISION.search(line)
        if not found:
            continue
        decision = found.group("decision")
        kinds[decision] += 1
        if _IS_DOMINANCE not in decision.lower():
            continue

        reason = _REASON.search(line)
        got = _OVERLAP.search(reason.group("reason")) if reason else None
        if not got:
            unparsed += 1
            continue
        raw = float(got.group("frac"))
        overlaps.append(raw / 100.0 if raw > 1.5 else raw)

    print("  decisions seen in this window:")
    for kind, n in kinds.most_common():
        print(f"    {kind:<20} {n}")
    print()

    dominance = sum(n for k, n in kinds.items()
                    if _IS_DOMINANCE in k.lower())
    if not dominance:
        print("  Dominance never fired. It has discarded nothing in this "
              "window, so the\n  threshold is not deciding anything and its "
              "calibration does not matter\n  yet. Worth saying plainly "
              "rather than defending a number that has never\n  been used.")
        return 0

    if not overlaps:
        print(f"  Dominance fired {dominance} times, but no overlap value "
              f"could be read from\n  those lines. The rule is working and "
              f"the logging is not: the reason\n  string does not carry the "
              f"number that triggered it. That is worth fixing\n  before the "
              f"threshold can be argued about at all.")
        if unparsed:
            print(f"    ({unparsed} dominance lines had no readable "
                  f"fraction)")
        return 0

    overlaps.sort()
    near = [o for o in overlaps
            if o < args.threshold + args.near]
    print(f"  dominance fired {len(overlaps)} time(s), overlap at the moment "
          f"it did:")
    print(f"    smallest : {overlaps[0]:.4f}")
    print(f"    median   : {overlaps[len(overlaps) // 2]:.4f}")
    print(f"    largest  : {overlaps[-1]:.4f}")
    print()

    if near:
        print(f"  {len(near)} of {len(overlaps)} were within {args.near} of "
              f"the {args.threshold} line.")
        print(f"  The threshold is load-bearing: move it and those corridors "
              f"change fate.\n  One observed search is thin evidence for a "
              f"number doing that much work.")
    else:
        margin = overlaps[0] - args.threshold
        print(f"  The closest overlap cleared the line by {margin:.4f}. In "
              f"this window the\n  threshold could have been anywhere below "
              f"{overlaps[0]:.2f} and pruned exactly the\n  same corridors - "
              f"so its precise value has not mattered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
