#!/usr/bin/env bash
# install-to: scripts
#
# publish_changes.sh - commit this change set and push it, with the
# checks that make that safe on a public repository.
#
# WHY NOT `git add -A`. This repository is public and the .gitignore file
# opens by saying every pattern in it has held a live credential at some
# point. A blanket add is how a Caddyfile with basic-auth hashes, a .pem,
# or a database with a run record ends up on GitHub, and once pushed it is
# public whether or not the next commit removes it. So this script stages
# a named list and nothing else. A file not on the list is not committed,
# however tempting that is when the tree is dirty.
#
# WHAT IT REFUSES ON. A detached HEAD, the wrong remote, a branch behind
# origin, a staged file that .gitignore should have caught, an obvious
# credential in the staged diff, or a failing test suite. Each one stops
# before anything is written; nothing here rewrites history and nothing
# uses --force.
#
# WHAT IT WILL NOT DO. It will not commit /root/who-came.sh,
# /root/report-mail.sh or /root/build_report.py. Those are not in the
# repository, and adding them would put an operator address and a mail
# path into a public tree. If you want them tracked, that is a decision
# to make deliberately, with the addresses parameterised first.
#
# Usage:
#   ./scripts/publish_changes.sh              # check and show, commit nothing
#   ./scripts/publish_changes.sh --commit     # run the gates, then commit
#   ./scripts/publish_changes.sh --commit --push
#
#   --no-tests        skip the suite (it is the gate; say why in the log)
#   --remote NAME     default origin
#
# The GROUP_n_* variables below are read through indirect expansion. The
# linter cannot follow that, so it reports every one of them as unused.
# shellcheck disable=SC2034
set -uo pipefail

COMMIT=0
PUSH=0
RUN_TESTS=1
REMOTE=origin
EXPECT_REPO="turbulence-agent"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)   COMMIT=1 ;;
    --push)     PUSH=1 ;;
    --no-tests) RUN_TESTS=0 ;;
    --remote)   REMOTE="$2"; shift ;;
    -h|--help)  sed -n '/^# Usage:/,/^$/p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
say()  { printf '  %s\n' "$*"; }
dim()  { printf '  \033[2m%s\033[0m\n' "$*"; }
warn() { printf '  \033[33mwarn  %s\033[0m\n' "$*"; }
die()  { printf '  \033[31mSTOP  %s\033[0m\n' "$*" >&2; exit 1; }

# Each group becomes one commit, so the history says what changed and
# why rather than "updates". Format: subject, then the files.
#
GROUP_1_SUBJECT="Mark the operator's own probes at ingest"
GROUP_1_FILES=(app/edge_events.py
               scripts/ingest_edge_events.py
               tests/test_edge_probes.py)
GROUP_1_BODY="check_edge.sh sets a distinctive user agent on every probe so
they can be excluded from the status page. Nothing read it. Four vectors
an hour, each tripping several rule families, put the operator's own
traffic at 625 of 673 firewall detections - the health check watching
itself, presented as attack traffic.

The Coraza log line carries no user agent, so a firewall detection cannot
be recognised from its own record. probe_addresses reads the access log,
which does carry one, and marks detections from the same address within
the window. Edge blocks have the agent on the record and use it directly,
with the address set catching the scanner vector, which sends a scanner's
agent by design.

Marked, not dropped. The rows stay in the table; summary() sets them
aside and reports the count under self_check so the panel can account for
the difference rather than appear to lose traffic between deploys.

The dedup key is unchanged, so re-reading a window ingested before the
flag existed stays idempotent."

GROUP_2_SUBJECT="Check the interpreter the report job actually uses"
GROUP_2_FILES=(scripts/check_edge.sh)
GROUP_2_BODY="The resolution check asserted that resolve_origin('8.8.8.8')
returns US, and passed throughout a period when every address in the
six-hourly digest printed '?'. It runs under the venv; report-mail.sh ran
its builder under system python as root from /root, where neither geoip2
nor the app package is importable. resolve_origin returns None on a
failed read rather than raising, so the digest rendered the failure as a
finding.

A check that exercises its own environment can only vouch for its own
environment. It now asks report-mail.sh --check about its own
interpreter, guarded by a grep for the flag: a version predating it has
no default arm in its case statement, so an unrecognised argument would
fall through and send mail on every run of this script.

Also: the failure message said 'three known addresses' while the block
asserts on two, and the count is now read from the last line of the
remote output so a warning on stdout cannot be mistaken for a failure."

GROUP_3_SUBJECT="Red team the explainer's output validator"
GROUP_3_FILES=(tests/test_redteam_validator.py
               scripts/redteam_explainer.py)
GROUP_3_BODY="Every rejection anyone had read was a false positive, so the
guardrail had no demonstrated true positives. This adds a violating
paragraph per rule and asserts the rule fires, plus ten defects found by
running against the real validate(): five correct paragraphs it discards
and five violating ones it accepts.

The false positives share one root cause, pinned by a white-box test on
_clauses: an enumeration tail of three or more words becomes its own
clause and the negation governing it is stranded in the previous one.
'or severe.' merges and passes; 'or severe on this route.' does not.

Defects are xfail(strict=True), so the suite stays green and a fix turns
the test into a loud failure rather than being quietly undocumented.

redteam_explainer.py runs the same facts through the shipped prompt and a
stripped one to measure what the validator is worth. It calls the client
directly because explain() hardcodes SYSTEM_PROMPT; no application code
is modified or monkeypatched."

GROUP_4_SUBJECT="Add the backfill and publish helpers"
GROUP_4_FILES=(scripts/backfill_probes.sh
               scripts/publish_changes.sh)
GROUP_4_BODY="Rows ingested before the probe flag existed cannot be marked
the same way: the address was resolved to a country and discarded, so
only the payload, country and network remain, and matching on those is
inference rather than measurement.

Kept but not run. The 30-day retention retires the miscounted rows on its
own, which needs no judgement call about which of them were probes.
Committed so the option and its reasoning are on the record.

publish_changes.sh stages a named list rather than everything, because
the .gitignore in this repository opens by saying every pattern in it has
held a live credential. A blanket add on a public repository is how one
of them gets published."

# GROUPS is a bash built-in holding the current user's supplementary
# group ids. Assigning to it is silently ignored, and the loop below then
# reads the real one - "0" for root - producing an unbound-variable error
# on GROUP_0_SUBJECT with nothing to say it came from a name collision.
COMMIT_GROUPS=(1 2 3 4)

# ------------------------------------------------------------- preflight
bold "Preflight"

command -v git >/dev/null || die "git is not installed"
git rev-parse --git-dir >/dev/null 2>&1 || die "not inside a git repository"
cd "$(git rev-parse --show-toplevel)" || die "cannot reach the repository root"
say "repository $(pwd)"

URL=$(git remote get-url "$REMOTE" 2>/dev/null) \
  || die "no remote called $REMOTE"
case "$URL" in
  *"$EXPECT_REPO"*) say "remote $REMOTE -> $URL" ;;
  *) die "remote $REMOTE is $URL, which is not $EXPECT_REPO" ;;
esac

BRANCH=$(git rev-parse --abbrev-ref HEAD)
[[ "$BRANCH" == "HEAD" ]] \
  && die "detached HEAD - check out a branch before committing"
say "branch $BRANCH"

if [[ -n "$(git diff --cached --name-only)" ]]; then
  git diff --cached --name-only | sed 's/^/    /'
  die "something is already staged; commit or reset it first"
fi

# Behind origin means a push would be rejected, or would need a merge
# nobody asked for. Better to stop and let that be a decision.
git fetch "$REMOTE" --quiet 2>/dev/null || warn "could not fetch $REMOTE"
if git rev-parse "$REMOTE/$BRANCH" >/dev/null 2>&1; then
  BEHIND=$(git rev-list --count "HEAD..$REMOTE/$BRANCH" 2>/dev/null || echo 0)
  AHEAD=$(git rev-list --count "$REMOTE/$BRANCH..HEAD" 2>/dev/null || echo 0)
  say "$AHEAD ahead, $BEHIND behind $REMOTE/$BRANCH"
  [[ "${BEHIND:-0}" -gt 0 ]] \
    && die "branch is behind $REMOTE/$BRANCH - pull and re-run"
else
  warn "$REMOTE/$BRANCH does not exist yet; a push would create it"
fi

# ---------------------------------------------------------- what is here
bold "Files in this change set"

declare -a PRESENT=()
for g in "${COMMIT_GROUPS[@]}"; do
  subject_var="GROUP_${g}_SUBJECT"; files_var="GROUP_${g}_FILES[@]"
  printf '  %s\n' "${!subject_var}"
  found=0
  for f in "${!files_var}"; do
    if [[ ! -e "$f" ]]; then
      dim "    -  $f (not present, skipped)"
    elif git check-ignore -q "$f"; then
      die "$f is covered by .gitignore - it must not be committed"
    elif [[ -z "$(git status --porcelain -- "$f")" ]]; then
      dim "    =  $f (unchanged)"
    else
      printf '    +  %s\n' "$f"
      PRESENT+=("$f"); found=$((found+1))
    fi
  done
  [[ "$found" -eq 0 ]] && dim "    nothing to commit in this group"
done

if [[ "${#PRESENT[@]}" -eq 0 ]]; then
  bold "Nothing to do"
  dim "no file in the change set differs from the last commit"
  exit 0
fi

# Anything dirty that is not on the list. Not fatal - a working tree has
# scratch files in it - but it is named, so nothing is committed by
# accident and nothing is left behind by surprise.
bold "Dirty but not in this change set"
OTHERS=$(git status --porcelain | awk '{print $2}' \
         | grep -vxF "$(printf '%s\n' "${PRESENT[@]}")" || true)
if [[ -n "$OTHERS" ]]; then
  sed 's/^/    /' <<<"$OTHERS"
  dim "these are left alone"
else
  dim "nothing else is modified"
fi

# ----------------------------------------------------------- secret scan
bold "Scanning the content to be committed"

# Blocking patterns. Deliberately few and high signal: a scanner that
# cries wolf gets bypassed, and then it catches nothing at all.
#
# Each literal is broken with a one-character class - sk-an[t]- rather
# than sk-ant- - so the patterns do not match their own definitions.
# Without that this script refuses to commit itself, which looks like a
# real finding and is not.
BLOCK='BEGIN [A-Z ]*PRIVATE KEY|sk-an[t]-[A-Za-z0-9]'
BLOCK="$BLOCK"'|aws_secret_acce[s]s_key|xox[baprs]-[0-9A-Za-z]'
BLOCK="$BLOCK"'|BEGIN CERTIFI[C]ATE'

# Worth a look, not worth refusing over. check_edge.sh has carried the
# instance address since it was written.
NOTICE='[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|@gmail\.com'
NOTICE="$NOTICE"'|passwo[r]d|secre[t]='

HITS=0
for f in "${PRESENT[@]}"; do
  if grep -qEI "$BLOCK" "$f" 2>/dev/null; then
    grep -nEI "$BLOCK" "$f" 2>/dev/null | head -3 | sed "s|^|    $f:|"
    HITS=$((HITS+1))
  fi
done
if [[ "$HITS" -gt 0 ]]; then
  die "a credential pattern is in $HITS file(s) - do not push this"
fi
say "no credential patterns"

FLAGGED=0
for f in "${PRESENT[@]}"; do
  if grep -qEI "$NOTICE" "$f" 2>/dev/null; then
    grep -nEI "$NOTICE" "$f" 2>/dev/null | head -2 | cut -c1-72 \
      | sed "s|^|    $f:|"
    FLAGGED=$((FLAGGED+1))
  fi
done
if [[ "$FLAGGED" -gt 0 ]]; then
  dim "addresses and mail patterns above are shown, not blocked - this"
  dim "repository already carries the instance address in check_edge.sh"
else
  say "no addresses or mail patterns"
fi

# ------------------------------------------------------------- the suite
bold "Test suite"
if [[ "$RUN_TESTS" -eq 0 ]]; then
  warn "skipped with --no-tests; the gate is off for this run"
elif [[ -d tests ]]; then
  PYTEST=(python3 -m pytest)
  [[ -x .venv/bin/python ]] && PYTEST=(.venv/bin/python -m pytest)
  OUT=$("${PYTEST[@]}" -q 2>&1)
  RC=$?
  tail -3 <<<"$OUT" | sed 's/^/    /'
  [[ "$RC" -eq 0 ]] || die "the suite did not pass - nothing was committed"
  say "suite passed"
else
  warn "no tests directory found"
fi

# ------------------------------------------------------------- the diff
bold "What would be committed"
git diff --stat -- "${PRESENT[@]}" | sed 's/^/    /'
git status --porcelain -- "${PRESENT[@]}" | grep '^??' | sed 's/^/    new  /'

if [[ "$COMMIT" -ne 1 ]]; then
  bold "Nothing was committed"
  dim "re-run with --commit, and --push to publish it"
  exit 0
fi

printf '\n  commit %s file(s) to %s? [y/N] ' "${#PRESENT[@]}" "$BRANCH"
read -r reply
[[ "$reply" == "y" || "$reply" == "Y" ]] || die "not confirmed"

# ------------------------------------------------------------- commit
bold "Committing"
MADE=0
for g in "${COMMIT_GROUPS[@]}"; do
  subject_var="GROUP_${g}_SUBJECT"
  body_var="GROUP_${g}_BODY"
  files_var="GROUP_${g}_FILES[@]"

  declare -a stage=()
  for f in "${!files_var}"; do
    [[ -e "$f" ]] || continue
    [[ -n "$(git status --porcelain -- "$f")" ]] && stage+=("$f")
  done
  [[ "${#stage[@]}" -eq 0 ]] && continue

  git add -- "${stage[@]}" || die "could not stage ${stage[*]}"
  # Belt and braces: confirm git staged exactly what was asked for and
  # nothing was swept in by a pathspec matching a directory.
  STAGED=$(git diff --cached --name-only | sort)
  WANTED=$(printf '%s\n' "${stage[@]}" | sort)
  if [[ "$STAGED" != "$WANTED" ]]; then
    git reset --quiet
    die "staging added more than requested; nothing committed"
  fi

  git commit --quiet -m "${!subject_var}" -m "${!body_var}" \
    || die "commit failed for group $g"
  say "$(git log -1 --format='%h  %s')"
  for f in "${stage[@]}"; do dim "      $f"; done
  MADE=$((MADE+1))
done

say "$MADE commit(s) made"

# --------------------------------------------------------------- push
if [[ "$PUSH" -ne 1 ]]; then
  bold "Not pushed"
  dim "review with: git log --stat -${MADE}"
  dim "then:        git push $REMOTE $BRANCH"
  dim "or undo:     git reset --soft HEAD~${MADE}"
  exit 0
fi

bold "Pushing"
printf '  push %s commit(s) to %s/%s? [y/N] ' "$MADE" "$REMOTE" "$BRANCH"
read -r reply
if [[ "$reply" != "y" && "$reply" != "Y" ]]; then
  dim "left committed locally; push when ready"
  exit 0
fi

if git push "$REMOTE" "$BRANCH"; then
  say "pushed"
  dim "this repository is public; the commits are now readable by anyone"
else
  die "push failed - the commits are still here, nothing was lost"
fi
