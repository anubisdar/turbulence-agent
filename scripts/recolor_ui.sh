#!/usr/bin/env bash
# install-to: scripts
#
# recolor_ui.sh - raise the contrast of the web interface and move the
# corridor accent from green to light blue.
#
# Two problems are being fixed, and the second matters more:
#
#   1. The accent was cyan-green. Light blue reads better against the dark
#      basemap and is what was asked for.
#   2. Several greys failed contrast against the panel. `--dim` was 2.8:1,
#      below any legibility floor, and it carries the eyebrows, stat labels
#      and score-bar keys. The unfilled portion of the score bar was ~1.1:1,
#      effectively invisible - so an empty coverage segment looked like
#      nothing at all rather than like an empty bar. That bar is the one
#      graphic carrying the scoring rubric, so it has to show absence.
#
# Idempotent: run it twice and the second run reports nothing to do.
# A backup is written before any edit.
#
# Usage:
#   ./scripts/recolor_ui.sh
#   ./scripts/recolor_ui.sh --check      # report only, change nothing
#   ./scripts/recolor_ui.sh --revert     # restore the most recent backup
#
set -euo pipefail

PROJ="${PROJ:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
PAGE="$PROJ/app/web/static/index.html"
BACKUP_DIR="$PROJ/.drop-backups"

say()   { printf '  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn()  { printf '  \033[33m%s\033[0m\n' "$*"; }

MODE="apply"
case "${1:-}" in
  --check)  MODE="check" ;;
  --revert) MODE="revert" ;;
  "")       ;;
  *)        echo "unknown option: $1" >&2; exit 1 ;;
esac

[[ -f "$PAGE" ]] || {
  echo "page not found: $PAGE" >&2
  echo "run ./scripts/install_static.sh first" >&2
  exit 1
}

# ---------------------------------------------------------------- revert

if [[ "$MODE" == "revert" ]]; then
  head_ "Reverting"
  latest=$(ls -1dt "$BACKUP_DIR"/*/index.html 2>/dev/null | head -1 || true)
  [[ -n "$latest" ]] || { echo "  no backup found" >&2; exit 1; }
  cp "$latest" "$PAGE"
  say "restored from ${latest#$PROJ/}"
  exit 0
fi

# ---------------------------------------------------------------- edits
#
# old -> new. Order matters only in that no new value may appear as a later
# old value, or a second substitution would undo the first.

declare -a SUBS=(
  # --- panel and structure: more separation, visible hairlines
  '  --raised:#141E29;|  --raised:#18232F;'
  '  --rule:#1F2C3A;|  --rule:#2A3A4A;'
  '  --ink:#E6EDF3;|  --ink:#F0F5F9;'

  # --- text greys: --dim was 2.8:1 against the panel and carries the
  #     eyebrows, stat labels and score-bar keys
  '  --muted:#7B8E9F;|  --muted:#A8B8C6;'
  '  --dim:#4E5F6E;|  --dim:#7E90A0;'

  # --- accent: cyan-green to light blue
  '  --live:#00E5C0;|  --live:#4CC9F0;'
  '  --winner:#5BF7D6;|  --winner:#9BE8FF;'
  '  --dead:#54697C;|  --dead:#8296A8;'
  '  --caution:#FFB454;|  --caution:#FFC061;'
  '  --alarm:#FF6B5A;|  --alarm:#FF8577;'

  # --- the signature element: the unfilled bar track must be visible, or
  #     an empty coverage segment reads as absent rather than as empty
  '.seg{background:#0A1219;|.seg{background:#26333F;'

  # --- button text was tuned to sit on green
  'color:#04120F;|color:#041521;'

  # --- map literals in the JavaScript
  "color: winner ? '#5BF7D6' : kept ? '#00E5C0' : '#54697C',|color: winner ? '#9BE8FF' : kept ? '#4CC9F0' : '#8296A8',"
  "fillColor: winner ? '#5BF7D6' : kept ? '#00E5C0' : '#54697C',|fillColor: winner ? '#9BE8FF' : kept ? '#4CC9F0' : '#8296A8',"
  "      color: c.is_winner ? '#5BF7D6' : c.kept ? '#00E5C0' : '#54697C',|      color: c.is_winner ? '#9BE8FF' : c.kept ? '#4CC9F0' : '#8296A8',"

  # --- pruned corridors were nearly gone against the basemap; they are
  #     meant to be seen, since the discarded branches are the point
  'opacity: kept ? 0.95 : 0.5,|opacity: kept ? 0.95 : 0.65,'
  'fillOpacity: isSel ? 0.24 : winner ? 0.16 : kept ? 0.09 : 0.05,|fillOpacity: isSel ? 0.26 : winner ? 0.18 : kept ? 0.11 : 0.08,'
  'weight: c.is_winner ? 2 : 1, opacity: c.kept ? 0.9 : 0.4,|weight: c.is_winner ? 2 : 1, opacity: c.kept ? 0.9 : 0.6,'
)

head_ "Checking $([[ "$MODE" == check ]] && echo "(no changes will be made)")"

pending=0
already=0
missing=()
for sub in "${SUBS[@]}"; do
  old="${sub%%|*}"
  new="${sub#*|}"
  if grep -qF -- "$old" "$PAGE"; then
    pending=$((pending + 1))
  elif grep -qF -- "$new" "$PAGE"; then
    already=$((already + 1))
  else
    missing+=("$old")
  fi
done

say "$pending change(s) to apply"
say "$already already applied"
if ((${#missing[@]})); then
  warn "${#missing[@]} pattern(s) not found - the page may have been edited:"
  for m in "${missing[@]}"; do warn "    ${m:0:70}"; done
fi

if [[ "$MODE" == "check" ]]; then
  head_ "Check only - nothing written"
  exit 0
fi

if ((pending == 0)); then
  head_ "Nothing to do"
  exit 0
fi

# ---------------------------------------------------------------- apply

stamp="$BACKUP_DIR/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$stamp"
cp "$PAGE" "$stamp/index.html"

python3 - "$PAGE" "${SUBS[@]}" <<'PYEOF'
import sys

page = sys.argv[1]
subs = [s.split("|", 1) for s in sys.argv[2:]]
text = open(page, encoding="utf-8").read()

applied = 0
for old, new in subs:
    if old in text:
        text = text.replace(old, new)
        applied += 1

open(page, "w", encoding="utf-8").write(text)
print(f"  applied {applied} change(s)")
PYEOF

# ---------------------------------------------------------------- verify

head_ "Verifying"

fail=0
for hex in '#00E5C0' '#5BF7D6' '#0A1219' '#04120F'; do
  if grep -qF -- "$hex" "$PAGE"; then
    warn "old colour still present: $hex"
    fail=1
  fi
done
for hex in '#4CC9F0' '#9BE8FF' '#26333F'; do
  grep -qF -- "$hex" "$PAGE" || { warn "new colour missing: $hex"; fail=1; }
done

open_tags=$(grep -o '<div' "$PAGE" | wc -l)
close_tags=$(grep -o '</div>' "$PAGE" | wc -l)
[[ "$open_tags" == "$close_tags" ]] || {
  warn "div tags unbalanced: $open_tags open, $close_tags close"; fail=1; }

if ((fail)); then
  warn "verification found problems. restore with:"
  warn "    ./scripts/recolor_ui.sh --revert"
  exit 1
fi

say "colours swapped, markup intact"
say "backup: ${stamp#$PROJ/}/index.html"

head_ "Done"
say "Reload with a hard refresh - Ctrl+Shift+R - or the browser will serve"
say "the cached page and it will look unchanged."
