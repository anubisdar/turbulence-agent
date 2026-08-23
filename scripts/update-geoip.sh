#!/usr/bin/env bash
#
# update-geoip.sh - refresh the GeoLite2 country database.
#
# MaxMind rebuilds this weekly. A stale database misplaces addresses that
# have been reassigned, which for a geo-filter means blocking people who
# should be allowed - a failure that looks like a bug in the application
# rather than a data problem.
#
# The new file is validated before it replaces the live one. A truncated
# download or an authentication failure returns an HTML error page, and
# writing that over a working database would take the site down at the next
# reload rather than at download time.
#
# Install:
#   sudo install -m 700 update-geoip.sh /usr/local/sbin/update-geoip.sh
#   echo 'MAXMIND_LICENSE_KEY=your_key' | sudo tee /etc/maxmind.env
#   sudo chmod 600 /etc/maxmind.env
#   sudo crontab -e
#     17 4 * * 3 /usr/local/sbin/update-geoip.sh >> /var/log/geoip-update.log 2>&1
#
# Wednesday at 04:17 UTC: MaxMind publishes on Tuesdays, and an odd minute
# avoids the top-of-hour crowd hitting their servers.

set -euo pipefail

DB_DIR=/usr/share/GeoIP

#: Three databases, each answering a different question.
#:
#:   Country  the edge filter reads this one; Caddy needs it present
#:   City     used for the region only, never the city - on the free tier a
#:            city is often the registrant's address rather than the user's,
#:            and city plus a timestamp identifies a person on a quiet site
#:   ASN      whose network the request came from, which is what actually
#:            distinguishes a scraper from a person once every other country
#:            is blocked
EDITIONS=(GeoLite2-Country GeoLite2-City GeoLite2-ASN)
ENV_FILE=/etc/maxmind.env
OWNER=caddy

log() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "FAILED: $*"; exit 1; }

[[ $EUID -eq 0 ]] || die "must run as root"
[[ -f "$ENV_FILE" ]] || die "$ENV_FILE not found; it must hold MAXMIND_LICENSE_KEY"

# shellcheck disable=SC1090
. "$ENV_FILE"
[[ -n "${MAXMIND_LICENSE_KEY:-}" ]] || die "MAXMIND_LICENSE_KEY is not set"

# A pasted key picks up carriage returns, trailing spaces and occasionally a
# stray invisible character. Any of those go into the URL and come back as
# "Invalid license key", which reads as a credential problem and is not one.
MAXMIND_LICENSE_KEY=$(printf '%s' "$MAXMIND_LICENSE_KEY" | tr -d '[:space:]')
[[ "$MAXMIND_LICENSE_KEY" =~ ^[A-Za-z0-9_]+$ ]] || die \
  "the licence key contains characters a MaxMind key does not use; check $ENV_FILE for a stray paste"
log "key: ${#MAXMIND_LICENSE_KEY} characters, ending ${MAXMIND_LICENSE_KEY: -4}"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

failures=0
for EDITION in "${EDITIONS[@]}"; do
DB_PATH="$DB_DIR/$EDITION.mmdb"

log "downloading $EDITION"
# -L matters: MaxMind answers with a 302 to a signed storage URL rather than
# the file itself. Without it curl returns an empty body, which fails the
# gzip check and reads as a rejected licence key rather than an unfollowed
# redirect.
URL="https://download.maxmind.com/app/geoip_download"
if ! curl -fsSL --max-time 120 \
     -o "$WORK/db.tar.gz" \
     "$URL?edition_id=$EDITION&license_key=$MAXMIND_LICENSE_KEY&suffix=tar.gz"; then
  log "FAILED: $EDITION did not download"
  failures=$((failures + 1))
  continue
fi

# A rejected key returns 200 with a plain-text body, not an error status, so
# the file type is the only reliable signal that this is really a database.
if ! file "$WORK/db.tar.gz" | grep -q gzip; then
  log "response was not a gzip archive. First line:"
  head -c 200 "$WORK/db.tar.gz" | sed 's/^/    /'
  log "FAILED: $EDITION did not return a database"
  failures=$((failures + 1))
  continue
fi

tar -xzf "$WORK/db.tar.gz" -C "$WORK"
NEW=$(find "$WORK" -name "$EDITION.mmdb" -print -quit)
if [[ -z "$NEW" ]]; then
  log "FAILED: no $EDITION.mmdb inside the archive"
  failures=$((failures + 1))
  continue
fi

NEW_SIZE=$(stat -c%s "$NEW")
if [[ "$NEW_SIZE" -lt 500000 ]]; then
  log "FAILED: $EDITION is only $NEW_SIZE bytes, too small to be real"
  failures=$((failures + 1))
  continue
fi

# Unchanged is the common case: MaxMind publishes weekly and cron may run
# before a new build. Replacing an identical file would churn Caddy for
# nothing.
if [[ -f "$DB_PATH" ]] && cmp -s "$NEW" "$DB_PATH"; then
  log "$EDITION already current ($NEW_SIZE bytes)"
  continue
fi

OLD_SIZE=0
[[ -f "$DB_PATH" ]] && OLD_SIZE=$(stat -c%s "$DB_PATH")

# Keep the previous file. If a new database turns out to be wrong, having
# the old one on disk is the difference between a rollback and a re-download.
if [[ -f "$DB_PATH" ]]; then
  cp -p "$DB_PATH" "$DB_PATH.previous"
fi

install -o "$OWNER" -g "$OWNER" -m 644 "$NEW" "$DB_PATH"
log "$EDITION installed: $OLD_SIZE -> $NEW_SIZE bytes"
rm -rf "${WORK:?}"/*
done

if [[ "$failures" -gt 0 ]]; then
  log "$failures edition(s) failed. The country database is the one Caddy"
  log "needs; the others only affect the status page."
fi

# Caddy reads the database at config load, so it keeps using the old one in
# memory until it is told otherwise.
if systemctl is-active --quiet caddy; then
  if caddy validate --config /etc/caddy/Caddyfile >/dev/null 2>&1; then
    systemctl reload caddy
    log "caddy reloaded"
  else
    log "WARNING: the Caddyfile does not validate, so caddy was not reloaded."
    log "The new database is in place and will be picked up at the next restart."
  fi
else
  log "caddy is not running; the database will be read when it starts"
fi
