#!/usr/bin/env bash
# install-to: deploy
#
# bootstrap.sh - configure a fresh Ubuntu 24.04 EC2 instance to serve the
# turbulence agent. Run once, on the instance, as root or with sudo.
#
# What it sets up:
#   - Python dependencies including PyTorch, which is the bulk of the ~2 GB
#   - the embedding model, pre-downloaded so the first real request is not
#     also a HuggingFace round trip
#   - a systemd unit running uvicorn on 127.0.0.1 with ONE worker
#   - Caddy in front, terminating TLS and holding basic auth
#
# ONE WORKER IS DELIBERATE. Every corridor search writes to the route fix
# cache. Several workers against one SQLite file produce intermittent
# "database is locked" errors under concurrent use. A demo instance has no
# need for more.
#
# HTTPS WITHOUT A DOMAIN. sslip.io resolves 3-14-159-26.sslip.io to
# 3.14.159.26, and Let's Encrypt will issue for it, so a bare EC2 instance
# gets a real certificate with nothing to buy. The alternative - basic auth
# over plain HTTP - would send the password in clear text.
#
# Usage:
#   sudo ./bootstrap.sh --user demo --password 'something-long'
#   sudo ./bootstrap.sh --user demo --password 'x' --domain agent.example.com
#
set -euo pipefail

APP_USER="turbulence"
APP_DIR="/opt/turbulence-agent"
AUTH_USER=""
AUTH_PASS=""
DOMAIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)     AUTH_USER="$2"; shift 2 ;;
    --password) AUTH_PASS="$2"; shift 2 ;;
    --domain)   DOMAIN="$2";    shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
[[ -n "$AUTH_USER" && -n "$AUTH_PASS" ]] || {
  echo "--user and --password are required. Without them the endpoint is" >&2
  echo "open, and every search spends against a metered API allowance." >&2
  exit 1
}

say()   { printf '  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- hostname

if [[ -z "$DOMAIN" ]]; then
  PUBLIC_IP=$(curl -fsS --max-time 5 \
    -H "X-aws-ec2-metadata-token: $(curl -fsS -X PUT \
      'http://169.254.169.254/latest/api/token' \
      -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null)" \
    http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)
  [[ -n "$PUBLIC_IP" ]] || { echo "could not read the public IP" >&2; exit 1; }
  DOMAIN="${PUBLIC_IP//./-}.sslip.io"
  say "no domain given, using $DOMAIN"
fi

# ---------------------------------------------------------------- packages

head_ "Installing packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip sqlite3 rsync curl \
                       debian-keyring debian-archive-keyring apt-transport-https

if ! command -v caddy >/dev/null; then
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -fsSL https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy
fi

# Rate limiting is a plugin, not a core directive, and a Caddy without it
# refuses to start rather than ignoring the config. That is the right
# failure mode and a poor surprise, so the module goes in before any
# Caddyfile references it.
if ! caddy list-modules 2>/dev/null | grep -q "http.handlers.rate_limit"; then
  say "adding the rate limit module to Caddy"
  caddy add-package github.com/mholt/caddy-ratelimit || {
    warn "could not add the rate limit module. Caddy will not start with a"
    warn "Caddyfile that uses it. Either add it by hand:"
    warn "    caddy add-package github.com/mholt/caddy-ratelimit"
    warn "or comment out the rate_limit block, and do not remove basic auth"
    warn "until one of those is done."
  }
fi
say "packages ready"

# ---------------------------------------------------------------- app user

head_ "Application account"
id "$APP_USER" &>/dev/null || useradd --system --create-home \
  --home-dir "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
mkdir -p "$APP_DIR"/{app,scripts,data}
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
say "$APP_USER owns $APP_DIR"

# ---------------------------------------------------------------- python

head_ "Python environment (this is the slow part - PyTorch is ~2 GB)"
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q \
  fastapi "uvicorn[standard]" pydantic pydantic-settings \
  langgraph sqlite-vec sentence-transformers \
  pyproj shapely \
  anthropic maxminddb
say "dependencies installed"

# ---- edge event ingest -----------------------------------------------
#
# Firewall detections, challenge refusals and edge blocks happen in Caddy or
# before a search starts, so the application never sees them. They were
# reconstructed from logs while rendering the status page, which meant three
# panels each carrying a different window from the rest of the site. This
# moves them into the same database on the same retention.
head_ "Edge event ingest"
cat > /etc/systemd/system/ingest-edge-events.service <<UNIT
[Unit]
Description=Ingest edge events into the turbulence agent database
After=network.target

[Service]
Type=oneshot
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/scripts/ingest_edge_events.py
UNIT

cat > /etc/systemd/system/ingest-edge-events.timer <<UNIT
[Unit]
Description=Ingest edge events every five minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
# The ingest re-reads thirty minutes each run and deduplicates, so a missed
# window costs nothing and a persistent one is caught up automatically.
Persistent=true

[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now ingest-edge-events.timer
say "ingest timer enabled (every 5 minutes)"

# The ingest reads the journal, which needs group membership the app user
# does not have by default.
usermod -a -G systemd-journal "$APP_USER" || warn "could not add $APP_USER to systemd-journal"


# Importing every module that reaches an external source, because a missing
# dependency here does not crash the service: the source is caught, reported
# as unavailable, and the search continues with an honest absence. That is
# correct behaviour and it is indistinguishable from quiet weather, so the
# defect can sit in production for days. Checked once, at install, loudly.
head_ "Import check"
missing=0
for mod in app.sources.awc app.sources.gairmet app.sources.aeroapi \
           app.reasoning.evidence app.reasoning.explainer app.runs; do
  if sudo -u "$APP_USER" bash -c "cd '$APP_DIR' && '$APP_DIR/.venv/bin/python' \
       -c 'import $mod' 2>/dev/null"; then
    say "ok       $mod"
  else
    reason=$(sudo -u "$APP_USER" bash -c "cd '$APP_DIR' && \
      '$APP_DIR/.venv/bin/python' -c 'import $mod' 2>&1 | tail -1")
    err "FAILED   $mod"
    err "         $reason"
    missing=$((missing + 1))
  fi
done
if [[ "$missing" -gt 0 ]]; then
  warn "$missing module(s) will not import. Any external source they reach"
  warn "will report as unavailable, which reads exactly like quiet weather."
fi

head_ "Pre-downloading the embedding model"
# Run from a directory the app user owns. sentence_transformers first
# checks for a local directory matching the model name, relative to the
# working directory - so running this from wherever bootstrap was invoked
# makes it stat a path the app user may not be able to read, and the
# resulting PermissionError looks like a download failure rather than a
# working-directory problem.
sudo -u "$APP_USER" HF_HOME="$APP_DIR/.cache/huggingface" \
  bash -c "cd '$APP_DIR' && '$APP_DIR/.venv/bin/python' -c \"
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
print('  model cached')
\"" || {
  warn "the model could not be pre-downloaded. This is not fatal: it will"
  warn "be fetched on the first search instead, which adds about 30 seconds"
  warn "to that one request. Continuing."
}

# ---------------------------------------------------------------- secrets

head_ "Environment file"
ENV_FILE="/etc/turbulence-agent.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<EOF
# Both keys are metered. Treat them as credentials: never echo them, and
# rotate rather than investigate if one is ever exposed.
AEROAPI_KEY=
ANTHROPIC_API_KEY=

# Tells the app it is serving anonymous callers. Lowers the ceilings on the
# expensive request parameters and hides the interactive docs.
TURBULENCE_PUBLIC=1

# Origin, destination and date together are an itinerary. Off by default,
# and it should stay off on a public deployment.
TURBULENCE_LOG_TRIP_CONTENT=0

# Cloudflare Turnstile. Both halves are needed before the challenge appears;
# with neither set the site behaves exactly as it did before. The site key is
# public and rendered into the page; the secret key never leaves the server.
# Get both free at dash.cloudflare.com -> Turnstile.
TURNSTILE_SITE_KEY=
TURNSTILE_SECRET_KEY=

TURBULENCE_DB=$APP_DIR/data/retrieval.db
HF_HOME=$APP_DIR/.cache/huggingface
EOF
  chmod 600 "$ENV_FILE"
  chown root:root "$ENV_FILE"
  say "created $ENV_FILE - put the AeroAPI key in it before starting"
else
  say "$ENV_FILE already exists, left alone"
fi

# ---------------------------------------------------------------- service

head_ "systemd unit"
cat > /etc/systemd/system/turbulence-agent.service <<EOF
[Unit]
Description=Turbulence-aware flight ranking agent
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
# One worker. The route fix cache writes on every corridor search, and
# several workers against one SQLite file will collide.
ExecStart=$APP_DIR/.venv/bin/uvicorn app.web.api:app \\
    --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$APP_DIR/data $APP_DIR/.cache

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable -q turbulence-agent
say "turbulence-agent.service installed"

# ---------------------------------------------------------------- caddy

head_ "Caddy"
# Before the Caddyfile, not after: Caddy opens its log file at startup and
# refuses to run if it cannot, so a directory created later is created too
# late. The service user comes from the package rather than being assumed.
CADDY_USER=$(awk -F= '/^User=/{print $2}' /usr/lib/systemd/system/caddy.service 2>/dev/null)
CADDY_USER=${CADDY_USER:-caddy}
mkdir -p /var/log/caddy
# Recursive on purpose. A failed first start leaves a root-owned, mode-600
# access.log behind, and chowning only the directory leaves that file
# unwritable - which reads as a directory permission problem and is not one.
chown -R "$CADDY_USER:$CADDY_USER" /var/log/caddy
chmod 755 /var/log/caddy
say "log directory owned by $CADDY_USER"

HASH=$(caddy hash-password --plaintext "$AUTH_PASS")
cat > /etc/caddy/Caddyfile <<EOF
{
    # Rate limiting needs this module. A build without it fails loudly at
    # startup rather than serving unlimited traffic, which is the safer
    # direction to fail.
    order rate_limit before basic_auth
}

$DOMAIN {
    encode gzip

    # THE HIGHEST-VALUE CONTROL HERE. Every search spends against a metered
    # AeroAPI allowance, and a search with an explanation also spends against
    # an Anthropic key that has no fixed ceiling. A scraper finding this
    # endpoint would exhaust both without meaning to.
    #
    # Ten a minute is generous for a person reading each result and useless
    # to anything automated.
    rate_limit {
        zone searches {
            key {remote_host}
            events 10
            window 1m
        }
    }

    # A search takes 12 to 25 seconds against live APIs, and the service runs
    # a single worker. Without a ceiling a stalled request holds a connection
    # and a worker together.
    reverse_proxy 127.0.0.1:8000 {
        transport http {
            response_header_timeout 90s
        }
    }

    # Without this the endpoint is open, and every corridor search spends
    # against a metered API allowance. Remove it deliberately, not by
    # forgetting it.
    basic_auth {
        $AUTH_USER $HASH
    }

    # JSON rather than console. The console format is easier to read by eye
    # and drops the request headers, so a blocked request records that it
    # was blocked and nothing about who sent it. The user agent and the
    # source address are the two things worth knowing about traffic that
    # never reaches the application, and JSON carries both.
    log {
        output file /var/log/caddy/access.log {
            # Caddy creates the file mode 600 by default, which the
            # application cannot read - and it reports that as a quiet edge
            # rather than as a permission problem unless told otherwise.
            mode 644
            roll_size 20mb
            roll_keep 5
        }
        format json
    }
}
EOF
systemctl reload caddy 2>/dev/null || systemctl restart caddy
say "serving $DOMAIN with basic auth"

# ---------------------------------------------------------------- done

head_ "Next steps"
cat <<EOF
  1. Put both API keys in $ENV_FILE
  2. Push the code and database from your workstation:
         ./scripts/deploy.sh <this-host>
  3. Start it:
         sudo systemctl start turbulence-agent
  4. Open https://$DOMAIN  (user: $AUTH_USER)

  Security group must allow 80 and 443 inbound. Port 80 is needed for the
  Let's Encrypt challenge even though everything redirects to 443.

  BEFORE REMOVING BASIC AUTH, set a hard monthly spend cap in the Anthropic
  console. It is the only control that still holds if the rate limiting is
  misconfigured, and an anonymous endpoint calling a paid model without one
  has no ceiling at all.

  Logs:    journalctl -u turbulence-agent -f
  Restart: sudo systemctl restart turbulence-agent
EOF
