"""Names vulture reports that are used in ways it cannot see.

Vulture ignores scopes and matches on token names, so anything reached
dynamically, through a framework, or as a required-but-unread parameter
looks dead to it. Suppressing those here keeps the real signal readable: a
tool whose output is 90% noise gets ignored, and the one genuine finding
goes with it.

Every entry needs a reason. An unexplained line here is a place to hide a
defect.

    python3 -m vulture app/ tests/ vulture_whitelist.py --min-confidence 60
"""

# --- injected transports -----------------------------------------------
# Test doubles must match the transport signature `(path, params)` even
# though most of them ignore both. The parameters are required, not dead.
def _transport(p, q):        # noqa: ARG001 - signature, not usage
    """The shape every injected transport must have."""
    return p, q


_transport(None, None)

# --- FastAPI ------------------------------------------------------------
# Route handlers and middleware are called by the framework, never by name.
from app.web import api  # noqa: E402

api.status
api.status_page
api.health
api.fixes
api.index
api.corridor_search
api.reputation_search
api.correlate
api.security_headers
api.origin_and_destination_must_differ

# --- public interfaces --------------------------------------------------
# Exported for callers outside this codebase, or kept deliberately for the
# operator scripts rather than the service.
from app import logging_setup  # noqa: E402

logging_setup.new_request_id
logging_setup.set_request_id
