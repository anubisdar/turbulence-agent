#!/usr/bin/env python3
# install-to: scripts
"""Email a weekly_check.sh report.

Split out of the bash script because reading a file and sending it is
friendlier here than in bash, and because it can be exercised on its own
without re-running the whole check.

Credentials come from environment variables, the same pattern AEROAPI_KEY
and ANTHROPIC_API_KEY already follow in this project (see
turbulence-agent.env.example) - never a value hardcoded here or committed
anywhere.

A send failure is reported clearly but is never allowed to look like the
check itself failed: weekly_check.sh has already written the report and
decided its own PASS/FAIL before this ever runs, and this script's exit
code only ever describes whether the email went out.

Usage:
    export WEEKLY_REPORT_SMTP_HOST=smtp.gmail.com
    export WEEKLY_REPORT_SMTP_USER=you@gmail.com
    export WEEKLY_REPORT_SMTP_PASS=...   # an app password, not the account password
    python3 scripts/send_weekly_report.py reports/weekly/2026-09-29T181712Z.md --status PASS
"""
from __future__ import annotations

import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


def send(report: Path, status: str, to: str, host: str, port: int,
         user: str, password: str, sender: str) -> None:
    body = report.read_text(encoding="utf-8")

    msg = EmailMessage()
    msg["Subject"] = f"[turbulence-agent] weekly check: {status} — {report.stem}"
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(
        f"Weekly check result: {status}\n\n"
        f"Full report attached and inlined below.\n\n"
        f"{'-' * 60}\n\n{body}"
    )
    msg.add_attachment(body.encode("utf-8"), maintype="text",
                       subtype="markdown", filename=report.name)

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.starttls(context=context)
        smtp.login(user, password)
        smtp.send_message(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path, help="path to the .md report")
    ap.add_argument("--status", choices=("PASS", "FAIL"), required=True)
    ap.add_argument("--to", default=os.environ.get(
        "WEEKLY_REPORT_TO", "matthew.darlage@gmail.com"))
    args = ap.parse_args()

    if not args.report.exists():
        sys.exit(f"no report at {args.report}")

    host = os.environ.get("WEEKLY_REPORT_SMTP_HOST")
    port = int(os.environ.get("WEEKLY_REPORT_SMTP_PORT", "587"))
    user = os.environ.get("WEEKLY_REPORT_SMTP_USER")
    password = os.environ.get("WEEKLY_REPORT_SMTP_PASS")
    sender = os.environ.get("WEEKLY_REPORT_FROM", user or "")

    missing = [name for name, val in (
        ("WEEKLY_REPORT_SMTP_HOST", host),
        ("WEEKLY_REPORT_SMTP_USER", user),
        ("WEEKLY_REPORT_SMTP_PASS", password),
    ) if not val]
    if missing:
        print(f"Not sent — {', '.join(missing)} not set. The report is "
             f"still at {args.report}; see turbulence-agent.env.example "
             f"for setup.", file=sys.stderr)
        return 1

    try:
        send(args.report, args.status, args.to, host, port, user, password,
            sender)
    except Exception as e:  # noqa: BLE001 - report exactly what broke, then exit
        print(f"Send failed ({type(e).__name__}: {e}). The report is still "
             f"at {args.report}.", file=sys.stderr)
        return 1

    print(f"Sent to {args.to} via {host}:{port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
