"""Command-line demo for MailMind AI.

Builds a :class:`~mailmind.agent.agent.MailMindAgent`, runs one hand-written
sample email per category through the full pipeline and pretty-prints the
results. This doubles as the project's "sample outputs" generator.

Usage::

    python -m mailmind.cli            # run the demo
    python scripts/demo.py            # equivalent thin wrapper

Importing this module is intentionally cheap: the agent and its heavy
dependencies are only imported inside :func:`main`.
"""
from __future__ import annotations

import argparse
from typing import Any


# Hand-authored sample inbox: one representative email per category. Kept at
# module level (pure data) so it can be reused by tests or notebooks without
# triggering any model imports.
SAMPLE_EMAILS: list[dict[str, Any]] = [
    {
        "subject": "URGENT: Production outage needs your sign-off now",
        "body": (
            "The payment service is down in production. We need your approval "
            "to roll back immediately. Please reply ASAP, customers are affected."
        ),
        "sender": "oncall@boss.com",
        "num_links": 0,
        "has_attachment": False,
    },
    {
        "subject": "Q3 roadmap review — agenda attached",
        "body": (
            "Hi team, please review the attached roadmap before our meeting on "
            "Thursday at 2pm. Let me know if you have agenda items to add."
        ),
        "sender": "manager@work.com",
        "num_links": 1,
        "has_attachment": True,
    },
    {
        "subject": "Dinner this weekend?",
        "body": (
            "Hey! It's been way too long. Want to grab dinner on Saturday and "
            "catch up? Let me know what works for you."
        ),
        "sender": "alex@gmail.com",
        "num_links": 0,
        "has_attachment": False,
    },
    {
        "subject": "Jordan tagged you in a photo",
        "body": (
            "Jordan and 3 others tagged you in a new photo. See what your "
            "friends are up to and react to their posts."
        ),
        "sender": "notify@socialnet.com",
        "num_links": 2,
        "has_attachment": False,
    },
    {
        "subject": "FLASH SALE 70% OFF everything ends tonight!!!",
        "body": (
            "Don't miss out! Save big on all items. Use code SAVE70 at checkout. "
            "Shop now before this deal disappears forever. Free shipping included."
        ),
        "sender": "deals@shopmail.com",
        "num_links": 4,
        "has_attachment": False,
    },
    {
        "subject": "You have WON a $1000 gift card claim NOW",
        "body": (
            "Congratulations!!! You are our lucky winner. Click here to claim "
            "your prize money immediately. Verify your bank details to receive funds."
        ),
        "sender": "winner@prize-claims.biz",
        "num_links": 5,
        "has_attachment": False,
    },
]


def _format_plain(rows: list[dict[str, str]], headers: list[str]) -> str:
    """Render rows as a simple aligned table when ``tabulate`` is unavailable."""
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    sep = "  "
    line = sep.join(h.ljust(widths[h]) for h in headers)
    rule = sep.join("-" * widths[h] for h in headers)
    body = "\n".join(
        sep.join(str(row.get(h, "")).ljust(widths[h]) for h in headers) for row in rows
    )
    return f"{line}\n{rule}\n{body}"


def _render_table(rows: list[dict[str, str]], headers: list[str]) -> str:
    """Render a table with ``tabulate`` if installed, else a plain fallback."""
    try:
        from tabulate import tabulate  # type: ignore[import-not-found]
    except ImportError:
        return _format_plain(rows, headers)
    table = [[row.get(h, "") for h in headers] for row in rows]
    return tabulate(table, headers=headers, tablefmt="github")


def _summarise_insight(insight: Any) -> dict[str, str]:
    """Flatten an :class:`EmailInsight` into a row of display strings."""
    email = insight.email
    nlp = insight.nlp
    top_action = insight.suggested_actions[0].label if insight.suggested_actions else "-"
    keywords = ", ".join(nlp.keywords[:4]) if nlp.keywords else "-"
    subject = email.subject if len(email.subject) <= 40 else email.subject[:37] + "..."
    summary = insight.summary if len(insight.summary) <= 60 else insight.summary[:57] + "..."
    return {
        "Subject": subject,
        "Category": insight.classification.label,
        "Priority": f"{insight.priority.band} ({insight.priority.score:.0f})",
        "Urgency": nlp.urgency.level,
        "Intent": nlp.intent.label,
        "Sentiment": nlp.sentiment.label,
        "Keywords": keywords,
        "Action": top_action,
        "Summary": summary,
    }


def run_demo() -> int:
    """Process the sample inbox and print a formatted results table."""
    # Imported here (not at module scope) so ``import mailmind.cli`` stays cheap.
    from mailmind.agent.agent import MailMindAgent
    from mailmind.schema import Email

    print("MailMind AI — sample inbox demo\n" + "=" * 33 + "\n")

    agent = MailMindAgent()
    emails = [Email.from_dict(payload) for payload in SAMPLE_EMAILS]
    insights = [agent.process_email(email) for email in emails]

    headers = [
        "Subject",
        "Category",
        "Priority",
        "Urgency",
        "Intent",
        "Sentiment",
        "Keywords",
        "Action",
    ]
    rows = [_summarise_insight(insight) for insight in insights]
    print(_render_table(rows, headers))

    print("\nSummaries\n" + "-" * 9)
    for insight in insights:
        print(f"- [{insight.classification.label}] {insight.summary}")

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Currently exposes a single ``demo`` action."""
    parser = argparse.ArgumentParser(
        prog="mailmind",
        description="MailMind AI command-line demo.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=["demo"],
        help="Action to run (default: demo).",
    )
    parser.parse_args(argv)
    return run_demo()


if __name__ == "__main__":
    raise SystemExit(main())
