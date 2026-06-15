#!/usr/bin/env python3
"""Generate the project's documented *sample outputs*.

Runs a diverse, hand-written demo inbox through the full :class:`MailMindAgent`
pipeline and writes two artefacts:

* ``docs/SAMPLE_OUTPUTS.md`` — a human-readable report (table + per-email detail).
* ``data/sample_predictions.json`` — the structured insights for every email.

Run from the repository root::

    PYTHONPATH=src python3 scripts/sample_outputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the package importable without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mailmind import config  # noqa: E402
from mailmind.agent import MailMindAgent  # noqa: E402
from mailmind.schema import Email  # noqa: E402

# --------------------------------------------------------------------------- #
# A deliberately diverse demo inbox (two messages per category).
# --------------------------------------------------------------------------- #
DEMO_INBOX: list[Email] = [
    Email(
        subject="URGENT: Production outage needs your approval",
        body="The payment service is down in production. We need your approval "
        "to roll back the release immediately. Please reply ASAP.",
        sender="oncall@company.com", sender_name="On-Call Team",
        timestamp="2026-06-14T08:40:00", has_attachment=False, num_links=1,
    ),
    Email(
        subject="Your account statement and invoice #482190 are ready",
        body="Your monthly statement for May is available. Invoice #482190 for "
        "$1,240 is due on June 20. Please complete payment to avoid a late fee.",
        sender="billing@bank.com", sender_name="Billing Team",
        timestamp="2026-06-13T10:00:00", has_attachment=True, num_links=1,
    ),
    Email(
        subject="Q3 roadmap review — agenda attached",
        body="Hi team, please review the attached roadmap before our meeting on "
        "Thursday at 2pm. Let me know if you have agenda items to add.",
        sender="pm@work.com", sender_name="Priya Patel",
        timestamp="2026-06-12T14:30:00", has_attachment=True, num_links=0,
    ),
    Email(
        subject="Can you review my pull request today?",
        body="The deploy is blocked on your review of the search refactor. "
        "Could you take a look before EOD? It is fairly small.",
        sender="dev@work.com", sender_name="Tom Walsh",
        timestamp="2026-06-14T09:05:00", has_attachment=False, num_links=1,
    ),
    Email(
        subject="Dinner this weekend?",
        body="Hey! It's been way too long. Are you free for dinner this "
        "Saturday? Thinking of trying that new place downtown. Let me know!",
        sender="sara@gmail.com", sender_name="Sara Lee",
        timestamp="2026-06-13T19:20:00", has_attachment=False, num_links=0,
    ),
    Email(
        subject="Jordan tagged you in a photo",
        body="Jordan and 3 others tagged you in a new photo. See what your "
        "friends are up to and react to their posts in the app.",
        sender="notifications@facebook.com", sender_name="Facebook",
        timestamp="2026-06-11T21:00:00", has_attachment=False, num_links=2,
    ),
    Email(
        subject="FLASH SALE 70% OFF everything ends tonight",
        body="Save big on all items. Use code SAVE70 at checkout. Free shipping "
        "on all orders this week. Shop now before the deal ends. Unsubscribe anytime.",
        sender="deals@shop.com", sender_name="Shop Deals",
        timestamp="2026-06-14T06:00:00", has_attachment=False, num_links=4,
    ),
    Email(
        subject="You have WON a $1000 gift card claim NOW",
        body="Congratulations!!! You are our lucky winner. To claim your $1000 "
        "gift card, click the link and confirm your bank details immediately!!!",
        sender="winner@claim-prize.info", sender_name="Prize Department",
        timestamp="2026-06-14T03:15:00", has_attachment=False, num_links=3,
    ),
]


def _row(insight) -> dict:
    return {
        "subject": insight.email.subject,
        "predicted_category": insight.classification.label,
        "confidence": round(insight.classification.confidence, 3),
        "priority_score": insight.priority.score,
        "priority_band": insight.priority.band,
        "urgency": insight.nlp.urgency.level,
        "intent": insight.nlp.intent.label,
        "sentiment": insight.nlp.sentiment.label,
        "keywords": insight.nlp.keywords[:5],
        "summary": insight.summary,
        "top_action": insight.suggested_actions[0].label
        if insight.suggested_actions else "",
        "flags": insight.flags,
    }


def main() -> None:
    agent = MailMindAgent()
    insights = agent.process_inbox(DEMO_INBOX)
    rows = [_row(i) for i in insights]

    # ---- JSON artefact ----
    json_path = config.DATA_DIR / "sample_predictions.json"
    json_path.write_text(json.dumps(
        {"classifier": type(agent.classifier).__name__,
         "predictions": [i.to_dict() for i in insights]},
        indent=2,
    ))

    # ---- Markdown report ----
    lines: list[str] = []
    lines.append("# MailMind AI — Sample Outputs\n")
    lines.append(
        f"Generated by running `scripts/sample_outputs.py` through the full "
        f"agent pipeline (classifier: **{type(agent.classifier).__name__}**). "
        "The inbox below is sorted by the agent's computed priority score.\n"
    )
    lines.append("## Prioritised inbox\n")
    lines.append("| # | Priority | Category | Conf. | Urgency | Intent | Sentiment | Subject |")
    lines.append("|---|----------|----------|-------|---------|--------|-----------|---------|")
    for n, r in enumerate(rows, 1):
        lines.append(
            f"| {n} | **{r['priority_band']}** ({r['priority_score']}) | "
            f"{r['predicted_category']} | {r['confidence']:.2f} | {r['urgency']} | "
            f"{r['intent']} | {r['sentiment']} | {r['subject'][:46]} |"
        )
    lines.append("\n## Per-email analysis\n")
    for n, (ins, r) in enumerate(zip(insights, rows), 1):
        lines.append(f"### {n}. {r['subject']}\n")
        lines.append(f"- **Sender:** {ins.email.sender}")
        lines.append(
            f"- **Category:** {r['predicted_category']} "
            f"(confidence {r['confidence']:.2f})"
        )
        lines.append(
            f"- **Priority:** {r['priority_band']} — score {r['priority_score']}/100"
        )
        if ins.priority.reasons:
            lines.append(f"  - Reasons: {'; '.join(ins.priority.reasons)}")
        lines.append(f"- **Urgency:** {r['urgency']}  |  **Intent:** {r['intent']}"
                     f"  |  **Sentiment:** {r['sentiment']}")
        lines.append(f"- **Keywords:** {', '.join(r['keywords'])}")
        lines.append(f"- **Summary:** {r['summary']}")
        actions = ", ".join(a.label for a in ins.suggested_actions)
        lines.append(f"- **Suggested actions:** {actions}")
        if r["flags"]:
            lines.append(f"- **Flags:** {', '.join(r['flags'])}")
        lines.append("")

    md_path = config.DOCS_DIR / "SAMPLE_OUTPUTS.md"
    md_path.write_text("\n".join(lines))

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"Processed {len(insights)} emails with {type(agent.classifier).__name__}.")


if __name__ == "__main__":
    main()
