"""Rule-based next-action recommender for MailMind AI.

Given the full enrichment of an email -- its predicted :class:`Classification`,
the :class:`NLPSignals` extracted from its text and the computed
:class:`Priority` -- :func:`suggest_actions` proposes a short, ordered list of
the most useful things the user could do next (reply, unsubscribe, archive ...).

The rules are intentionally transparent: every suggestion carries a short,
human-readable ``reason`` so the UI can explain *why* it was offered.
"""
from __future__ import annotations

from ..schema import Classification, NLPSignals, SuggestedAction, as_email

# Categories whose mail typically warrants a prompt, considered reply.
_RESPONSIVE_CATEGORIES = {"Important", "Work"}
# Intents that signal the sender is waiting on the user.
_REPLY_INTENTS = {"request", "question"}

# Upper bound on how many suggestions we return, keeping the UI focused.
_MAX_ACTIONS = 4

# Generic fallbacks used to guarantee the minimum of two suggestions when the
# rules above produce only the default "archive".
_SECONDARY_FALLBACKS = (
    SuggestedAction(
        "archive",
        "Archive",
        "Keep the inbox tidy once the message has been handled.",
    ),
    SuggestedAction(
        "mark_read",
        "Mark as read",
        "Nothing actionable here -- clear it from the unread count.",
    ),
)


def suggest_actions(
    email: "object",
    classification: Classification,
    nlp: NLPSignals,
    priority: "object",
) -> list[SuggestedAction]:
    """Return an ordered list of suggested actions (most relevant first).

    Between two and four :class:`SuggestedAction` objects are returned. The list
    always ends with a sensible default ("archive") so the user is never left
    without an option, and duplicate actions are collapsed while preserving the
    order in which they were first proposed.

    Parameters
    ----------
    email:
        The email under consideration (``Email``/``dict``/``str`` accepted).
    classification:
        The predicted category and confidence.
    nlp:
        The extracted NLP signals (intent, urgency, sentiment, keywords).
    priority:
        The computed priority, exposing ``score`` and ``band``.

    Returns
    -------
    list[SuggestedAction]
        Two to four ordered, de-duplicated suggestions.
    """
    mail = as_email(email)
    category = classification.label
    intent = nlp.intent.label if nlp.intent is not None else "fyi"
    urgency = nlp.urgency.level if nlp.urgency is not None else "low"
    band = getattr(priority, "band", "Low")

    actions: list[SuggestedAction] = []

    # --- Spam: get it out of the inbox decisively. -----------------------
    if category == "Spam":
        actions.append(
            SuggestedAction(
                "delete",
                "Delete & block",
                "Classified as spam -- safe to remove and block the sender.",
            )
        )
        actions.append(
            SuggestedAction(
                "report_spam",
                "Report spam",
                "Reporting improves future spam filtering.",
            )
        )
        return _finalise(actions)

    # --- A response is expected. -----------------------------------------
    wants_reply = intent in _REPLY_INTENTS or (
        category in _RESPONSIVE_CATEGORIES and urgency == "high"
    )
    if wants_reply:
        if urgency == "high":
            reason = "High urgency and the sender is awaiting a response."
            label = "Reply now"
        else:
            reason = "The sender asked a question or made a request."
            label = "Reply"
        actions.append(SuggestedAction("reply", label, reason))

    # --- Meetings should hit the calendar. -------------------------------
    if intent == "meeting":
        actions.append(
            SuggestedAction(
                "add_to_calendar",
                "Add to calendar",
                "The message proposes or references a meeting.",
            )
        )

    # --- Action-required mail needs a review pass. -----------------------
    if intent == "action_required":
        actions.append(
            SuggestedAction(
                "review",
                "Review & approve",
                "The message requests an explicit action or approval.",
            )
        )

    # --- Promotions: offer an exit and a tidy-up. ------------------------
    if category == "Promotions":
        actions.append(
            SuggestedAction(
                "unsubscribe",
                "Unsubscribe",
                "Promotional mail -- unsubscribe to reduce future noise.",
            )
        )

    # --- Low-value, low-priority mail: dismiss quickly. ------------------
    if band == "Low" and category in {"Social", "Promotions"}:
        actions.append(
            SuggestedAction(
                "mark_read",
                "Mark as read",
                "Low priority -- clear it from the unread count.",
            )
        )

    # --- Default fallback, always available. -----------------------------
    actions.append(
        SuggestedAction(
            "archive",
            "Archive",
            "Keep the inbox tidy once the message has been handled.",
        )
    )

    return _finalise(actions)


def _finalise(actions: list[SuggestedAction]) -> list[SuggestedAction]:
    """De-duplicate by action key (preserving order) and cap the list length.

    A trailing "archive" fallback is appended when fewer than two distinct
    suggestions survive, guaranteeing the contractually required two-to-four
    range.
    """
    seen: set[str] = set()
    deduped: list[SuggestedAction] = []
    for action in actions:
        if action.action in seen:
            continue
        seen.add(action.action)
        deduped.append(action)

    if len(deduped) < 2:
        for fallback in _SECONDARY_FALLBACKS:
            if fallback.action not in seen:
                deduped.append(fallback)
                seen.add(fallback.action)
            if len(deduped) >= 2:
                break

    return deduped[:_MAX_ACTIONS]
