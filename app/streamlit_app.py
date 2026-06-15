"""MailMind AI — interactive Streamlit dashboard.

A single-file, polished front-end for the MailMind AI email assistant. It lets a
user load a synthetic demo inbox, watch the agent classify and prioritise every
message, drill into the NLP signals behind any email and record feedback that
the behavioural learner uses to personalise future scoring.

Run it from the repository root with::

    streamlit run app/streamlit_app.py

The module inserts the project ``src`` directory onto ``sys.path`` at import time
so the ``mailmind`` package resolves without an editable install. Every call into
``mailmind`` is wrapped defensively: if the trained model is missing the agent
transparently falls back to the rule-based :class:`HeuristicClassifier`, and the
dashboard still renders end-to-end.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# --------------------------------------------------------------------------- #
# Make the ``mailmind`` package importable without installation.
# app/streamlit_app.py -> parents[1] is the repo root, /src holds the package.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mailmind import config  # noqa: E402  (import after sys.path tweak)
from mailmind.schema import Email  # noqa: E402

# --------------------------------------------------------------------------- #
# Presentation constants
# --------------------------------------------------------------------------- #
PAGE_TITLE = "MailMind AI — Your Inbox, Intelligently Organized"

BAND_EMOJI: dict[str, str] = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "⚪",
}
BAND_ORDER: list[str] = ["Critical", "High", "Medium", "Low"]

SENTIMENT_EMOJI: dict[str, str] = {
    "positive": "🙂",
    "neutral": "😐",
    "negative": "🙁",
}

ACTION_EMOJI: dict[str, str] = {
    "opened": "👀",
    "replied": "↩️",
    "ignored": "🙈",
    "deleted": "🗑️",
}

DEMO_PER_CATEGORY = 4   # 4 * 6 categories -> a tidy 24-email demo inbox


# --------------------------------------------------------------------------- #
# Cached singletons — the agent + database are expensive, build them once.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Booting the MailMind agent…")
def get_agent() -> Any:
    """Return a process-wide :class:`MailMindAgent`.

    The agent loads the trained model when available and otherwise initialises
    itself around the heuristic classifier, so this never raises even on a fresh
    checkout. Returns ``None`` only if the agent module cannot be imported at all.

    The agent is wired to the *same* cached :class:`Database` the sidebar reads
    from, so feedback recorded in the dashboard is persisted and the
    "Feedback events recorded" counter reflects it.
    """
    try:
        from mailmind.agent.agent import MailMindAgent

        return MailMindAgent(db=get_database())
    except Exception:  # pragma: no cover - defensive, depends on sibling modules
        return None


@st.cache_resource(show_spinner=False)
def get_database() -> Any:
    """Return a process-wide :class:`Database`, or ``None`` if unavailable."""
    try:
        from mailmind.db.database import Database

        return Database()
    except Exception:  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- #
# Demo data
# --------------------------------------------------------------------------- #
def _fallback_demo_inbox(per_category: int) -> list[Email]:
    """Build a small synthetic inbox without depending on ``mailmind.data``.

    Used only when the real dataset generator is not importable (it is written in
    parallel by a sibling subsystem). The samples are deliberately category-typical
    so the classifier and heuristics both have signal to work with.
    """
    templates: dict[str, list[tuple[str, str]]] = {
        "Important": [
            ("URGENT: account security alert",
             "We detected a sign-in from a new device. Please verify immediately to "
             "keep your account secure. This requires action today."),
            ("Action required: contract signature",
             "The legal team needs your signature before end of day to finalise the "
             "deal. Please review the attached terms and respond ASAP."),
        ],
        "Work": [
            ("Q3 planning sync",
             "Can we meet tomorrow at 10am to align on the Q3 roadmap and assign "
             "owners for each workstream? Agenda attached."),
            ("Code review feedback",
             "Left a few comments on your pull request. Mostly minor, but the retry "
             "logic needs another look before we merge."),
        ],
        "Personal": [
            ("Dinner this weekend?",
             "Hey! It's been ages. Want to grab dinner on Saturday and catch up? Let "
             "me know what works for you."),
            ("Photos from the trip",
             "Finally sorted through the holiday photos — sharing a few of my "
             "favourites. Hope you're doing well!"),
        ],
        "Social": [
            ("Alex tagged you in a photo",
             "You have a new notification. Alex tagged you in a photo. See what your "
             "friends are sharing today."),
            ("New connection request",
             "Jordan wants to connect with you on the network. View their profile and "
             "respond to the request."),
        ],
        "Promotions": [
            ("50% OFF everything — today only!",
             "Our biggest sale of the season is here! Save 50% on your entire order. "
             "Shop now before this exclusive deal ends at midnight."),
            ("You've earned a $20 reward",
             "Congratulations! A $20 reward is waiting in your account. Redeem it on "
             "your next purchase and enjoy free shipping."),
        ],
        "Spam": [
            ("Congratulations! You WON $1,000,000",
             "You have been selected as our lucky winner! Claim your million dollar "
             "prize now by clicking the link and sending your bank details."),
            ("Re: your inheritance funds",
             "Dear friend, I am a barrister handling a large inheritance. I need your "
             "urgent help to transfer the funds. Reply with your account number."),
        ],
    }

    senders: dict[str, tuple[str, str, str]] = {
        "Important": ("security", "ceo.com", "Security Team"),
        "Work": ("teammate", "work.com", "Sam Carter"),
        "Personal": ("jamie", "gmail.com", "Jamie Lee"),
        "Social": ("noreply", "socialnet.com", "SocialNet"),
        "Promotions": ("offers", "shopmail.com", "MegaStore Deals"),
        "Spam": ("winner", "lucky-prize.biz", "Prize Dept"),
    }

    rng = random.Random(config.RANDOM_SEED)
    inbox: list[Email] = []
    for category in config.CATEGORIES:
        user, domain, name = senders[category]
        pool = templates[category]
        for i in range(per_category):
            subject, body = pool[i % len(pool)]
            day = rng.randint(1, 28)
            hour = rng.randint(0, 23)
            inbox.append(
                Email(
                    subject=subject,
                    body=body,
                    sender=f"{user}{i + 1}@{domain}",
                    sender_name=name,
                    recipient="me@inbox.com",
                    timestamp=f"2026-06-{day:02d}T{hour:02d}:00:00",
                    has_attachment=category in {"Important", "Work"} and i % 2 == 0,
                    num_links=2 if category in {"Promotions", "Spam"} else 0,
                    thread_id=f"{category.lower()}-{i}",
                    label=category,
                )
            )
    rng.shuffle(inbox)
    return inbox


def build_demo_inbox(per_category: int = DEMO_PER_CATEGORY) -> list[Email]:
    """Generate a demo inbox, preferring the real ``mailmind.data`` generator.

    Tries ``mailmind.data.generate_dataset`` first and coerces whatever it returns
    (a list of :class:`Email`, dicts, or a DataFrame) into ``Email`` objects.
    Falls back to a self-contained generator if the data module is absent.
    """
    total = per_category * len(config.CATEGORIES)
    try:
        from mailmind.data import generate_dataset  # type: ignore

        raw = generate_dataset(samples_per_category=per_category)
        emails = _coerce_to_emails(raw)
        if emails:
            return emails[:total]
    except Exception:  # pragma: no cover - sibling module may differ/absent
        pass
    return _fallback_demo_inbox(per_category)


def _coerce_to_emails(raw: Any) -> list[Email]:
    """Best-effort conversion of a generator's output into ``Email`` objects."""
    if raw is None:
        return []
    # pandas DataFrame -> list of row dicts
    if hasattr(raw, "to_dict") and hasattr(raw, "columns"):
        try:
            raw = raw.to_dict(orient="records")
        except Exception:  # pragma: no cover - defensive
            return []
    if not isinstance(raw, (list, tuple)):
        return []

    emails: list[Email] = []
    for item in raw:
        if isinstance(item, Email):
            emails.append(item)
        elif isinstance(item, dict):
            try:
                emails.append(Email.from_dict(item))
            except Exception:  # pragma: no cover - defensive
                continue
    return emails


# --------------------------------------------------------------------------- #
# Insight processing (cached on the inbox identity to avoid re-running models)
# --------------------------------------------------------------------------- #
def process_inbox(emails: list[Email]) -> list[Any]:
    """Run the agent over ``emails`` and return insights sorted by priority.

    Falls back to an empty list if the agent is unavailable, keeping the page
    renderable. The result is cached in session state keyed by the inbox so the
    (potentially heavy) model only runs when the inbox actually changes.
    """
    agent = get_agent()
    if agent is None or not emails:
        return []

    cache_key = _inbox_signature(emails)
    cached = st.session_state.get("_insight_cache")
    if cached and cached.get("key") == cache_key:
        return cached["insights"]

    try:
        insights = agent.process_inbox(emails)
    except Exception as exc:  # pragma: no cover - defensive on sibling modules
        st.warning(f"Could not process the inbox: {exc}")
        insights = []

    st.session_state["_insight_cache"] = {"key": cache_key, "insights": insights}
    return insights


def _inbox_signature(emails: list[Email]) -> str:
    """A stable identity for a list of emails, used as a cache key."""
    return "|".join(e.id for e in emails)


# --------------------------------------------------------------------------- #
# Small rendering helpers
# --------------------------------------------------------------------------- #
def _band_label(band: str) -> str:
    """Band name decorated with its status emoji."""
    return f"{BAND_EMOJI.get(band, '⚪')} {band}"


def _color_swatch(category: str) -> str:
    """An inline coloured dot followed by the category name (HTML)."""
    color = config.CATEGORY_COLORS.get(category, "#888888")
    return (
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
        f"background:{color};margin-right:6px;'></span>{category}"
    )


def _chip(text: str, color: str = "#1d3557") -> str:
    """Render a rounded badge/chip as an HTML string."""
    return (
        f"<span style='background:{color}1a;color:{color};border:1px solid {color}55;"
        f"border-radius:12px;padding:2px 10px;margin:2px;display:inline-block;"
        f"font-size:0.8rem;'>{text}</span>"
    )


def _insight_for(insights: list[Any], email_id: str) -> Optional[Any]:
    """Look up the insight whose underlying email matches ``email_id``."""
    for ins in insights:
        if getattr(ins.email, "id", None) == email_id:
            return ins
    return None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    """Render the sidebar: branding, demo loader and agent status."""
    with st.sidebar:
        st.title("📬 MailMind AI")
        st.caption("Your Inbox, Intelligently Organized")
        st.divider()

        if st.button("📥 Load demo inbox", use_container_width=True, type="primary"):
            with st.spinner("Generating a demo inbox…"):
                st.session_state["emails"] = build_demo_inbox()
            st.session_state.pop("_insight_cache", None)
            st.toast("Loaded a fresh demo inbox.", icon="📥")

        emails = st.session_state.get("emails", [])
        st.metric("Emails in inbox", len(emails))

        st.divider()
        st.subheader("Engine status")
        agent = get_agent()
        classifier_name, model_loaded = _classifier_status(agent)
        st.write(f"**Classifier:** {classifier_name}")
        if model_loaded:
            st.success("Trained model loaded", icon="✅")
        else:
            st.info("Heuristic fallback (no trained model)", icon="🧭")

        db = get_database()
        if db is not None:
            try:
                st.caption(f"Feedback events recorded: {sum(db.action_counts().values())}")
            except Exception:  # pragma: no cover - defensive
                pass

        st.divider()
        st.caption("Built with scikit-learn, NLTK & Streamlit.")


def _classifier_status(agent: Any) -> tuple[str, bool]:
    """Return ``(classifier_class_name, trained_model_loaded)`` defensively."""
    if agent is None:
        return ("unavailable", False)
    classifier = getattr(agent, "classifier", None)
    if classifier is None:
        return ("HeuristicClassifier", False)
    name = type(classifier).__name__
    loaded = bool(getattr(classifier, "is_fitted", False))
    return (name, loaded)


# --------------------------------------------------------------------------- #
# Inbox tab
# --------------------------------------------------------------------------- #
def render_inbox_tab(insights: list[Any]) -> None:
    """Render the sortable, filterable inbox table plus the email detail panel."""
    st.subheader("📨 Inbox")
    if not insights:
        st.info("Load the demo inbox from the sidebar to get started.", icon="👈")
        return

    filtered = _apply_filters(insights)
    if not filtered:
        st.warning("No emails match the current filters.")
        return

    _render_inbox_table(filtered)
    st.divider()
    render_email_detail(filtered)


def _apply_filters(insights: list[Any]) -> list[Any]:
    """Apply the category / band / search-box filters chosen in the UI."""
    cols = st.columns([2, 1, 2])
    with cols[0]:
        categories = st.multiselect(
            "Categories", options=config.CATEGORIES, default=config.CATEGORIES
        )
    with cols[1]:
        bands = st.multiselect("Priority band", options=BAND_ORDER, default=BAND_ORDER)
    with cols[2]:
        query = st.text_input("Search subject / sender", value="", placeholder="e.g. invoice")

    query_lc = query.strip().lower()
    out: list[Any] = []
    for ins in insights:
        if ins.classification.label not in categories:
            continue
        if ins.priority.band not in bands:
            continue
        if query_lc:
            haystack = f"{ins.email.subject} {ins.email.sender} {ins.email.sender_name}".lower()
            if query_lc not in haystack:
                continue
        out.append(ins)
    return out


def _render_inbox_table(insights: list[Any]) -> None:
    """Render the ranked inbox as a styled, sortable dataframe."""
    import pandas as pd

    rows = []
    for ins in insights:
        rows.append(
            {
                "Priority": _band_label(ins.priority.band),
                "Score": round(ins.priority.score, 1),
                "Category": ins.classification.label,
                "Sender": ins.email.sender_name or ins.email.sender,
                "Subject": ins.email.subject,
                "Urgency": ins.nlp.urgency.level,
                "Intent": ins.nlp.intent.label,
            }
        )
    frame = pd.DataFrame(rows)

    def _highlight_category(value: str) -> str:
        color = config.CATEGORY_COLORS.get(value, "")
        return f"color: {color}; font-weight: 600;" if color else ""

    styled = frame.style.map(_highlight_category, subset=["Category"])

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )


# --------------------------------------------------------------------------- #
# Email detail panel
# --------------------------------------------------------------------------- #
def render_email_detail(insights: list[Any]) -> None:
    """Render the per-email deep-dive for the email selected in the selectbox."""
    st.subheader("🔍 Email detail")

    def _format_option(email_id: str) -> str:
        ins = _insight_for(insights, email_id)
        if ins is None:
            return email_id
        return f"{_band_label(ins.priority.band)}  ·  {ins.email.subject[:60]}"

    options = [ins.email.id for ins in insights]
    selected_id = st.selectbox(
        "Choose an email", options=options, format_func=_format_option, key="detail_select"
    )
    insight = _insight_for(insights, selected_id)
    if insight is None:
        return

    email = insight.email
    st.markdown(f"### {email.subject or '(no subject)'}")
    meta = f"**From:** {email.sender_name or 'Unknown'} &lt;{email.sender}&gt;"
    if email.timestamp:
        meta += f"  ·  **Received:** {email.timestamp}"
    st.markdown(meta, unsafe_allow_html=True)

    flag_bits = []
    if email.has_attachment:
        flag_bits.append("📎 attachment")
    if email.num_links:
        flag_bits.append(f"🔗 {email.num_links} link(s)")
    for flag in insight.flags:
        flag_bits.append(f"🚩 {flag}")
    if flag_bits:
        st.markdown("  ".join(_chip(b) for b in flag_bits), unsafe_allow_html=True)

    st.markdown("**Body**")
    st.text(email.body or "(empty body)")

    st.divider()
    _render_insight_metrics(insight)
    st.divider()
    _render_nlp_section(insight)
    st.divider()
    _render_actions_and_feedback(insight)


def _render_insight_metrics(insight: Any) -> None:
    """Show classification + priority headline metrics side by side."""
    cls = insight.classification
    prio = insight.priority
    cols = st.columns(3)
    cols[0].markdown("**Predicted category**")
    cols[0].markdown(_color_swatch(cls.label), unsafe_allow_html=True)
    cols[0].caption(f"Confidence: {cls.confidence:.0%}")

    cols[1].metric("Priority score", f"{prio.score:.1f}", _band_label(prio.band))
    cols[2].markdown("**Why this priority**")
    if prio.reasons:
        for reason in prio.reasons:
            cols[2].markdown(f"- {reason}")
    else:
        cols[2].caption("No specific drivers recorded.")

    if cls.probabilities:
        with st.expander("Class probabilities"):
            import pandas as pd

            prob_frame = (
                pd.Series(cls.probabilities, name="probability")
                .sort_values(ascending=False)
                .to_frame()
            )
            st.bar_chart(prob_frame)


def _render_nlp_section(insight: Any) -> None:
    """Show sentiment, urgency, intent, keywords and the extractive summary."""
    nlp = insight.nlp
    cols = st.columns(3)

    sent = nlp.sentiment
    cols[0].markdown("**Sentiment**")
    cols[0].markdown(
        f"{SENTIMENT_EMOJI.get(sent.label, '😐')} {sent.label.title()}"
    )
    cols[0].caption(f"score {sent.score:+.2f} · conf {sent.confidence:.0%}")

    urg = nlp.urgency
    cols[1].markdown("**Urgency**")
    cols[1].markdown(f"⏱️ {urg.level.title()}  ·  {urg.score:.0%}")
    if urg.cues:
        cols[1].caption("Cues: " + ", ".join(urg.cues))

    intent = nlp.intent
    cols[2].markdown("**Intent**")
    cols[2].markdown(f"🎯 {intent.label.replace('_', ' ').title()}")
    cols[2].caption(f"confidence {intent.confidence:.0%}")

    if nlp.keywords:
        st.markdown("**Keywords**")
        st.markdown(
            "".join(_chip(k, config.CATEGORY_COLORS.get("Work", "#1d3557")) for k in nlp.keywords),
            unsafe_allow_html=True,
        )

    if insight.summary:
        st.markdown("**Summary**")
        st.info(insight.summary)


def _render_actions_and_feedback(insight: Any) -> None:
    """Show suggested actions and the four feedback buttons."""
    if insight.suggested_actions:
        st.markdown("**Suggested actions**")
        action_cols = st.columns(max(len(insight.suggested_actions), 1))
        for col, action in zip(action_cols, insight.suggested_actions):
            help_text = action.reason or None
            col.button(action.label, key=f"sa_{insight.email.id}_{action.action}", help=help_text)

    st.markdown("**Record your feedback**")
    st.caption("Feedback trains the behavioural learner and personalises future scoring.")
    fb_cols = st.columns(len(config.VALID_ACTIONS))
    for col, action in zip(fb_cols, config.VALID_ACTIONS):
        label = f"{ACTION_EMOJI.get(action, '•')} {action.title()}"
        if col.button(label, key=f"fb_{insight.email.id}_{action}", use_container_width=True):
            _record_feedback(insight.email, action)


def _record_feedback(email: Email, action: str) -> None:
    """Send a feedback action to the agent and surface the outcome to the user."""
    agent = get_agent()
    if agent is None:
        st.warning("Agent unavailable — feedback not recorded.")
        return
    try:
        agent.record_feedback(email, action)
        st.session_state.pop("_insight_cache", None)  # rescore on next render
        st.toast(f"Recorded '{action}'. Future scoring updated.", icon="✅")
        st.success(f"Feedback '{action}' recorded for this sender.")
    except Exception as exc:  # pragma: no cover - defensive on sibling modules
        st.error(f"Could not record feedback: {exc}")


# --------------------------------------------------------------------------- #
# Analytics tab
# --------------------------------------------------------------------------- #
def render_analytics_tab(insights: list[Any]) -> None:
    """Render distribution charts and any saved evaluation metrics/figures."""
    st.subheader("📊 Analytics")
    if insights:
        _render_distribution_charts(insights)
    else:
        st.info("Load the demo inbox to see live distributions.", icon="👈")

    st.divider()
    _render_saved_metrics()
    _render_saved_figures()


def _render_distribution_charts(insights: list[Any]) -> None:
    """Bar charts for category mix and priority-band mix of the live inbox."""
    import pandas as pd

    cat_counts = {c: 0 for c in config.CATEGORIES}
    band_counts = {b: 0 for b in BAND_ORDER}
    for ins in insights:
        cat_counts[ins.classification.label] = cat_counts.get(ins.classification.label, 0) + 1
        band_counts[ins.priority.band] = band_counts.get(ins.priority.band, 0) + 1

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Category distribution**")
        cat_frame = pd.DataFrame({"emails": cat_counts})
        st.bar_chart(cat_frame, color="#1d3557")
    with cols[1]:
        st.markdown("**Priority band distribution**")
        band_frame = pd.DataFrame(
            {"emails": [band_counts[b] for b in BAND_ORDER]}, index=BAND_ORDER
        )
        st.bar_chart(band_frame, color="#e63946")


def _render_saved_metrics() -> None:
    """Display the trained model's saved accuracy/F1 metrics, if present."""
    metrics_path = config.METRICS_PATH
    if not metrics_path.exists():
        st.caption("No saved model metrics yet — train a model to populate this section.")
        return

    import json

    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception:  # pragma: no cover - defensive
        st.caption("Saved metrics file could not be parsed.")
        return

    st.markdown("**Trained model performance**")
    accuracy = _first_present(metrics, ("accuracy", "test_accuracy"))
    f1 = _first_present(metrics, ("f1", "f1_macro", "macro_f1", "f1_score"))
    model_name = _first_present(metrics, ("model", "model_name", "best_model"))

    cols = st.columns(3)
    cols[0].metric("Accuracy", f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else "—")
    cols[1].metric("F1 (macro)", f"{f1:.2%}" if isinstance(f1, (int, float)) else "—")
    cols[2].metric("Best model", str(model_name) if model_name is not None else "—")

    with st.expander("Raw metrics"):
        st.json(metrics)


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first value found among ``keys`` in ``data`` (else ``None``)."""
    for key in keys:
        if key in data:
            return data[key]
    return None


def _render_saved_figures() -> None:
    """Embed any saved evaluation figures (confusion matrix, etc.)."""
    figures_dir = config.FIGURES_DIR
    if not figures_dir.is_dir():
        return
    images = sorted(
        p for p in figures_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        return
    st.markdown("**Saved evaluation figures**")
    for image_path in images:
        st.image(
            str(image_path),
            caption=image_path.stem.replace("_", " ").title(),
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #
def render_about_tab() -> None:
    """Render a short description of the system and its tech stack."""
    st.subheader("ℹ️ About MailMind AI")
    st.markdown(
        """
**MailMind AI** is an agentic email assistant that reads your inbox the way a
thoughtful chief-of-staff would: it understands each message, decides how much it
matters and tells you what to do next.

For every email the agent runs a full pipeline:

1. **Classification** — a TF-IDF + scikit-learn model (with a transparent
   rule-based fallback) sorts mail into *Important, Work, Personal, Social,
   Promotions* and *Spam*.
2. **NLP enrichment** — keyword extraction, VADER sentiment, urgency detection
   and intent recognition surface the signals hidden in the text.
3. **Priority scoring** — a weighted model blends category importance, urgency,
   sender reputation, your learned behaviour and freshness into a single 0–100
   score and a *Critical / High / Medium / Low* band.
4. **Action & learning** — the agent summarises the message, suggests next
   actions, and folds your feedback (opened / replied / ignored / deleted) back
   into future scoring through a behavioural learner.

#### Tech stack
- **Python 3.9+** with a clean ``src`` layout
- **scikit-learn**, **scipy**, **NumPy**, **pandas** for the ML core
- **NLTK** (punkt, stopwords, WordNet, VADER) for classical NLP
- **joblib** for model persistence and **SQLite** for behavioural memory
- **Streamlit** for this dashboard; **matplotlib** for evaluation figures

Everything runs locally on a synthetic corpus — no real email and no cloud APIs.
        """
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Configure the page and render the full dashboard."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📬", layout="wide")

    st.title("📬 MailMind AI")
    st.caption("Your Inbox, Intelligently Organized")

    render_sidebar()

    emails: list[Email] = st.session_state.get("emails", [])
    insights = process_inbox(emails)

    inbox_tab, analytics_tab, about_tab = st.tabs(["📨 Inbox", "📊 Analytics", "ℹ️ About"])
    with inbox_tab:
        render_inbox_tab(insights)
    with analytics_tab:
        render_analytics_tab(insights)
    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()
