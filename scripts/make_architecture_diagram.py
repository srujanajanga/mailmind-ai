#!/usr/bin/env python3
"""Render the MailMind AI architecture diagram to ``docs/screenshots/``.

Pure-matplotlib so it needs no system Graphviz. Produces a clean, layered
block diagram suitable for the README and the project slide deck.

    PYTHONPATH=src python3 scripts/make_architecture_diagram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mailmind import config  # noqa: E402

# Palette
INK = "#1d3557"
ACCENT = "#e63946"
TEAL = "#2a9d8f"
SAND = "#e9c46a"
SLATE = "#457b9d"
LIGHT = "#f1faee"


def box(ax, x, y, w, h, text, fc, tc="white", fs=10, bold=True):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor="white", facecolor=fc, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal",
            zorder=3, wrap=True)


def arrow(ax, p1, p2, color=INK):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16, linewidth=1.6,
        color=color, zorder=1, shrinkA=2, shrinkB=2,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    ax.set_title("MailMind AI — System Architecture",
                 fontsize=17, fontweight="bold", color=INK, pad=14)

    # ---- Row 1: ingestion ----
    box(ax, 0.4, 7.0, 3.0, 0.8, "Email Source\n(IMAP / Synthetic Corpus)", SLATE)
    box(ax, 4.9, 7.0, 3.2, 0.8, "Preprocessing\nclean · tokenise · features", SLATE)
    box(ax, 9.6, 7.0, 3.0, 0.8, "SQLite Store\nemails · insights · actions", "#6c757d")

    # ---- Row 2: NLP + ML (parallel analysis) ----
    box(ax, 0.4, 5.2, 5.9, 1.3,
        "NLP Layer  (spaCy / NLTK)\n"
        "keywords · sentiment (VADER)\nurgency cues · intent", TEAL, fs=10)
    box(ax, 6.7, 5.2, 5.9, 1.3,
        "ML Classifier  (scikit-learn)\n"
        "TF-IDF + engineered features\n6-class model — 91% accuracy", ACCENT, fs=10)

    # ---- Row 3: intelligence ----
    box(ax, 0.4, 3.5, 5.9, 1.1,
        "Contextual Intelligence\nsender importance · recency · VIP", SAND, tc=INK)
    box(ax, 6.7, 3.5, 5.9, 1.1,
        "Behavioural Intelligence\nlearns from open/reply/ignore/delete", SAND, tc=INK)

    # ---- Row 4: agent ----
    box(ax, 2.0, 1.7, 9.0, 1.1,
        "Agentic Orchestrator\n"
        "prioritise · flag urgent · summarise · suggest actions · adapt over time",
        INK, fs=11)

    # ---- Row 5: interfaces ----
    box(ax, 1.6, 0.2, 4.2, 0.85, "REST API  (FastAPI)", "#264653")
    box(ax, 7.2, 0.2, 4.2, 0.85, "Dashboard  (Streamlit)", "#264653")

    # ---- arrows ----
    arrow(ax, (3.4, 7.4), (4.9, 7.4))
    arrow(ax, (6.5, 7.0), (5.5, 6.5))   # preprocessing -> NLP
    arrow(ax, (6.5, 7.0), (9.0, 6.5))   # preprocessing -> ML
    arrow(ax, (3.3, 5.2), (3.3, 4.6))   # NLP -> context
    arrow(ax, (9.6, 5.2), (9.6, 4.6))   # ML -> behaviour
    arrow(ax, (3.3, 3.5), (5.5, 2.8))   # context -> agent
    arrow(ax, (9.6, 3.5), (7.5, 2.8))   # behaviour -> agent
    arrow(ax, (4.5, 1.7), (3.7, 1.05))  # agent -> API
    arrow(ax, (8.5, 1.7), (9.3, 1.05))  # agent -> UI
    # feedback loop agent -> store -> behaviour
    arrow(ax, (11.0, 2.2), (11.6, 7.0), color=ACCENT)
    ax.text(12.2, 4.6, "feedback\nloop", color=ACCENT, fontsize=9,
            fontweight="bold", ha="center", va="center", rotation=90)

    fig.tight_layout()
    out = config.FIGURES_DIR / "architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
