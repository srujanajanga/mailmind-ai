"""Central configuration for MailMind AI.

Every path, label and tunable hyper-parameter lives here so the rest of the
code-base never hard-codes a magic value. Paths are resolved relative to the
repository root, which is derived from this file's location, so the project
works regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
# src/mailmind/config.py  ->  parents[2] == repository root
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
DOCS_DIR: Path = ROOT_DIR / "docs"
FIGURES_DIR: Path = DOCS_DIR / "screenshots"

DATASET_PATH: Path = DATA_DIR / "emails.csv"
MODEL_PATH: Path = MODELS_DIR / "mailmind_classifier.joblib"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"
COMPARISON_PATH: Path = MODELS_DIR / "model_comparison.csv"
DB_PATH: Path = ROOT_DIR / "mailmind.db"

for _d in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Classification labels (the six target categories)
# --------------------------------------------------------------------------- #
CATEGORIES: list[str] = [
    "Important",
    "Work",
    "Personal",
    "Social",
    "Promotions",
    "Spam",
]

# Stable colour map reused by the UI and the evaluation figures.
CATEGORY_COLORS: dict[str, str] = {
    "Important": "#e63946",
    "Work": "#1d3557",
    "Personal": "#2a9d8f",
    "Social": "#457b9d",
    "Promotions": "#e9c46a",
    "Spam": "#6c757d",
}

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42

# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
SAMPLES_PER_CATEGORY: int = 700      # -> ~4,200 row synthetic corpus
TEST_SIZE: float = 0.2

# Fraction of rows deliberately made "hard": their body is borrowed from a
# confusable neighbouring category (label and sender kept), so that the corpus
# carries the realistic overlap real inboxes exhibit (e.g. Promotions vs Spam,
# Important vs Work) instead of being perfectly separable.
DATASET_AMBIGUITY: float = 0.12
# Fraction of rows whose *label* is flipped to a confusable neighbour, modelling
# the human mislabelling present in any real annotated corpus. Off by default so
# the ground truth stays clean for unit tests; the training pipeline enables it.
DATASET_LABEL_NOISE: float = 0.0

# --------------------------------------------------------------------------- #
# Feature / model hyper-parameters
# --------------------------------------------------------------------------- #
TFIDF_PARAMS: dict = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
    "max_features": 20000,
}

# Numeric/engineered feature columns produced by the feature extractor.
NUMERIC_FEATURES: list[str] = [
    "body_length",
    "subject_length",
    "num_links",
    "has_attachment",
    "exclaim_count",
    "uppercase_ratio",
    "urgency_hits",
    "money_hits",
]

# --------------------------------------------------------------------------- #
# Priority scoring weights (used by context.ContextScorer)
# --------------------------------------------------------------------------- #
PRIORITY_WEIGHTS: dict[str, float] = {
    "category": 0.34,      # how important the predicted category is
    "urgency": 0.24,       # urgency cues found by the NLP layer
    "sender": 0.22,        # sender importance (VIP / domain / behaviour)
    "behavior": 0.12,      # learned engagement with this sender/category
    "freshness": 0.08,     # newer mail floats up slightly
}

# Base "importance" score (0-1) the model assigns to each category.
CATEGORY_PRIORITY: dict[str, float] = {
    "Important": 1.00,
    "Work": 0.78,
    "Personal": 0.70,
    "Social": 0.42,
    "Promotions": 0.20,
    "Spam": 0.05,
}

# Default VIP senders / domains considered high importance out of the box.
VIP_DOMAINS: set[str] = {
    "ceo.com", "boss.com", "university.edu", "bank.com", "gov.in",
}
VIP_KEYWORDS: set[str] = {"manager", "director", "ceo", "professor", "dean"}

# Priority bands (score is 0-100).
PRIORITY_BANDS: list[tuple[str, float]] = [
    ("Critical", 80.0),
    ("High", 60.0),
    ("Medium", 40.0),
    ("Low", 0.0),
]

# --------------------------------------------------------------------------- #
# Behavioural-learning weights (how much each action moves engagement)
# --------------------------------------------------------------------------- #
ACTION_WEIGHTS: dict[str, float] = {
    "replied": 1.0,
    "opened": 0.5,
    "ignored": -0.4,
    "deleted": -0.8,
}
VALID_ACTIONS: tuple[str, ...] = tuple(ACTION_WEIGHTS.keys())


def band_for_score(score: float) -> str:
    """Map a 0-100 priority score to a human-readable band."""
    for name, threshold in PRIORITY_BANDS:
        if score >= threshold:
            return name
    return "Low"
