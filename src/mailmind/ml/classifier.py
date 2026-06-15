"""Email category classifiers for MailMind AI.

Two classifiers live here:

* :class:`HeuristicClassifier` — a zero-training, keyword-scoring fallback used
  before a model has been fitted (or when the persisted model is missing).
* :class:`MailMindClassifier` — a thin, persistable wrapper around a scikit-learn
  :class:`~sklearn.pipeline.Pipeline` (preprocessor + estimator) that speaks the
  project's :class:`~mailmind.schema.Classification` data contract.

Both classifiers accept emails in any supported form (Email / dict / str) and
return calibrated probabilities over :data:`mailmind.config.CATEGORIES`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mailmind import config
from mailmind.schema import Classification, Email, as_email
from mailmind.ml.features import build_preprocessor, emails_to_frame


# --------------------------------------------------------------------------- #
# Heuristic (no-training) classifier
# --------------------------------------------------------------------------- #
class HeuristicClassifier:
    """Rule/keyword classifier usable without any training.

    Each category owns a small keyword lexicon; the email text is scored against
    every lexicon and the scores are turned into a probability distribution via
    a soft-max. This provides a sensible cold-start label so the agent always
    has *something* to work with before the ML model is trained.
    """

    #: Per-category keyword lexicons (lower-case, matched as substrings).
    LEXICONS: dict[str, tuple[str, ...]] = {
        "Important": (
            "urgent", "asap", "immediately", "important", "action required",
            "deadline", "critical", "final notice", "verify", "suspended",
            "alert", "security", "expires", "overdue", "attention",
        ),
        "Work": (
            "meeting", "project", "deadline", "report", "client", "schedule",
            "agenda", "invoice", "proposal", "review", "team", "deliverable",
            "standup", "sync", "quarter", "budget", "presentation",
        ),
        "Personal": (
            "family", "dinner", "weekend", "mom", "dad", "friend", "birthday",
            "vacation", "home", "love", "miss you", "catch up", "lunch",
            "thanks", "photos", "kids",
        ),
        "Social": (
            "invitation", "party", "event", "rsvp", "followed", "liked",
            "comment", "friend request", "connect", "network", "tagged",
            "mentioned", "group", "celebrate", "join us", "meetup",
        ),
        "Promotions": (
            "sale", "discount", "offer", "deal", "free", "coupon", "promo",
            "save", "limited time", "shop", "buy now", "% off", "clearance",
            "subscribe", "newsletter", "exclusive",
        ),
        "Spam": (
            "win", "winner", "prize", "lottery", "cash", "claim", "viagra",
            "congratulations", "click here", "guaranteed", "risk-free",
            "act now", "wire transfer", "prince", "bitcoin", "$$$",
        ),
    }

    def __init__(self) -> None:
        self.labels: list[str] = list(config.CATEGORIES)

    def _scores(self, email: "Email | dict | str") -> dict[str, float]:
        text = as_email(email).text.lower()
        scores: dict[str, float] = {}
        for label in self.labels:
            scores[label] = float(
                sum(text.count(kw) for kw in self.LEXICONS.get(label, ()))
            )
        return scores

    def classify(self, email: "Email | dict | str") -> Classification:
        """Classify a single email using keyword scoring."""
        scores = self._scores(email)
        ordered = [scores[label] for label in self.labels]

        if max(ordered) == 0.0:
            # No cue matched anywhere — default to the neutral "Personal" bucket
            # with a flat-ish distribution rather than a fake confident guess.
            uniform = 1.0 / len(self.labels)
            probs = {label: uniform for label in self.labels}
            return Classification(
                label="Personal", confidence=uniform, probabilities=probs
            )

        probabilities = _softmax(ordered)
        probs = dict(zip(self.labels, probabilities))
        best_idx = int(np.argmax(probabilities))
        return Classification(
            label=self.labels[best_idx],
            confidence=float(probabilities[best_idx]),
            probabilities={k: float(v) for k, v in probs.items()},
        )

    def classify_many(
        self, emails: "Iterable[Email | dict | str]"
    ) -> list[Classification]:
        """Classify a batch of emails."""
        return [self.classify(e) for e in emails]

    def predict(self, emails: "Iterable[Email | dict | str]") -> list[str]:
        """Return only the predicted label for each email."""
        return [c.label for c in self.classify_many(emails)]


# --------------------------------------------------------------------------- #
# Machine-learning classifier
# --------------------------------------------------------------------------- #
class MailMindClassifier:
    """Persistable scikit-learn classifier over the six MailMind categories.

    The classifier wraps a :class:`~sklearn.pipeline.Pipeline` of
    ``build_preprocessor()`` + an estimator. It is agnostic to the concrete
    estimator: probabilities come from ``predict_proba`` when available, and are
    otherwise derived from ``decision_function`` via a soft-max so that callers
    always receive a proper distribution aligned to :attr:`labels`.
    """

    def __init__(
        self,
        pipeline: Optional[Pipeline] = None,
        labels: Optional[list[str]] = None,
    ) -> None:
        self.pipeline: Optional[Pipeline] = pipeline
        self.labels: list[str] = list(labels) if labels else list(config.CATEGORIES)

    # -- lifecycle --------------------------------------------------------- #
    @property
    def is_fitted(self) -> bool:
        """Whether the wrapped estimator has been fitted."""
        if self.pipeline is None:
            return False
        estimator = self.pipeline.steps[-1][1]
        try:
            check_is_fitted(estimator)
            return True
        except (NotFittedError, Exception):  # noqa: BLE001 - any state failure
            return False

    def _ensure_pipeline(self) -> Pipeline:
        if self.pipeline is None:
            from sklearn.linear_model import LogisticRegression

            self.pipeline = Pipeline(
                steps=[
                    ("features", build_preprocessor()),
                    (
                        "estimator",
                        LogisticRegression(
                            max_iter=1000, C=4.0, random_state=config.RANDOM_SEED
                        ),
                    ),
                ]
            )
        return self.pipeline

    def fit(
        self,
        emails: "Iterable[Email | dict | str]",
        labels: "Iterable[str]",
    ) -> "MailMindClassifier":
        """Fit the pipeline on ``emails`` with their ground-truth ``labels``."""
        pipeline = self._ensure_pipeline()
        X = emails_to_frame(emails)
        y = list(labels)
        pipeline.fit(X, y)

        # Order our public label list to match the fitted estimator's classes,
        # falling back to the canonical category order for anything unseen.
        fitted_classes = [str(c) for c in pipeline.steps[-1][1].classes_]
        ordered = [c for c in config.CATEGORIES if c in fitted_classes]
        extras = [c for c in fitted_classes if c not in ordered]
        self.labels = ordered + extras
        return self

    # -- prediction -------------------------------------------------------- #
    def _check_ready(self) -> Pipeline:
        if not self.is_fitted or self.pipeline is None:
            raise NotFittedError(
                "MailMindClassifier has not been fitted; call fit() or load()."
            )
        return self.pipeline

    def predict(self, emails: "Iterable[Email | dict | str]") -> list[str]:
        """Predict the category label for each email."""
        pipeline = self._check_ready()
        X = emails_to_frame(emails)
        return [str(p) for p in pipeline.predict(X)]

    def predict_proba(self, emails: "Iterable[Email | dict | str]") -> np.ndarray:
        """Return a ``(n_samples, n_labels)`` probability matrix.

        Columns are aligned to :attr:`labels`. Estimators lacking
        ``predict_proba`` fall back to a soft-max over ``decision_function``.
        """
        pipeline = self._check_ready()
        X = emails_to_frame(emails)
        estimator = pipeline.steps[-1][1]
        estimator_classes = [str(c) for c in estimator.classes_]

        if hasattr(estimator, "predict_proba"):
            proba = np.asarray(pipeline.predict_proba(X), dtype=float)
        else:
            scores = np.asarray(pipeline.decision_function(X), dtype=float)
            if scores.ndim == 1:
                # Binary decision_function -> two-column logits.
                scores = np.column_stack([-scores, scores])
            proba = _softmax_2d(scores)

        return self._align_columns(proba, estimator_classes)

    def _align_columns(
        self, proba: np.ndarray, estimator_classes: list[str]
    ) -> np.ndarray:
        """Reorder probability columns to match :attr:`labels`."""
        index = {cls: i for i, cls in enumerate(estimator_classes)}
        aligned = np.zeros((proba.shape[0], len(self.labels)), dtype=float)
        for j, label in enumerate(self.labels):
            if label in index:
                aligned[:, j] = proba[:, index[label]]
        return aligned

    def classify(self, email: "Email | dict | str") -> Classification:
        """Classify a single email into a :class:`Classification`."""
        return self.classify_many([email])[0]

    def classify_many(
        self, emails: "Iterable[Email | dict | str]"
    ) -> list[Classification]:
        """Classify a batch of emails into :class:`Classification` objects."""
        materialised = list(emails)
        if not materialised:
            return []
        proba = self.predict_proba(materialised)
        results: list[Classification] = []
        for row in proba:
            best_idx = int(np.argmax(row))
            results.append(
                Classification(
                    label=self.labels[best_idx],
                    confidence=float(row[best_idx]),
                    probabilities={
                        label: float(p) for label, p in zip(self.labels, row)
                    },
                )
            )
        return results

    # -- persistence ------------------------------------------------------- #
    def save(self, path: "str | Path" = config.MODEL_PATH) -> None:
        """Persist the fitted pipeline and label order to ``path`` via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "labels": self.labels}, path)

    @classmethod
    def load(cls, path: "str | Path" = config.MODEL_PATH) -> "MailMindClassifier":
        """Load a classifier previously written by :meth:`save`.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No saved model found at {path}")
        payload = joblib.load(path)
        return cls(pipeline=payload.get("pipeline"), labels=payload.get("labels"))


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def _softmax(values: list[float]) -> np.ndarray:
    """Numerically stable soft-max over a 1-D sequence."""
    arr = np.asarray(values, dtype=float)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    total = exp.sum()
    if total == 0.0 or math.isnan(total):
        return np.full(arr.shape, 1.0 / arr.size)
    return exp / total


def _softmax_2d(matrix: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise soft-max over a 2-D array."""
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
