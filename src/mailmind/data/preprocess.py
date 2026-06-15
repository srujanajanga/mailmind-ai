"""Dataset preprocessing helpers shared by training and evaluation.

This module turns a raw email frame (as produced by
:mod:`mailmind.data.dataset_generator`) into a model-ready frame by attaching a
normalised ``clean_text`` column, and provides a stratified train/test split so
every category is represented proportionally in both halves.

The text normalisation is delegated to :func:`mailmind.utils.text.clean_text`,
keeping the cleaning logic in exactly one place across the project.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from mailmind import config
from mailmind.utils.text import clean_text


def add_clean_text(
    df: pd.DataFrame,
    text_from: Sequence[str] = ("subject", "body"),
) -> pd.DataFrame:
    """Return a copy of *df* with ``text`` and ``clean_text`` columns added.

    Args:
        df: Source frame. Must contain every column named in *text_from*.
        text_from: Ordered column names whose values are concatenated (in that
            order) to form the raw ``text`` for each row.

    Returns:
        A new frame (the input is not mutated) with two extra columns:

        * ``text`` — the raw concatenation of *text_from* columns.
        * ``clean_text`` — the normalised, model-ready text from
          :func:`mailmind.utils.text.clean_text`.
    """
    missing = [col for col in text_from if col not in df.columns]
    if missing:
        raise KeyError(f"add_clean_text: missing column(s) {missing}")

    out = df.copy()
    parts = [out[col].fillna("").astype(str) for col in text_from]
    raw = parts[0]
    for part in parts[1:]:
        raw = raw.str.cat(part, sep=". ")
    out["text"] = raw.str.strip()
    out["clean_text"] = out["text"].map(clean_text)
    return out


def train_test_split_frame(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into train/test frames, stratified on the ``label`` column.

    Stratification keeps each category's proportion identical in both splits,
    which is important for the balanced six-class corpus. If the ``label``
    column is absent the split falls back to a plain random partition.

    Args:
        df: Frame to split.
        test_size: Fraction of rows assigned to the test set.
        seed: Seed forwarded to scikit-learn for a reproducible split.

    Returns:
        A ``(train_df, test_df)`` tuple of frames with their original index
        reset to a clean range index.
    """
    stratify = df["label"] if "label" in df.columns else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
