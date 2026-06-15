"""Synthetic dataset subsystem for MailMind AI.

Bundles the labelled-corpus generator and the preprocessing/splitting helpers
used by the training pipeline, re-exporting the public functions for convenient
top-level access::

    from mailmind.data import generate_dataset, train_test_split_frame
"""
from __future__ import annotations

from mailmind.data.dataset_generator import (
    dataset_summary,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from mailmind.data.preprocess import add_clean_text, train_test_split_frame

__all__ = [
    "generate_dataset",
    "save_dataset",
    "load_dataset",
    "dataset_summary",
    "add_clean_text",
    "train_test_split_frame",
]
