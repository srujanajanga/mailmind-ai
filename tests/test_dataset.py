"""Tests for the synthetic dataset generator (``mailmind.data``)."""
from __future__ import annotations

from mailmind import config
from mailmind.data import generate_dataset

# The exact column order emitted by the generator.
EXPECTED_COLUMNS = [
    "id",
    "sender",
    "sender_name",
    "sender_domain",
    "subject",
    "body",
    "timestamp",
    "has_attachment",
    "num_links",
    "label",
]


def test_generate_dataset_shape_and_balance():
    """Six balanced classes and the right total row count."""
    per_category = 30
    df = generate_dataset(samples_per_category=per_category, seed=config.RANDOM_SEED)

    assert len(df) == per_category * len(config.CATEGORIES)

    counts = df["label"].value_counts()
    assert set(counts.index) == set(config.CATEGORIES)
    assert all(counts[label] == per_category for label in config.CATEGORIES)


def test_generate_dataset_columns():
    """The frame exposes exactly the expected columns in order."""
    df = generate_dataset(samples_per_category=5, seed=config.RANDOM_SEED)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_generate_dataset_non_empty_subject_and_body():
    """Every generated row has a non-empty subject and body."""
    df = generate_dataset(samples_per_category=10, seed=config.RANDOM_SEED)
    assert df["subject"].astype(str).str.strip().str.len().gt(0).all()
    assert df["body"].astype(str).str.strip().str.len().gt(0).all()


def test_generate_dataset_is_deterministic():
    """Same seed -> identical frames across two independent calls."""
    df_a = generate_dataset(samples_per_category=8, seed=config.RANDOM_SEED)
    df_b = generate_dataset(samples_per_category=8, seed=config.RANDOM_SEED)
    assert df_a.equals(df_b)


def test_generate_dataset_seed_changes_output():
    """A different seed yields a different corpus (sanity check on the PRNG)."""
    df_a = generate_dataset(samples_per_category=8, seed=1)
    df_b = generate_dataset(samples_per_category=8, seed=2)
    assert not df_a.equals(df_b)
