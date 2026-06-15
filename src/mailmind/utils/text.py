"""Dependency-light text utilities.

The whole NLP/ML stack funnels through these helpers, so they are written to
work *with or without* heavy optional packages:

* NLTK is used for tokenisation / stop-words / lemmatisation when its data is
  available, but every call silently falls back to a pure-Python implementation
  so the pipeline never crashes on a fresh machine.
* spaCy, if installed, is used opportunistically for richer keyword extraction.
"""
from __future__ import annotations

import re
from functools import lru_cache

# --------------------------------------------------------------------------- #
# A compact built-in stop-word list (fallback when NLTK data is absent)
# --------------------------------------------------------------------------- #
_BUILTIN_STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he her here hers
    herself him himself his how i if in into is isn't it its itself let's me
    more most mustn't my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan't she should shouldn't so some
    such than that the their theirs them themselves then there these they this
    those through to too under until up very was wasn't we were weren't what
    when where which while who whom why with won't would wouldn't you your
    yours yourself yourselves re ll ve s t m d
    """.split()
)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MULTISPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Optional NLTK wiring (best-effort, fully cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _nltk_stopwords() -> frozenset[str]:
    try:
        from nltk.corpus import stopwords  # type: ignore

        return frozenset(stopwords.words("english"))
    except Exception:
        return _BUILTIN_STOPWORDS


@lru_cache(maxsize=1)
def _nltk_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer  # type: ignore

        lemm = WordNetLemmatizer()
        lemm.lemmatize("tests")  # force-trigger the wordnet lookup once
        return lemm
    except Exception:
        return None


def get_stopwords() -> frozenset[str]:
    """Return the active stop-word set (NLTK if available, else built-in)."""
    return _nltk_stopwords()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def normalize_whitespace(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", text or "").strip()


def strip_urls_emails(text: str) -> str:
    text = _URL_RE.sub(" url ", text or "")
    text = _EMAIL_RE.sub(" email ", text)
    return text


def tokenize(text: str, lower: bool = True) -> list[str]:
    """Word-tokenise using a regex (robust and dependency-free)."""
    if not text:
        return []
    if lower:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def clean_text(
    text: str,
    *,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_len: int = 2,
) -> str:
    """Produce a normalised, model-ready version of ``text``.

    Lower-cases, strips URLs/emails, tokenises, optionally removes stop-words
    and lemmatises, and re-joins to a single space-separated string.
    """
    text = strip_urls_emails(text or "")
    tokens = tokenize(text, lower=True)

    stops = get_stopwords() if remove_stopwords else frozenset()
    lemm = _nltk_lemmatizer() if lemmatize else None

    out: list[str] = []
    for tok in tokens:
        if len(tok) < min_len:
            continue
        if tok in stops:
            continue
        if lemm is not None:
            tok = lemm.lemmatize(tok)
        out.append(tok)
    return " ".join(out)


def content_tokens(text: str) -> list[str]:
    """Lower-cased tokens with stop-words removed (no lemmatisation)."""
    stops = get_stopwords()
    return [t for t in tokenize(text, lower=True) if t not in stops and len(t) > 2]


def sentences(text: str) -> list[str]:
    """Split text into sentences (NLTK punkt if available, else regex)."""
    text = normalize_whitespace(text)
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore

        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def uppercase_ratio(text: str) -> float:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def count_links(text: str) -> int:
    return len(_URL_RE.findall(text or ""))


def ensure_nltk_data(quiet: bool = True) -> dict[str, bool]:
    """Best-effort download of the NLTK corpora the project can use.

    Returns a {resource: success} map. Safe to call repeatedly and offline —
    failures are swallowed because every consumer has a fallback path.
    """
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "vader_lexicon": "sentiment/vader_lexicon",
    }
    status: dict[str, bool] = {}
    try:
        import nltk  # type: ignore
    except Exception:
        return {r: False for r in resources}

    for name, path in resources.items():
        try:
            nltk.data.find(path)
            status[name] = True
        except LookupError:
            try:
                status[name] = bool(nltk.download(name, quiet=quiet))
            except Exception:
                status[name] = False
    # Reset caches so freshly downloaded data is picked up.
    _nltk_stopwords.cache_clear()
    _nltk_lemmatizer.cache_clear()
    return status
