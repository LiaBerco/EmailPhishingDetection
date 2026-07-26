"""Load the Kaggle per-source CSVs and harmonise them into one tidy frame.

The corpora have different columns (some carry sender/date/urls, some are just
subject/body/label). We reduce everything to a common schema:

    source | date | subject | body | label

and — importantly for a *content-only* detector — we strip URLs out of the text
so the model cannot secretly lean on the link signal that belongs to the
partner's URL/sender detector (Student B).
"""
from __future__ import annotations
import hashlib
import re
import pandas as pd

from . import config as C

# A URL regex close to the one used when the corpora were built. We replace every
# match with a single " URL " token rather than deleting it, so the *presence* of
# a link is still (weakly) visible but the specific domain/path is gone.
_URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_WS_RE = re.compile(r'\s+')


def strip_urls(text: str) -> str:
    """Replace URLs with a neutral URL token."""
    return _URL_RE.sub(" URL ", str(text))


def clean_text(text: str) -> str:
    """Collapse whitespace and truncate over-long bodies."""
    text = _WS_RE.sub(" ", str(text)).strip()
    return text[: C.MAX_CHARS]


def load_corpus(name: str) -> pd.DataFrame:
    """Load one CSV and return the harmonised columns.

    Missing columns (e.g. Enron has no `date`/`subject`) are filled with "".
    """
    df = pd.read_csv(C.RAW_DIR / name, dtype=str, keep_default_na=False, na_filter=False)
    out = pd.DataFrame()
    out["source"] = [name.replace(".csv", "")] * len(df)
    out["date"] = df["date"] if "date" in df.columns else ""
    out["subject"] = df["subject"] if "subject" in df.columns else ""
    # body first, then strip URLs, then clean
    body = df["body"] if "body" in df.columns else df.get("text_combined", "")
    out["subject"] = out["subject"].map(lambda s: clean_text(strip_urls(s)))
    out["body"] = pd.Series(body, index=df.index).map(lambda s: clean_text(strip_urls(s)))
    out["label"] = pd.to_numeric(df[C.LABEL_COL], errors="coerce").fillna(0).astype(int)
    # drop rows with no usable text at all
    mask = (out["subject"].str.len() > 0) | (out["body"].str.len() > 0)
    return out[mask].reset_index(drop=True)


def load_all(names=None) -> pd.DataFrame:
    """Load and stack several corpora (default: all we use)."""
    if names is None:
        names = list(C.CORPORA.keys())
    frames = [load_corpus(n) for n in names]
    df = pd.concat(frames, ignore_index=True)
    # de-duplicate on the actual text (some corpora share messages)
    df = df.drop_duplicates(subset=["subject", "body"]).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    return df


def make_text(df: pd.DataFrame) -> pd.Series:
    """The single text field the model sees: subject + body."""
    return (df["subject"].astype(str) + " " + df["body"].astype(str)).str.strip()


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
