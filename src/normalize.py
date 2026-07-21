"""Normalisation defence — fold obfuscated text back towards plain ASCII.

Design principle (this matters for an honest evaluation): the defence must NOT be
the exact inverse of our own attack, otherwise "it recovers" is a tautology.
So the normaliser is built from *general* rules that a real defender could write
without knowing the attacker's exact substitution table:

  1. Unicode NFKD normalisation + strip combining marks  -> folds many look-alikes
     and full-width variants automatically.
  2. Remove zero-width / invisible characters.
  3. Map a curated table of confusable letters (Cyrillic/Greek -> Latin), the kind
     the Unicode "confusables" data (UTS #39) is built from.
  4. Lower-case.
  5. (optional) de-leet obvious digit/symbol substitutions.
  6. (optional) collapse single-character separators inside a word ("u.r.g.e.n.t").

Because the rules are general, the attack can still slip past them (a confusable
we did not list, or an invisible char we did not strip) — which is exactly the
"held-out / adaptive attacker" gap we want to measure, not hide.
"""
from __future__ import annotations
import unicodedata
from pathlib import Path

# Zero-width and other invisible characters an attacker can inject between letters.
ZERO_WIDTH = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E], None
)

# A curated confusables table (Cyrillic/Greek homoglyphs -> Latin). This is a
# practical stand-in for the full Unicode UTS #39 confusables set: it covers the
# common Latin/Cyrillic look-alikes a defender would list. It is intentionally
# INCOMPLETE — a few exotic look-alikes are left out so the adaptive attacker has
# something to exploit (see attacks.ADAPTIVE).
CONFUSABLES = {
    # Cyrillic look-alikes (the letters our SEEN homoglyph attack uses)
    "а": "a", "ь": "b", "с": "c", "ԁ": "d", "е": "e", "ɡ": "g", "һ": "h",
    "і": "i", "ј": "j", "к": "k", "ӏ": "l", "м": "m", "ո": "n", "о": "o",
    "р": "p", "г": "r", "ѕ": "s", "т": "t", "υ": "u", "ν": "v", "ѡ": "w",
    "х": "x", "у": "y",
    # a few common Greek look-alikes as well
    "ο": "o", "α": "a", "ε": "e", "ρ": "p", "τ": "t", "ι": "i", "κ": "k",
    # NOT listed (used by the adaptive attacker): ɑ ҽ ϲ ı ꜱ ᴏ
}
_CONF_TABLE = {ord(k): v for k, v in CONFUSABLES.items()}

LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "@": "a"}
_LEET_TABLE = {ord(k): v for k, v in LEET.items()}

# Punctuation an attacker injects between letters (u.r.g.e.n.t). NOTE: space is
# intentionally excluded — collapsing spaces would merge real words together.
_SEPARATORS = set(".-_*|/~")


def strip_invisible(text: str) -> str:
    return text.translate(ZERO_WIDTH)


def fold_unicode(text: str) -> str:
    """NFKD, then drop combining marks; folds full-width & many accented forms."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def map_confusables(text: str) -> str:
    return text.translate(_CONF_TABLE)


def de_leet(text: str) -> str:
    return text.translate(_LEET_TABLE)


def collapse_separators(text: str) -> str:
    """Remove separators sitting between two single letters: u.r.g.e.n.t -> urgent.

    We only join when a separator is flanked by letters, so real punctuation
    between words is mostly preserved.
    """
    out = []
    for i, ch in enumerate(text):
        if ch in _SEPARATORS and 0 < i < len(text) - 1:
            if text[i - 1].isalpha() and text[i + 1].isalpha():
                continue  # drop this separator
        out.append(ch)
    return "".join(out)


def normalize_text(text: str, deleet: bool = True, separators: bool = True) -> str:
    """Full defensive normalisation pipeline (rule-based, curated table).

    This is the *original* defence: a small hand-curated confusables table plus
    generic NFKD folding. It is intentionally incomplete, which is what the
    adaptive attacker exploits. Kept as the baseline in the arms-race comparison;
    `normalize_skeleton` is the upgraded, principled defence."""
    text = str(text)
    text = strip_invisible(text)
    text = map_confusables(text)   # map look-alikes before folding/casing
    text = fold_unicode(text)
    text = text.lower()
    if deleet:
        text = de_leet(text)
    if separators:
        text = collapse_separators(text)
    return text


# ---------------------------------------------------------------------------
# Upgraded defence: the FULL Unicode confusables skeleton (UTS #39).
#
# Instead of a hand-listed table, we fold every character through the official
# Unicode "confusables" mapping (the same data browsers use to detect spoofed
# domains). This is the principled version of the defence: it closes the gap the
# adaptive attacker opened, because the unmapped look-alikes it relied on ARE in
# the official table. We load the mapping from a bundled copy of the Unicode data
# file for offline reproducibility, and fall back to the curated table if absent.
# ---------------------------------------------------------------------------
_CONFUSABLES_FILE = Path(__file__).resolve().parents[1] / "data" / "unicode" / "confusables.txt"


def _load_skeleton_table():
    """Build ord -> replacement from the Unicode confusables file, restricted to
    non-ASCII sources (so clean ASCII is never altered), with the curated
    Cyrillic/Greek->ASCII table taking priority where the two overlap."""
    table = {}
    try:
        for line in _CONFUSABLES_FILE.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or ";" not in line:
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            try:
                src = int(parts[0], 16)
                tgt = "".join(chr(int(cp, 16)) for cp in parts[1].split())
            except ValueError:
                continue
            if src >= 128:                      # never remap clean ASCII
                table[src] = tgt
    except OSError:
        pass                                     # file missing -> curated fallback below
    # curated ASCII targets win over the (sometimes non-ASCII) official prototype
    table.update(_CONF_TABLE)
    return table


_SKELETON_TABLE = _load_skeleton_table()
SKELETON_LOADED = len(_SKELETON_TABLE) > len(_CONF_TABLE)   # True once the file parsed


def map_skeleton(text: str) -> str:
    """Fold every character through the full Unicode confusables skeleton."""
    return text.translate(_SKELETON_TABLE)


def normalize_skeleton(text: str, deleet: bool = True, separators: bool = True) -> str:
    """Upgraded defence: full Unicode-confusables skeleton, then NFKD fold, then
    de-leet / separator collapse. Same shape as `normalize_text`, stronger table."""
    text = str(text)
    text = strip_invisible(text)
    text = map_skeleton(text)      # full confusables table (was: curated only)
    text = fold_unicode(text)
    text = text.lower()
    if deleet:
        text = de_leet(text)
    if separators:
        text = collapse_separators(text)
    return text
