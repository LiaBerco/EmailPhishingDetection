"""Evasion attacks — character-level obfuscation of email text.

Attacks are grouped so the defence can be evaluated honestly (see the plan, 4.7):

  * SEEN     - homoglyph + leetspeak + separators. The normalisation defence is
               designed with this family in mind.
  * HELD-OUT - full-width Latin characters. A *different* family the defence was
               not tuned for; a principled normaliser should still catch it via
               its generic Unicode (NFKD) folding. (attack_zerowidth is a second
               unseen family kept for experimentation.)
  * ADAPTIVE - an attacker who knows about normalisation and uses confusable
               letters that are deliberately absent from the normaliser's table,
               so they survive it. This measures the defence *ceiling*.

`budget` is the per-eligible-character perturbation probability; kept moderate so
the text stays human-readable (the perceptibility constraint used by TextBugger /
VIPER).
"""
from __future__ import annotations
import numpy as np

# Latin -> Cyrillic look-alikes that the normaliser's confusables table DOES cover
# (so the SEEN attack is recoverable). Homoglyphs look identical to a human, so even
# near-total substitution keeps the text readable — the perturbation is effectively
# imperceptible, which is what makes this attack realistic and dangerous.
HOMOGLYPH = {
    "a": "а", "b": "ь", "c": "с", "d": "ԁ", "e": "е", "g": "ɡ", "h": "һ",
    "i": "і", "j": "ј", "k": "к", "l": "ӏ", "m": "м", "n": "ո", "o": "о",
    "p": "р", "r": "г", "s": "ѕ", "t": "т", "u": "υ", "v": "ν", "w": "ѡ",
    "x": "х", "y": "у",
}
# Leetspeak substitutions (the normaliser de-leets these).
LEET = {"a": "@", "o": "0", "i": "1", "e": "3", "s": "$", "t": "7", "l": "1"}
# Confusables the normaliser does NOT list -> used by the adaptive attacker to show
# the defence ceiling (these survive normalisation).
ADAPTIVE = {"a": "ɑ", "e": "ҽ", "c": "ϲ", "i": "ı", "s": "ꜱ", "o": "ᴏ"}
ZERO_WIDTH = ["​", "‌", "⁠"]


def _substitute(text, rng, mapping, budget):
    out = []
    for ch in text:
        low = ch.lower()
        if low in mapping and rng.random() < budget:
            out.append(mapping[low])
        else:
            out.append(ch)
    return "".join(out)


# ---- individual attacks ----------------------------------------------------
def attack_homoglyph(text, rng, budget=0.8):
    return _substitute(text, rng, HOMOGLYPH, budget)


def attack_leet(text, rng, budget=0.5):
    return _substitute(text, rng, LEET, budget)


def attack_separator(text, rng, budget=0.3):
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if ch.isalpha() and i < len(text) - 1 and text[i + 1].isalpha() and rng.random() < budget:
            out.append(".")
    return "".join(out)


def attack_zerowidth(text, rng, budget=0.4):
    """Inject invisible characters between letters."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < budget:
            out.append(str(rng.choice(ZERO_WIDTH)))
    return "".join(out)


def attack_fullwidth(text, rng, budget=0.9):
    """HELD-OUT family: full-width Latin letters (A -> Ａ, U+FF21..).

    These break word tokens just like homoglyphs, but they are NOT in the
    normaliser's confusables table. They are recovered anyway, because the
    normaliser's *general* NFKD step folds full-width forms back to ASCII — so
    this measures whether a principled defence generalises to an unseen family.
    """
    out = []
    for ch in text:
        if ("a" <= ch <= "z" or "A" <= ch <= "Z") and rng.random() < budget:
            # map to the full-width block
            base = ord("Ａ") if ch.isupper() else ord("ａ")
            out.append(chr(base + (ord(ch.lower()) - ord("a")) if ch.islower()
                           else base + (ord(ch) - ord("A"))))
        else:
            out.append(ch)
    return "".join(out)


def attack_mixed(text, rng, budget=0.8):
    """SEEN attack: homoglyphs, a lighter dose of leet, and some separators."""
    t = _substitute(text, rng, HOMOGLYPH, budget)
    t = _substitute(t, rng, LEET, budget * 0.4)
    t = attack_separator(t, rng, budget=0.15)
    return t


def attack_adaptive(text, rng, budget=0.85):
    """ADAPTIVE attacker: only uses confusables absent from the normaliser table."""
    return _substitute(text, rng, ADAPTIVE, budget)


def attack_targeted(text, rng, important, budget=0.9, mapping=None):
    """Grey-box attack: only perturb words the model finds important."""
    mapping = HOMOGLYPH if mapping is None else mapping
    words = text.split(" ")
    out = []
    for w in words:
        key = w.lower().strip(".,!?:;\"'()[]")
        out.append(_substitute(w, rng, mapping, budget) if key in important else w)
    return " ".join(out)


# ---- importance-guided (grey-box) evasion ---------------------------------
# The uniform attacks above perturb every eligible character with equal
# probability. A *smart* attacker does not: it spends its perturbation budget on
# the few tokens the model relies on most, flipping the decision with far fewer
# edits. This is the standard word-importance attack of DeepWordBug (Gao 2018),
# TextBugger (Li 2019) and HotFlip (Ebrahimi 2018), instantiated here on a linear
# model whose per-token pull we can read off exactly from its coefficients.

def word_key(tok):
    """Normalise a raw whitespace token to its lookup key (lower, strip punct)."""
    return tok.lower().strip(".,!?:;\"'()[]<>{}")


def rank_words_by_weight(text, vocab, coef, max_chars=None):
    """Unique alphabetic words in `text` that push toward phishing, highest pull
    first. `vocab` maps term -> column index; `coef` is the linear model's weight
    vector (positive = phishing). Words absent from the vocabulary score 0."""
    if max_chars:
        text = text[:max_chars]
    seen, scored = set(), []
    for tok in text.split(" "):
        k = word_key(tok)
        if not k or not k.isalpha() or k in seen:
            continue
        seen.add(k)
        w = coef[vocab[k]] if k in vocab else 0.0
        if w > 0:
            scored.append((w, k))
    scored.sort(reverse=True)
    return [k for _, k in scored]


def perturb_words(text, targets, mapping):
    """Substitute every letter (via `mapping`) of the tokens whose key is in
    `targets`; leave every other token untouched."""
    out = []
    for tok in text.split(" "):
        if word_key(tok) in targets:
            out.append("".join(mapping.get(ch.lower(), ch) for ch in tok))
        else:
            out.append(tok)
    return " ".join(out)


def attack_importance(text, vocab, coef, mapping=None, budget_words=8, max_chars=None):
    """Grey-box importance-guided evasion: perturb the top-`budget_words` most
    phishing-indicative words with `mapping` (default HOMOGLYPH for an undefended
    target; pass ADAPTIVE for the defence-aware ceiling). Far more efficient per
    edit than the uniform attacks."""
    mapping = HOMOGLYPH if mapping is None else mapping
    ranked = rank_words_by_weight(text, vocab, coef, max_chars=max_chars)
    return perturb_words(text, set(ranked[:budget_words]), mapping)


# ---- word-level / semantic attacks ----------------------------------------
# The attacks above all edit *characters*. Once a principled canonicaliser closes the
# character-obfuscation front (see normalize.normalize_skeleton), a rational attacker
# moves to editing *words* instead. These attacks leave the letters intact but change
# which words the model sees -- a different, harder threat class the character defence
# cannot touch.

# Plainer alternatives for common phishing words (a human reads the same meaning). Used
# by the synonym-substitution attack. Curated (offline, reproducible) rather than a big
# thesaurus, so the map is transparent and stable.
WORD_SYNONYMS = {
    "verify": "confirm", "verification": "confirmation", "account": "profile",
    "password": "credentials", "click": "select", "urgent": "important",
    "urgently": "promptly", "suspend": "pause", "suspended": "paused",
    "login": "signin", "security": "safety", "update": "refresh", "bank": "institution",
    "banking": "institution", "winner": "recipient", "prize": "reward", "dear": "hello",
    "alert": "notice", "expire": "lapse", "expired": "lapsed", "immediately": "now",
    "confirm": "check", "restricted": "limited", "invoice": "bill", "payment": "transfer",
    "refund": "reimbursement", "access": "entry", "unusual": "unexpected", "detected": "seen",
    "secure": "safe", "unauthorized": "unapproved", "billing": "charges",
    "credentials": "details", "notification": "message",
}


def attack_synonym(text, vocab, coef, budget_words=40, max_chars=None):
    """Replace the top phishing-pulling words with a plainer synonym (meaning kept)."""
    targets = {w for w in rank_words_by_weight(text, vocab, coef, max_chars=max_chars)
               if w in WORD_SYNONYMS}
    if budget_words:
        targets = set(list(targets)[:budget_words])
    out = []
    for tok in text.split(" "):
        k = word_key(tok)
        out.append(WORD_SYNONYMS[k] if k in targets else tok)
    return " ".join(out)


def attack_wordmask(text, vocab, coef, budget_words=20, max_chars=None):
    """Delete the top-`budget_words` phishing-pulling words (attacker rewrites around them)."""
    drop = set(rank_words_by_weight(text, vocab, coef, max_chars=max_chars)[:budget_words])
    return " ".join(tok for tok in text.split(" ") if word_key(tok) not in drop)


def attack_insert(text, ham_words, n, rng):
    """DILUTION: scatter `n` legitimate-looking words through the email so their combined
    'ham' weight outvotes the phishing words. `ham_words` is a pool of strongly-legitimate
    tokens (e.g. the model's most negative-coefficient words)."""
    if n <= 0 or not ham_words:
        return text
    pad = list(rng.choice(ham_words, size=min(n, len(ham_words)), replace=False))
    toks = text.split(" ")
    for w in pad:
        toks.insert(int(rng.integers(0, len(toks) + 1)), w)
    return " ".join(toks)


def attack_paraphrase(text, rng):
    """Rule-based, meaning-preserving rewrite: swap every word we have a synonym for and
    shuffle the sentence order. A cheap stand-in for a neural paraphraser (the honest
    stronger test) — used to probe whether the detector survives sentence-level rewriting."""
    toks = [WORD_SYNONYMS.get(word_key(w), w) for w in text.split(" ")]
    parts = [p.strip() for p in " ".join(toks).split(".") if p.strip()]
    rng.shuffle(parts)
    return ". ".join(parts)


# ---- apply to a whole column ----------------------------------------------
def attack_corpus(texts, labels, attack_fn, seed, only_positive=True, **kwargs):
    """Return a list of (optionally only-phishing) perturbed texts.

    Attacking only the positive (phishing) rows models an attacker trying to slip
    phishing past the filter; legitimate mail is left untouched.
    """
    rng = np.random.default_rng(seed)
    labels = list(labels)
    out = []
    for t, y in zip(texts, labels):
        if only_positive and y != 1:
            out.append(t)
        else:
            out.append(attack_fn(t, rng, **kwargs))
    return out
