# PhishFusion - Email Phishing Detector (content layer)

Course project for Data Science & Cyber, by Lia Bercovitz.

This is the content layer of a two-part, fused phishing detector. It reads only the
email text (subject + body) and decides phishing vs legitimate, explains its
decisions, and - the main focus - is stress-tested for robustness against
text-obfuscation attacks (homoglyphs, leetspeak, full-width characters, word
dilution). Predictions are written to `artifacts/` so they can later be fused with an
independent URL/sender detector.

The written report is in `report/main.pdf`. A plain-language walkthrough of the whole
build -> attack -> fix -> evolve story is in `docs/IMPROVEMENTS_STORY.md`.

## What the project answers

- How well does a TF-IDF + Logistic Regression model separate phishing from legitimate
  email on a single, confound-free corpus?
- How much does character-level obfuscation degrade it, and how much do character
  n-grams and Unicode normalisation recover - including against attacks the defence was
  not designed for?
- Do those attacks also transfer to a DistilBERT baseline?
- What does robustness cost in interpretability and in clean accuracy?
- What is one user report worth against a real phishing campaign, and what new attack
  surface does that feedback channel open?

## Data

The Kaggle "Phishing Email Dataset" compilation (per-source CSVs) goes in
`data/raw/kaggle_phishing/` (not committed; download it from Kaggle). The main
experiments use CEAS_08, the only large corpus with both classes from one source, which
removes the corpus-confound: a probe shows the phishing corpora are about 99% separable
by provenance, so pooling them with a separate legitimate-mail source would let a model
cheat. URLs are stripped (they are the partner detector's job) and the data is
rebalanced to a realistic legit-majority rate (35% phishing). Working set: 26,633
emails, frozen and hashed for reproducibility.

## Main results (real numbers from the executed notebook)

Clean detection (CEAS_08 test set):

| Detector | F1 | MCC | Recall |
|---|---|---|---|
| Keyword-list baseline | 0.25 | -0.05 | 0.21 |
| Multinomial / Complement NB | 0.97 / 0.98 | 0.96 / 0.96 | 0.95 / 0.96 |
| TF-IDF + Logistic Regression (union) | 0.998 | 0.997 | 0.998 |
| DistilBERT (CPU baseline) | 0.984 | 0.975 | 0.995 |

Robustness - recall on phishing under attack and defence:

| Setting | word-only | word + char |
|---|---|---|
| clean | 0.99 | 1.00 |
| homoglyph + leet | 0.71 | 0.93 |
| + normalisation | 0.99 | 1.00 |
| full-width (unseen) | 0.68 | 0.92 |
| + normalisation | 0.99 | 1.00 |
| adaptive (unmapped) | 0.89 | 0.98 |
| + normalisation | 0.96 | 0.99 |

A word-only model collapses under an imperceptible homoglyph attack (recall 0.99 to
0.71). Character n-grams are robust on their own, and normalisation restores the word
model - it even fixes an unseen full-width attack through generic Unicode folding. An
adaptive attacker who avoids the normaliser's hand-list keeps recall at 0.96, so the
defence helps a lot but is not complete; folding through the full Unicode confusables
set (UTS-39) instead of a hand-list restores recall and makes the attack much harder to
land.

Honest checks:

- Cross-corpus: trained on CEAS_08, tested on other corpora, F1 drops 0.998 to 0.37 -
  distribution shift. Re-thresholding recovers F1 to 0.75; the remainder is a vocabulary
  gap (out-of-vocabulary rate 7% to 20%).
- Temporal drift: on 1,303 real 2023-2025 phishing emails, the 2008 model catches only
  14%. Retraining on modern scams lifts that to 98%, with no forgetting and no rise in
  false alarms.
- Word-level dilution (padding a scam with legitimate-looking words) crashes recall to
  0.13. A monotone (non-negative-weight) model over word presence cannot be un-flagged by
  inserted text, at a clean-F1 cost of 0.998 to 0.912 - a safety net beside the main
  model, not a replacement.
- One report per real campaign lifts recall on the rest from 0.15 to 1.00 (campaign-blind
  ablation: 0.33), but only once homoglyphs are normalised first, and the report channel
  can itself be poisoned.

All result tables are in `results/tables/`, all figures in `results/figures/`.

## Improvements (build -> attack -> fix -> evolve)

The project is an iterative loop; the full plain-language version is in
`docs/IMPROVEMENTS_STORY.md`. In short:

1. Built a TF-IDF + Logistic Regression detector - near-perfect on clean in-corpus mail
   and explainable.
2. Attacked it with imperceptible homoglyph and leetspeak swaps; the word-only model
   collapsed.
3. Fixed it with character n-grams and Unicode normalisation; the recovery even held on
   an attack family it had never seen.
4. Hardened it against an adaptive attacker by folding through the full Unicode
   confusables set (UTS-39) instead of a hand-list.
5. Checked reality: both a change of corpus and a jump to 2023-2025 mail break the lab
   number; retraining on modern scams restores it with no forgetting.
6. Closed a second attack class (word-level dilution) with a monotone model that cannot
   be un-flagged by inserted text - a safety net beside the main model.
7. Added a human-in-the-loop step: one reported email propagates to the rest of its
   campaign.

## Future work

Directions I would take the content layer next, each grounded in the report's
references:

- A sentiment / urgency feature that scores emotional pressure as an extra signal.
- Edit-distance (Levenshtein) matching against known brand words, to catch typo-style
  obfuscation the character defence misses.
- A self-updating attack database that retrains on newly confirmed attacks.
- A semantic, meaning-based defence against LLM paraphrase attacks.
- Fusing this content layer with the independent URL/sender layer - the whole point of
  the two-part design.

## Repository layout

```
notebooks/content_detector.ipynb   the deliverable - runs end to end
src/                               config, data, normalisation, attacks, models, explain
docs/IMPROVEMENTS_STORY.md         plain-language build -> attack -> fix -> evolve story
report/main.pdf                    the written report
presentation/presentation.pptx     the slides (English + Hebrew speaker notes)
results/tables/                    result tables (CSV)
results/figures/                   figures (PNG)
artifacts/                         detector predictions (val, test, attacked) for fusion
data/                              parsed modern snapshots (raw Kaggle CSVs not committed)
run_all.py                         re-execute the notebook to regenerate everything
requirements.txt                   dependencies
```

## Running it

```
pip install -r requirements.txt
# put the Kaggle CSVs in data/raw/kaggle_phishing/, then:
python run_all.py
# or open notebooks/content_detector.ipynb and run all cells
```

Everything except DistilBERT is scikit-learn and runs in a few minutes. DistilBERT is
optional and off by default (set RUN_BERT to enable); it uses a GPU if one is present,
otherwise a small CPU subset.

## Notes and limitations

- The attack generator is for defensive evaluation only.
- CEAS_08 is from 2008; the cross-corpus and temporal-drift results are direct evidence
  that a single-corpus model does not transfer, so retraining on current mail is a
  requirement, not an option.
- The cross-corpus number, not the near-perfect in-corpus one, is the honest measure of
  real-world performance.
- This is one layer of a larger system: it deliberately ignores URLs and sender, which
  the partner detector handles, so the two can be fused.
