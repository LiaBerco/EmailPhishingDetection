# PhishFusion — Content / NLP Detector (Student A)

Course project for **Data Science & Cyber**. This is the **content** half of a
two-part phishing detector: it reads only the email **text** (subject + body) and
decides **phishing vs legitimate**, explains its decisions, and — the main focus —
is stress-tested for **robustness against text-obfuscation evasion attacks**
(homoglyphs, leetspeak, full-width characters). Predictions are written to
`artifacts/` so they can later be fused with a partner's URL/sender detector
(Student B).

The full design and rationale are in [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md);
the write-up is in [`report/main.tex`](report/main.tex) (Overleaf-ready) with a
readable copy in [`report/report.md`](report/report.md).

## Research questions

* **RQ1 — Signal.** How well does a TF-IDF + Naive-Bayes / Logistic-Regression model
  separate phishing from legitimate email on a single, confound-free corpus?
* **RQ2 — Robustness.** How much does character-level obfuscation degrade the
  detector, and how much do normalisation and character n-grams recover — including
  against attacks the defence was *not* designed for?
* **RQ3 — Transfer.** Do attacks crafted against the lexical model also hurt DistilBERT?
* **RQ4 — Trade-off.** What does robustness cost in interpretability?
* **RQ5 — Guarantees.** Can a defence be made *provable* rather than empirical, and what
  does the guarantee cost in detection accuracy?
* **RQ6 — Deployment.** What is a single user report worth against a real phishing campaign,
  and what new attack surface does that feedback channel create?
* **RQ7 — Neural rewriting.** Can an LLM reword a scam past the detector — the one attack class
  both certificates leave open? (Small probe; the union model resists, the monotone model gives ground.)

## Data

The Kaggle "Phishing Email Dataset" compilation (per-source CSVs) in
`data/raw/kaggle_phishing/`. The main experiments use **CEAS_08** — the only large
corpus that contains **both** classes from one source, which removes the
corpus-confound (a probe shows the phishing corpora are 99% separable by provenance,
so pooling them with a separate ham source would let a model cheat). URLs are
stripped from the text (they are the partner detector's job), and we rebalance to a
realistic **legit-majority** base rate (35% phishing), which also makes the evasion
test meaningful. Working set: **26,633 emails**.

```
# 1) download the Kaggle "Phishing Email Dataset" (Naser Abdullah Alam)
# 2) unzip the CSVs into data/raw/kaggle_phishing/
# 3) run the notebook (or run_all.py) — it harmonises, splits, and hashes the data
```

## Results (real, from the executed notebook)

**Clean detection** (CEAS_08 test set):

| Detector | F1 | MCC | AUC | Recall | Precision |
|---|---|---|---|---|---|
| Keyword-list baseline | 0.247 | −0.05 | 0.48 | 0.21 | 0.30 |
| TF-IDF + Logistic Regression (union) | **0.998** | **0.997** | 1.00 | 0.998 | 0.998 |
| Multinomial NB / Complement NB | 0.972 / 0.976 | 0.958 / 0.964 | 0.999 | — | — |
| DistilBERT (CPU, full-test eval) | 0.984 | 0.975 | — | 0.995 | — |

**Robustness — recall on phishing under attack/defence** (the centrepiece):

| Setting | word-only | word + char (union) |
|---|---|---|
| clean | 0.99 | 1.00 |
| homoglyph + leet (seen) | **0.71** | 0.93 |
| &nbsp;&nbsp;+ normalisation | 0.99 | 1.00 |
| full-width (held-out) | 0.68 | 0.92 |
| &nbsp;&nbsp;+ normalisation | 0.99 | 1.00 |
| adaptive attacker (unmapped) | 0.89 | 0.98 |
| &nbsp;&nbsp;+ normalisation | **0.96** | 0.99 |

The story: a naive **word-only** model collapses under an *imperceptible* homoglyph
attack (recall 0.99 → 0.71). **Character n-grams** are robust on their own, and
**normalisation** restores the word model — it even fixes an *unseen* full-width
attack (via generic Unicode folding). But an **adaptive** attacker who avoids the
normaliser's table keeps recall at 0.96: the defence helps a lot, but is no silver
bullet. That residual gap is what a second, non-text signal is meant to cover.

**A stronger, importance-guided attacker.** The adaptive attacker above perturbs
*uniformly*. A smarter one spends its budget on the model's highest-weight tokens
(the DeepWordBug / TextBugger / HotFlip family). At a *matched* budget it is far more
effective — **8 targeted homoglyph edits cut word-only recall to 0.74, versus 0.98 for
8 random edits** — and a median of just **9 words (~23% of letters, all invisible)**
flips a phishing email. Crucially it lowers the *defence ceiling*: a targeted adaptive
attacker (unmapped glyphs, then normalisation) drops the fully-defended union model to
**0.87** (word 0.67), versus the 0.99/0.96 the *uniform* attacker left. The residual
gap against a competent adversary is wider than a uniform attack implies — the
strongest form of "no silver bullet."

**...and then we close it (the arms race).** Replacing the hand-curated table with the
**full Unicode confusables set (UTS #39)** (`normalize_skeleton`) restores clean-level
recall (**0.993 word / 0.998 union**) against the targeted-adaptive attacker — the old
table did *nothing* (0.673) because the attacker used unmapped glyphs, which the official
set covers. It costs no more than the old table, and it yields a **certificate**: under
worst-case whole-family substitution, `normalize(attacked) == normalize(clean)` for
**100%** of phishing emails (0% for the old table) — the prediction *provably* cannot
change under the homoglyph family. **Iterative targeted adversarial training** closes the
same gap from the model side (recall vs the re-targeted attacker **0.44 → 0.999**, beating
one-shot uniform's 0.975). The honest end state: the cheap character-obfuscation front is
closed with a guarantee, forcing an attacker onto a costlier attack class.

**A second defence — adversarial training** (train on perturbed copies of the
phishing emails) is *complementary* to normalisation: it is near-perfect on the
attack family it trained on (homoglyph+leet, 1.00) but generalises poorly to an
unseen family (full-width, 0.78), whereas normalisation generalises there (0.99).
Combined, they cover every family (≥ 0.98) — so a learned defence belongs *alongside*
a principled input defence, not instead of it.

**Two defences, and they are complementary** (word-only recall):

| Attack family | undefended | normalisation | adversarial training | both |
|---|---|---|---|---|
| homoglyph+leet (seen) | 0.71 | 0.99 | **1.00** | 0.99 |
| full-width (held-out) | 0.68 | **0.99** | 0.78 | 0.99 |
| adaptive (unmapped) | 0.89 | 0.96 | 0.94 | **0.98** |

Adversarial training (retraining on perturbed phishing) is near-perfect on the family
it trained on but generalises poorly to unseen ones; normalisation generalises better
but is weaker against the adaptive attacker. Stacked, they cover every family.

**Cross-corpus (honest generalisation):** trained on CEAS_08, tested on Nazario +
Nigerian-Fraud (phishing) and Enron + Ling (ham), F1 drops **0.998 → 0.37** — the
real-world cost of distribution shift. *Diagnosed:* this is not a total loss of signal
— OOD **AUC is still 0.78** and the model is merely *under-confident* (OOD-phishing mean
score 0.33 vs 0.98 in-corpus), so re-thresholding to 0.1 recovers F1 to **0.75** (a
**prior/threshold shift**); the residual gap is a genuine **vocabulary gap** (OOV rate
7% → 20%). The headline 0.37 overstates the failure, but a real domain gap remains.

**Extra rigour / honesty checks.** Robustness numbers are stable across **K=10 attack
seeds** (sd ≤ 0.004). The normalisation defence is cheap but **not free**: applied to
all clean mail it roughly triples the false-positive rate (union 0.0012 → 0.0040) off a
tiny base. The detector is well-calibrated in-corpus (Brier 0.0020; isotonic → 0.0014)
but its probabilities collapse out-of-distribution — relevant for the fusion step.

**Modern phishing (2023–2025) — reality check.** On **1,303 real phishing emails from
2023–2025** (Jose Nazario's live archive), the CEAS-2008-trained model catches only
**14%** (2025: 11%) — genuine **concept drift**, worse than the cross-corpus number. And
measuring real obfuscation: mixed-script (homoglyph) tokens and zero-width characters are
**~0% in 2008** phishing but **1.8% / 4.0%** in modern phishing — so the attacks we defend
against are a real, emerging technique, not a straw man.

**The next attack class — hacking the words, not the letters.** With the letter-trick front
closed (above), a smart attacker changes *words*. Synonym swaps barely dent the model (0.998),
but **dilution** — burying the scam in legit-sounding filler — crashes recall **0.998 → 0.13**,
because a linear word-counting model just *adds up* signals. The letter-defence does nothing here
(0.997). **Fix:** adversarial training on the word attacks restores recall **0.28 → 1.00** with no
clean-mail loss. (Deeper point: bag-of-words is inherently vulnerable to dilution — structure-aware
features and fusion are the fundamental answers.)

**Evolving the detector (keeping it current).** A 2008 model catches only 15% of modern scams
(0.147 on the held-out modern split used for this experiment; 0.142 over the full 1,303-email
modern set quoted above — same finding, slightly different subset).
After **retraining with modern scam examples** (normal-mail source left unchanged, to avoid the
"cheating trap"), it catches **98%** — with **no forgetting** (0.998 on old scams) and **no rise
in false alarms** (0.7% on modern normal mail). A fully-modern detector scores even higher (F1
0.996) but a source-check shows its two sources are **99% separable by provenance**, so we
disclose that as inflated and quote the clean number.

**From mitigation to proof — a second certificate.** The evidence-floor below only *mitigates*
dilution. So we trained a model that **cannot** be diluted: force every weight to be non-negative
over **binary word-presence** features, and the score becomes *monotonically non-decreasing under
insertion* — adding text can only push it up, so padding **provably cannot** un-flag a phishing
email. Verified exactly: **zero** score decreases and **100%** of flagged phishing still flagged at
+40/+100/+300 inserted words, where the normal model collapses 0.998 → **0.129**. Two caveats we
state rather than bury. First, the proof needs **unigram** features: with bigrams, inserting a word
destroys the pair (`verify account` → `verify now account`), the score *can* fall, and the guarantee
breaks (measured: 388 of 500 emails). Second, **it costs real accuracy** — clean F1 **0.998 → 0.912**
(MCC 0.997 → 0.869), because a monotone model may only collect evidence of *guilt* and never uses a
word as evidence of innocence. So it is a **certified safety net, not a replacement**: OR'd with the
main model it puts a certified floor of **0.87** under heavy dilution (vs 0.13) at a false-alarm cost
of 0.1% → 2.3%. The guarantee covers *insertion* only — deleting or rewriting the scam still evades
it. (This is the classical **good-word attack** of Lowd & Meek 2005; the non-negativity construction
follows Fleshman et al. 2018 for malware.)

**Learning from people — and attacking that too.** A detector in an organisation sits next to
**users**, and targeted phishing goes to many employees at once. So: if one person reports one scam,
can we catch the rest of that campaign? There are **93 real campaigns** (groups of 3+ near-duplicates)
in the 1,303 modern scams. Revealing **one** member per campaign lifts recall on the rest from
**0.149 → 1.000**, with **no rise in false alarms**; one-shot retraining reaches 0.991. But the honest
figure is the **ablation**: remove that campaign's own report and recall falls to **0.329** — so the
gap 1.000 vs 0.329 is the real campaign effect, and the rest is just "modern examples help an old
model". Two attacks on the mechanism itself: **character obfuscation disables it completely** (100%
of campaign emails escape, because homoglyphs destroy the shared words the similarity uses) until the
UTS-39 canonicaliser is applied (back to 3%) — so that defence is a *precondition* for the feedback
loop, not an optional extra; and **false reports poison it**, raising false alarms 0.0012 → 0.0055 at
10% bad reports. The report button is a feature *and* an attack surface.

**How fast and how big is it?** Measured, not guessed: ~3 ms per email (p95 ~6 ms), several hundred
emails/sec/core, ~3 MB on disk, full retrain in seconds. Small and fast enough to run **on the
device**, so message content need never leave it — and retraining cadence is a policy choice, not a
compute limit.

**Further hardening & probing.** (1) *Dilution defence:* a principled, attack-agnostic
"evidence-floor" lifts recall on diluted phishing from **0.13 → 0.36–0.66** (trading false alarms
1%→10%); it doesn't fully solve dilution — a structural bag-of-words weakness. (2) *Paraphrase
attack:* a meaning-preserving rewrite does **not** evade (recall stays 0.998) — dilution remains the
only effective word attack. (3) *Second corpus:* the collapse-and-recover story **replicates** on
SpamAssassin (0.97 → 0.50 → 0.965). (4) *Confound check:* stripping mailing-list boilerplate does
**not** reduce the modern-retrain confound (0.99 → 0.99) — it's genre, not formatting.

The notebook and report also include a fuller **EDA** of interpretable meta-features:
a **Spearman correlation** analysis (phishing emails are *shorter*), **distribution
shape** (skewness/kurtosis — the ratio features are heavily right-skewed, which is *why*
a rank correlation and the median are used), a **VIF** redundancy check (length ≈
word-count, VIF ≈ 21), and a **Mann–Whitney U** test of the length claim (p ≪ 0.001);
plus **outlier** and **temporal** EDA, an **F2** (recall-weighted) metric, a dedicated
**error analysis** (only 4 FP / 4 FN on the clean test set, all borderline), and a
one-page **executive summary**.

All tables are in `results/tables/`, all figures in `results/figures/`.

## Repository layout

```
docs/PROJECT_PLAN.md      full plan: RQs, experiment matrix, methodology, papers
docs/IMPROVEMENTS_STORY.md   plain-language build→hack→fix→evolve narrative (talk script)
docs/DECISIONS_AND_PITFALLS.md   the bugs we caught + how, and the methodology rules
src/                      config · data · normalize · attacks · models(stats) · explain
notebooks/                student_A_content_detector.ipynb  (runs everything, end-to-end)
data/raw/kaggle_phishing/ the per-source CSVs (not committed; download from Kaggle)
data/llm_paraphrase/      frozen LLM paraphrase-attack inputs/verdicts (committed)
artifacts/                detector predictions (val, test, attacked) for fusion
results/tables/           T0..T40 (+T9b/T9c) result tables (CSV)
results/figures/          F1..F24 figures (PNG; incl. F13 EDA box plots)
report/                   main.tex (Overleaf) + report.md + references.bib + main.pdf
presentation/             phishfusion_presentation.html — the 25-slide talk (self-contained)
run_all.py                re-execute the notebook to regenerate everything
```

## Running it

```
pip install -r requirements.txt
# put the Kaggle CSVs in data/raw/kaggle_phishing/ , then:
python run_all.py                 # regenerate all tables/figures (incl. DistilBERT)
# or open notebooks/student_A_content_detector.ipynb and Run All
```

Everything except DistilBERT is scikit-learn and runs in a few minutes. DistilBERT
is optional and guarded (`RUN_BERT` / `PHISH_RUN_BERT=0` to skip); it uses a GPU if
one is present, otherwise a small CPU-bounded subset.

## Academic basis

The pipeline is grounded in the literature (full review in the report): Naive Bayes
spam filtering (Sahami 1998; Metsis 2006), Complement NB for imbalance (Rennie 2003),
character n-grams for robust text (Cavnar & Trenkle 1994), visual/adversarial NLP
attacks and shielding (Eger 2019 VIPER; Pruthi 2019; Boucher 2022), Unicode
confusables (UTS #39), additive "good-word" attacks and provable defences against them
(Lowd & Meek 2005; Jorgensen 2008; Fleshman 2018), phishing-campaign clustering
(Althobaiti 2023), and transformers (Vaswani 2017;
Devlin 2019; Sanh 2019). Two claims are actively *replicated* on our data: Complement
NB's imbalance advantage and the collapse-then-shield behaviour under visual attack; a
third, the good-word attack, we rediscovered independently before connecting it to the
literature.

## Notes / limitations

- The attack generator is for **defensive evaluation only**.
- CEAS_08 is from 2008; the cross-corpus collapse is direct evidence a single-corpus
  model does not transfer — a newer corpus is the obvious next step.
- The cross-corpus number, not the in-corpus one, is the honest measure of real-world
  performance.
