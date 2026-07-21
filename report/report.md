# Content-Based Phishing Detection with an Emphasis on Adversarial Robustness and Explainability

**Student A — Content / NLP Detector** · Data Science & Cyber

*This is a readable Markdown copy of the report. The formal, citation-typeset version is [`main.tex`](main.tex) (Overleaf-ready). All numbers come from the executed notebook (`notebooks/student_A_content_detector.ipynb`); tables/figures live in `results/`.*

---

## Abstract

Phishing email remains a leading entry point for attacks, and much of the signal lives in the message **text**. This project builds a content-only phishing detector with a classical NLP pipeline (TF-IDF features with Naive Bayes and Logistic Regression) and studies it along three axes: detection quality, explainability, and — the main focus — robustness to text-obfuscation evasion. On **CEAS_08**, the one large corpus with both classes from a single source, the task is easy on clean text (Logistic Regression F1 = 0.998); but we treat that near-perfect score as a property of clean, single-corpus data — not a claim about the wild — and the rest of the work is a deliberate arms race that tests when it holds. A word-only model is **fragile**: an imperceptible homoglyph attack drops recall from 0.99 to 0.71. Character n-grams and a Unicode-normalisation defence restore it and even repair an *unseen* full-width attack, but an adaptive, importance-guided attacker still opens a residual gap (union 0.99 → 0.87). We then close the character front with a **proof**: folding through the full Unicode confusables set (UTS #39) makes the prediction certifiably invariant to that attack family. Honest evaluation shows the lab number survives neither a change of corpus (F1 0.998 → 0.37, diagnosed as prior-shift plus vocabulary gap) nor time (a 2008 model catches 14% of real 2023–2025 phishing); **evolving** on modern mail lifts that to 0.98 with no forgetting. Pushing into the next attack class, a word-level **dilution** attack still evades the model (→ 0.13), which we answer with a second, orthogonal certificate: a monotone (non-negative-weight) model provably cannot be un-flagged by inserted text, at an honest cost of clean F1 0.998 → 0.912 — a certified safety net, not a replacement. We also examine the detector as part of a deployed system: one simulated user report protects the rest of a phishing campaign, and an LLM-paraphrase probe (validated by a second LLM) leaves the main model unmoved. Every headline number carries a bootstrap confidence interval, and classifiers are compared with McNemar's test. Throughout, where a defence is incomplete we say so — which is the core argument for fusing this content channel with a second, independent (URL/sender) detector.

---

## 1. Introduction

Phishing is a social-engineering attack in which a message impersonates a trustworthy sender to trick the recipient into revealing credentials, transferring money, or running malware. Email is its main delivery channel, and much of the deception is carried by **language**: urgency ("your account will be suspended"), authority ("security team"), calls to action ("verify now"). That makes natural language processing a natural detection tool.

This project is the **content** half of a two-part detector. Our job is to read only the subject and body and decide phishing vs legitimate; a partner detector (Student B) handles URLs and sender metadata, and the two are later fused. To keep the split clean we deliberately strip URLs out of the text — a content model should earn its score from language, not links.

Two properties beyond raw accuracy drive the work. **Explainability**: an analyst needs to know *why* a message was flagged. **Robustness**: filters have been in an arms race with attackers for decades, and a favourite evasion trick is to mangle text — `vеrify` with a look-alike character, or `cl1ck` in leetspeak — so a human still reads the word but a keyword filter does not.

**Contributions.** This is an empirical study, not a new method, and its value is in rigour and honesty. **(1)** A confound-controlled measurement of lexical detection on a single mixed-class corpus, with a robustness study that separates *seen*, *held-out*, and *adaptive* attacks so a defence is never tested against its own inverse. **(2)** An *arms race* where each defence provokes a smarter attack: char n-grams + normalisation answer homoglyph obfuscation, an importance-guided attacker lowers the ceiling, and a principled UTS #39 canonicaliser closes the character front with a *certified* invariance guarantee. **(3)** Honest external validity: the in-corpus number survives neither a change of corpus (F1 0.998 → 0.37, diagnosed as prior-shift + vocabulary gap) nor time (14% on real 2023–25 phishing), and *evolving* on modern mail recovers it to 0.98. **(4)** A second attack class — word-level *dilution* — answered by a second, orthogonal certificate: a monotone model that cannot be un-flagged by inserted text, with its accuracy cost and feature-order assumption stated plainly. **(5)** The detector examined *as part of a deployed system*: user-report campaign propagation (with its obfuscation and poisoning risks), an LLM-paraphrase probe validated by a second LLM, and a measured deployment envelope. Throughout we replicate two classical results (Complement NB under imbalance; lexical fragility to visual attacks), report bootstrap CIs and McNemar tests, and state every incomplete defence as such.

---

## 2. Background and Literature Review

**Classical spam/phishing filters.** Statistical junk-mail filtering goes back to Sahami et al. (1998), who framed spam as Naive Bayes text classification; Metsis et al. (2006) compared NB variants and is the template for our model comparison. These methods are fast and interpretable but treat text as a bag of words — blind to word order and brittle to obfuscation. Phishing-specific work (PILFER, Fette et al. 2007; Abu-Nimeh et al. 2007; Bergholz et al. 2010) added engineered features (URLs, HTML, headers) — powerful but ageing, and exactly the signals a content detector ignores. Salloum et al. (2022) give a recent systematic review.

**Features and robustness.** Character n-grams have long supported robust text categorisation (Cavnar & Trenkle 1994; Zhang et al. 2015). This is directly relevant — we find char n-grams far more resistant to obfuscation, because `cl1ck` still shares chunks with `click`.

**Adversarial NLP.** Small, human-imperceptible perturbations can flip a classifier (Goodfellow et al. 2015). In text, character attacks — DeepWordBug (Gao et al. 2018), TextBugger (Li et al. 2019), HotFlip (Ebrahimi et al. 2018) — evade classifiers with typos and look-alikes while staying readable (the *perceptibility constraint* we adopt). Boucher et al. (2022) show imperceptible Unicode tricks break deployed NLP systems. On defence, Eger et al. (2019, VIPER) study visual attacks and shielding, and Pruthi et al. (2019) normalise text before classification — our strategy, with a rule-based normaliser from the Unicode security data (UTS #39). Our robustness experiments are a scoped, tightened replication of the VIPER attack/shield idea on phishing email.

**Additive attacks and provable defences.** A separate, older evasion strategy does not disguise the message at all: it *adds* innocuous content until the classifier's summed evidence tips the other way. Lowd & Meek (2005) formalised this as the **good-word attack** on statistical spam filters, showing that appending enough common benign words pushes a large share of blocked spam past the filter — our dilution attack (§4.10) is an instance of it, which we rediscovered empirically before connecting it to this literature. Defences have mostly been heuristic: Jorgensen et al. (2008) counter it with multiple-instance learning over message segments. A stronger option is to make the attack impossible rather than unlikely — Fleshman et al. (2018) show, for malware, that constraining a model to non-negative weights means added features can only *increase* the maliciousness score, so evasion requires *removing* malicious content. We adopt that construction for phishing text in §4.12b and, because a certificate is only meaningful next to its price, measure what the constraint costs in clean accuracy.

**Transformers.** Self-attention (Vaswani et al. 2017) and BERT (Devlin et al. 2019) model context that bag-of-words ignores; DistilBERT (Sanh et al. 2019) is a smaller distillation used as a baseline. We report MCC throughout, as Chicco & Jurman (2020) argue it is more informative than F1/accuracy on imbalanced problems.

**Where this sits.** Most phishing-NLP work reports clean accuracy. Little reports robustness to cheap obfuscation *evaluated against unseen attacks*, and fewer quantify what robustness costs in interpretability. That gap is our target.

---

## 3. Methodology

**Data and the corpus-confound problem.** We use the Kaggle "Phishing Email Dataset" compilation (per-source CSVs). The danger with such packs is *corpus confounding*: if all phishing came from one corpus and all ham from another, a classifier could score highly just by recognising *which corpus* a message came from. We demonstrate this directly (§4.1). To avoid it, all main experiments use **CEAS_08**, the only large corpus containing **both** classes from one source — so the confound is impossible by construction.

Two deliberate preprocessing choices follow. **URLs are stripped** (replaced by a `URL` token), so the content model cannot ride on link features. We **rebalance to a legit-majority base rate** (35% phishing): real inboxes are mostly legitimate, and a phishing-majority prior actually *hides* evasion attacks (a model that loses its signal falls back to the majority class). Working set: 26,633 emails; a SHA-256 hash of the frozen snapshot is saved for reproducibility.

**Features.** Each email (subject + body) becomes a **TF-IDF** vector — Term Frequency × Inverse Document Frequency, which down-weights common words and up-weights rare, discriminative ones like "verify account". Three configurations: **word** (1–2 grams, stop-words removed), **char** (character 3–5 grams), **union** (both).

**Models.** Naive Bayes applies Bayes' theorem with the naive conditional-independence assumption. **Laplace (add-one) smoothing** solves the *zero-frequency problem* (one unseen word should not zero out a class). **Complement NB** (Rennie et al. 2003) targets class imbalance — which we *test*, not assume. **Logistic Regression** is the interpretable linear baseline. Selection uses validation F1; the test set is touched once.

**Evaluation and statistics.** We report accuracy, precision, recall, F1, MCC, AUC. Every headline metric carries a **bootstrap 95% CI** (1000 resamples); classifiers are compared with **McNemar's exact test**. Because a false negative (missed phishing) is worse than a false positive (false alarm), we also pick a **cost-optimal threshold** on validation with FN weighted 5× FP.

**Threat model and attacks.** The attacker edits text but not the pipeline (test-time *evasion*), perturbing only phishing emails (so we measure **recall on phishing**), bounded to stay human-readable. Three families:
- **Seen** — homoglyphs (Latin→Cyrillic) + leetspeak (the family the normaliser targets);
- **Held-out** — full-width characters (a different family, not tuned for);
- **Adaptive** — confusables deliberately absent from the normaliser's table.

In the standard adversarial-ML taxonomy these are all test-time **evasion** attacks (turning phishing into "legit"), never training-time **poisoning** — with one exception: the user-report channel (§5.1), where false reports are a genuine poisoning/label-flipping attack. The uniform attacks are *indiscriminate*; the importance-guided one (§4.6a) is *targeted* and **grey-box** (it reads the linear coefficients, not the training data); and we test *transferability* against DistilBERT (§4.8). We do **not** study model-extraction/theft (cloning the detector via query access) — a real risk for a deployed API, noted as future work.

**Defence.** Normalise before classifying: map a curated Unicode-confusables table to Latin, apply generic NFKD folding, strip invisible characters, lower-case, de-leet, collapse in-word separators. The table is intentionally incomplete — that is what gives the adaptive attacker room and lets us measure the defence *ceiling* rather than a tautology.

---

## 4. Experiments and Results

### 4.1 The corpus-confound probe
A classifier trained to predict *which corpus* a phishing email came from (CEAS / Nazario / Nigerian-Fraud) reached **98.8%** accuracy vs 33% chance. The corpora are trivially separable by provenance, so pooling single-class phishing corpora with a separate ham source would let a "detector" cheat — hence the single-corpus design.

### 4.2 Trivial baselines and the accuracy paradox

| | Accuracy | Precision | Recall | F1 | MCC | AUC |
|---|---|---|---|---|---|---|
| Majority (all legit) | 0.650 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |
| Keyword list | 0.556 | 0.304 | 0.208 | 0.247 | −0.054 | 0.476 |

The majority classifier scores 65% accuracy while catching **zero** phishing — the **accuracy paradox** in one line.

### 4.2b Exploratory data analysis of interpretable meta-features

TF-IDF is too high-dimensional to correlate token-by-token, so we engineered a few interpretable meta-features (length, word count, digit/uppercase/non-ASCII ratios, URL-token count) and ran the standard EDA toolkit on them — distribution shape, a rank correlation, a redundancy diagnostic, and a hypothesis test. The goal is not to add model inputs (the lexical model already saturates) but to turn each downstream decision into a *tested* statement.

**Distribution shape decides the right statistics.** We measured the **skewness** (asymmetry) and **excess kurtosis** (tailedness) of each feature. The ratio features are strongly right-skewed and heavy-tailed — average word length has skewness ≈ **6.6** and excess kurtosis > **60**, digit ratio skewness ≈ 3.2 — so they are nowhere near normal. That is exactly why **Spearman** rank correlation (and the **median**, not the mean) is the correct tool: monotonic association without Pearson's normality/linearity assumptions, robust to heavy tails, while Kendall's τ targets smaller samples. Character length looks only mildly skewed here *because* every email is truncated at `MAX_CHARS`, capping the long right tail by construction — a preprocessing choice that doubles as outlier control (an IQR/MAD scan on the truncated length flags essentially nothing).

**Length is the clearest signal — and we test it.** Phishing emails are *shorter* than legit ones (ρ ≈ −0.57 chars, −0.53 words) — a scam is a short, urgent message; legit mailing-list mail is long. Rather than eyeball a histogram, we test it: the phishing median length (≈ 290 chars) is far below the legit median (≈ 1,185), and because the distributions are skewed we use the non-parametric **Mann–Whitney U** test, which rejects equal distributions at *p ≪ 0.001*. Even so, the strongest meta-feature is only *moderate* — far from the near-perfect lexical model — confirming the real signal is *which words appear*.

**Redundancy, quantified.** Length and word count are nearly collinear (ρ ≈ 0.97). We quantify it with the **variance inflation factor (VIF)**, the standard multicollinearity diagnostic: both have **VIF ≈ 21**, well past the conventional threshold of 10, while every other meta-feature sits near 1–3. They carry almost the same information, so feeding both to a linear model would split one signal across two coefficients. This is why the whole meta-feature block is kept **diagnostic-only** and never enters the classifier.

**A data-quality check on the dates.** The corpus's `date` headers are 99.7% within the expected 2007–2009 window. We explicitly checked for the classic Unix-epoch trap — a *missing* date imputed to 0 becomes `1970-01-01` and would masquerade as a real timestamp in a temporal view — and found **none** (0 of 34,342 emails). The ≈0.3% of off-window dates are genuinely *spoofed* RFC-822 headers (implausible years such as 2086), which the temporal analysis already excludes, so the single-period snapshot is trustworthy. (Reproducible via `scripts/check_date_trap.py`.)

*(Distribution-shape table → `T9b_eda_shape`; VIF → `T9c_eda_vif`; box plots by class → `F13`.)*

### 4.3 Clean detection: features and models

Feature ablation (Logistic Regression):

| Features | Accuracy | Precision | Recall | F1 | F2 | MCC | F1 (95% CI) |
|---|---|---|---|---|---|---|---|
| word | 0.9962 | 0.9968 | 0.9925 | 0.9946 | 0.9933 | 0.9917 | 0.995 [0.992, 0.997] |
| char | 0.9968 | 0.9946 | 0.9962 | 0.9954 | 0.9959 | 0.9930 | 0.995 [0.993, 0.997] |
| **union** | **0.9985** | 0.9979 | 0.9979 | **0.9979** | 0.9979 | **0.9967** | 0.998 [0.996, 0.999] |

Model comparison (union features):

| Model | Accuracy | Precision | Recall | F1 | F2 | MCC | F1 (95% CI) |
|---|---|---|---|---|---|---|---|
| Multinomial NB | 0.9807 | 0.9977 | 0.9469 | 0.9716 | 0.9566 | 0.9577 | 0.972 [0.966, 0.977] |
| Complement NB | 0.9835 | 0.9972 | 0.9555 | 0.9759 | 0.9635 | 0.9638 | 0.976 [0.971, 0.981] |
| **Logistic Regression** | **0.9985** | 0.9979 | 0.9979 | **0.9979** | 0.9979 | **0.9967** | 0.998 [0.996, 0.999] |

*(F2 weights recall above precision — the right emphasis for phishing, where a miss costs more than a false alarm.)*

Logistic Regression wins, and McNemar's test confirms the gap over Complement NB is significant (*p* ≪ 0.001). Complement NB edges Multinomial NB, consistent with the literature.

### 4.4 Does Complement NB help under imbalance?
Rennie et al. (2003) claim CNB corrects NB under imbalance. We tested it: holding the vectoriser fixed, we retrained MNB and CNB on subsets from 5% to 50% phishing. The claim holds — at **5% phishing, CNB 0.944 vs MNB 0.892**; at a genuine 50/50 balance the gap disappears entirely (0.976 vs 0.976). A genuine confirmation of the mechanism on a different corpus: the advantage is *specifically* an imbalance correction, and it vanishes exactly where the theory says it should. (Above 35% phishing the working set has too few phishing emails to hit the target share by adding them, so we thin the legitimate mail instead — otherwise every point above 35% would silently repeat the 35% subset.)

### 4.5 Tuning and generalisation
A validation-only grid search over `C` gave a negligible validation-to-test F1 gap (no overfitting). The cost-optimal threshold (0.40, below the default 0.5) raises recall 0.9979 → 0.9984 at a tiny precision cost, reflecting the "a miss is worse than a false alarm" asymmetry.

### 4.6 Adversarial robustness — the main result

Recall on phishing under attack and defence:

| Setting | word-only | union |
|---|---|---|
| clean | 0.992 | 0.998 |
| homoglyph+leet (seen) | 0.711 | 0.933 |
| &nbsp;&nbsp;+ normalise | 0.992 | 0.998 |
| full-width (held-out) | 0.677 | 0.920 |
| &nbsp;&nbsp;+ normalise | 0.993 | 0.998 |
| adaptive (unmapped) | 0.888 | 0.983 |
| &nbsp;&nbsp;+ normalise | **0.963** | 0.987 |

On clean text both models catch nearly all phishing. Under an **imperceptible** homoglyph+leet attack the **word-only** model collapses (0.99 → 0.71) — nearly 30% of phishing sails through with no visible change. Two defences answer it: **character n-grams** are robust in their own right (union only drops to 0.93), and **normalisation** restores the word model to 0.99. Against a **held-out** full-width attack the word model is equally fragile (0.68) yet normalisation still repairs it (0.99) — via *generic* Unicode folding, not because we listed full-width characters. But an **adaptive** attacker keeps recall at 0.96 even after normalisation: the defence helps a lot, but is no silver bullet.

*(Multi-seed check: repeating every attack over K=10 seeds gives standard deviations ≤ 0.004 recall, so these drops and recoveries are stable, not lucky draws.)*

### 4.6a A stronger, importance-guided attacker

The adaptive attacker above perturbs *uniformly*. A real adversary spends its budget on the model's highest-weight tokens — the word-importance attack of DeepWordBug (Gao 2018), TextBugger (Li 2019) and HotFlip (Ebrahimi 2018), implemented here in the spirit of TextAttack (Morris 2020). Because the model is linear, each token's phishing pull is exactly its coefficient, so we rank the words of every phishing email and homoglyph only the top *k*. At a **matched** budget the targeted attacker is far stronger:

| words perturbed | targeted recall | uniform recall |
|---|---|---|
| 5 | 0.878 | 0.987 |
| 8 | 0.741 | 0.981 |
| 20 | 0.440 | 0.955 |

A median of just **9 words (~23% of letters, all invisible homoglyphs)** flips a phishing email; 58% of phishing emails are evaded within 25 edits. Crucially it lowers the *defence ceiling*:

| Attacker (all + normalise) | word recall | union recall |
|---|---|---|
| uniform adaptive | 0.963 | 0.987 |
| targeted adaptive (k=8) | 0.836 | 0.955 |
| targeted adaptive (k=20) | 0.673 | **0.879** |

The uniform adaptive attacker left the defended union model at 0.99; a *targeted* one drops it to 0.87 (word 0.67). This does not overturn the main result — char n-grams and normalisation still remove most of the risk — but the residual gap against a competent, defence-aware adversary is wider than a uniform attack implies: the strongest form of "no silver bullet."

### 4.6a-ii What the defence costs

Normalisation is applied to *every* email, so it must not damage legitimate mail. Normalising the whole clean test set leaves recall unchanged but raises the union model's false-positive rate from 0.0012 to 0.0040 (word 0.0017 → 0.0049), flipping about a dozen emails (mostly legitimate). The defence is cheap but **not free** — a small, quantifiable precision cost on ordinary traffic, worth stating rather than hiding.

### 4.6b Adversarial training — a second, complementary defence

Normalisation cleans the input *before* the model; **adversarial training** does the opposite — it augments the training set with homoglyph+leet perturbed copies of the phishing *training* emails and retrains (only training rows perturbed, so no leakage). Word-only recall on phishing:

| Attack family | undefended | normalisation | adv. training | adv. + norm |
|---|---|---|---|---|
| clean | 0.992 | 0.993 | 0.995 | 0.995 |
| homoglyph+leet (seen) | 0.711 | 0.992 | **1.000** | 0.995 |
| full-width (held-out) | 0.677 | **0.993** | 0.781 | 0.995 |
| adaptive (unmapped) | 0.888 | 0.963 | 0.938 | **0.983** |

The two defences fail differently. **Adversarial training** is near-perfect on the family it was trained on (homoglyph+leet, 1.00) but **generalises poorly** to a held-out family (full-width, 0.78) — it memorised specific look-alike characters, not the idea of obfuscation. **Normalisation** is the mirror image: it recovers the held-out family (0.99, via generic Unicode folding) but is weaker against the adaptive attacker. Combining them is best across every family (≥ 0.98). The lesson (in the spirit of Goodfellow et al., 2015) is that adversarial training belongs *alongside* a principled input defence, not instead of one.

### 4.6c The arms race — a principled Unicode canonicaliser + a certificate

§4.6a left a wound: the importance-guided attacker drops the defended union model to 0.88, because the hand-curated table misses the look-alikes it uses. The fix is to fold every character through the **full Unicode confusables set (UTS #39)** — the data browsers use to detect spoofed domains (`normalize_skeleton`). Re-running the adaptive attacker vs three defences:

| Attacker | none | old table | **skeleton (UTS-39)** |
|---|---|---|---|
| uniform-adaptive (word/union) | 0.885 / 0.983 | 0.960 / 0.986 | **0.993 / 0.998** |
| targeted-adaptive k=20 | 0.676 / 0.880 | 0.673 / 0.879 | **0.993 / 0.998** |
| targeted-adaptive k=all | 0.633 / 0.865 | 0.624 / 0.867 | **0.993 / 0.998** |

The old table is useless against the *targeted* attacker; the skeleton canonicaliser closes the gap completely — the "unmapped" glyphs *are* in the official set. It costs no more than the old table (recall unchanged, same tiny FPR; it only remaps non-ASCII). It also licenses a **certificate**: under worst-case whole-family substitution, `normalize(attacked) == normalize(clean)` for **100%** of phishing emails (vs 0% for the old table) — a deterministic guarantee that the prediction *cannot* change under the homoglyph+adaptive family. The caveat: the certificate covers the character-confusable family only; a rational attacker must now move to a different attack *class* (word insertion, spacing, paraphrase). The arms race ends by raising the cost of attack, not by a silver bullet.

### 4.6d Iterative adversarial training

The model-side complement: train → generate *targeted* adversarial phishing with the current model's weights → retrain, re-targeting the test attack each round. Recall vs the re-targeted importance-guided attacker: round 0 = **0.44** → round 1 = **0.996** → round 3 = **0.999**; the original one-shot *uniform* adv-training reaches only **0.975**. Training on the attacker's actual high-value edits is far more sample-efficient than scattering perturbations. Two independent, complementary ways to shut the character-obfuscation front — one at the input, one in the model.

### 4.7 Explainability and its tension with robustness
Because the model is linear over TF-IDF, a token's contribution is exactly *coefficient × TF-IDF*, and these sum (with the bias) to the log-odds. We use this at two levels: a **global** / *model-level* view (the top-weighted tokens overall) and a **local** / *instance-level* view (the *exact* per-token contributions for a single email — a faithful explanation, not a keyword heuristic and not a LIME/SHAP approximation). A linear model is **inherently interpretable**, so we get both levels directly and never need a model-agnostic surrogate like LIME/SHAP — which would only approximate the model and is known to degrade under the feature redundancy documented in §4.2b. For the most confident phishing email the decision is carried by the URL-token and brand/urgency cues; a legitimate email is carried by ordinary conversational and technical words. The caveat (RQ4): this legibility lives entirely in the **word** model, because character n-grams are not human-readable. The features that bought robustness are the ones we cannot easily explain — robustness and interpretability pull in opposite directions.

### 4.8 DistilBERT and attack transferability
We fine-tuned DistilBERT (CPU-only, one epoch on a bounded 3,000-email subset) as a transformer baseline, and **[P5]** evaluated it on the **full test set (5,327 emails)**, removing the earlier small-subset caveat. It reached **F1 0.984 (MCC 0.975)** — *below* the TF-IDF union model's 0.998: on keyword-driven phishing text the lexical signal is already strong, so contextual modelling buys little. The more interesting result is **transferability**: DistilBERT's recall on phishing did *not* fall under the homoglyph+leet attack (it stayed at **0.999**). Its sub-word (WordPiece) tokeniser breaks obfuscated words into pieces much as our character n-grams do, so an attack crafted against a lexical model transfers poorly to the transformer. The transformer's value here is therefore its built-in sub-word robustness and its role as an upgrade path for messier, order-dependent text.

### 4.9 Cross-corpus generalisation

| | Accuracy | Precision | Recall | F1 | MCC | AUC |
|---|---|---|---|---|---|---|
| In-corpus (CEAS test) | 0.9985 | 0.9979 | 0.9979 | 0.9979 | 0.9967 | 1.000 |
| Cross-corpus (OOD) | 0.586 | 0.773 | 0.243 | 0.369 | 0.236 | 0.784 |

Trained on CEAS_08 and tested on Nazario + Nigerian-Fraud (phishing) and Enron + Ling (ham), F1 falls **0.998 → 0.37**, driven by a recall collapse to 0.24. A detector tuned to one corpus's vocabulary misses most phishing written in another style — the honest real-world number.

### 4.9b Diagnosing the collapse

The bare F1 hides *why* the model fails. It is not confidently wrong: the mean score on OOD phishing is only **0.33** (vs 0.98 in-corpus), so at the fixed 0.5 threshold it quietly relabels most OOD phishing as legitimate — a **prior/threshold shift**, not a loss of signal. Two facts confirm it: OOD **AUC is still 0.78** (ranking remains informative), and lowering the threshold recovers most of the recall:

| threshold | recall | precision | F1 |
|---|---|---|---|
| 0.50 | 0.243 | 0.773 | 0.369 |
| 0.20 | 0.649 | 0.760 | 0.700 |
| 0.10 | 0.821 | 0.691 | **0.750** |
| 0.05 | 0.911 | 0.635 | 0.748 |

But it does not fully recover (best F1 0.75 vs 0.998 in-corpus), because of a genuine **vocabulary gap**: the unigram out-of-vocabulary rate on OOD phishing is 0.20, ~3× the in-corpus 0.07. So the 0.37 headline overstates the failure (much of it is a fixable operating point), yet a real domain gap remains — exactly what a deployed detector must be re-calibrated for.

### 4.9c Calibration for fusion

Because this detector's probability is meant to be *combined* with the partner detector, it should mean what it says. In-corpus the union model is already well calibrated (Brier 0.0020, ECE 0.009; the reliability curve tracks the diagonal), because its scores are sharply bimodal; an isotonic re-fit on validation still improves both (Brier 0.0014, ECE 0.0025). The caveat is §4.9b: under distribution shift the scores collapse toward the boundary, so a fusion stage should prefer rank/AUC-based combination or re-calibrated inputs over naive probability multiplication.

### 4.9d Modern phishing (2023–2025): temporal drift

Everything above trains and tests on 2008 mail. Does a 2008-trained detector still catch phishing written 15 years later? Using **1,303 real phishing emails from 2023–2025** (Jose Nazario's live archive, parsed by `scripts/parse_nazario_recent.py`), the CEAS-trained union model catches only:

| Year | recall@0.5 | n |
|---|---|---|
| 2023 | 0.153 | 419 |
| 2024 | 0.171 | 403 |
| 2025 | 0.108 | 481 |
| all 2023–25 | **0.142** | 1303 |

Worse than the cross-corpus number, no better on the most recent year. Same operating-point caveat as §4.9b (the fixed 0.5 threshold overstates it — lowering it recovers some recall), but the conclusion holds: a model frozen on old mail is largely blind to current phishing. This is the genuine **concept-drift** result the single-year CEAS corpus couldn't provide, and the strongest argument that a lab number has a shelf life.

### 4.9e Do real attackers actually obfuscate?

The robustness study assumes character obfuscation is a real technique. Measuring obfuscation signatures in real phishing (modern 2023–25 vs CEAS 2008):

| Signal | CEAS-2008 phishing | **modern 2023–25** | CEAS-2008 legit |
|---|---|---|---|
| non-ASCII frac | 0.0009 | 0.0061 | 0.0005 |
| Cyr/Grk/Arm letter % | 0.3% | 1.8% | 0.2% |
| **mixed-script token % (homoglyph)** | **0.0%** | **1.8%** | 0.1% |
| **zero-width char %** | **0.0%** | **4.0%** | 0.0% |

The obfuscations we attack with are **not hypothetical**: mixed-script and zero-width tricks are essentially absent in 2008 phishing but appear in a small, growing share of modern phishing. Modest in absolute terms — real phishing isn't *mostly* homoglyphs — but the technique now exists in the wild where it didn't before, and a homoglyph swap costs the attacker almost nothing. This is the empirical justification for the whole robustness study, and it closes the loop: we defend (§4.6c–d, with a certificate) against a real, emerging behaviour.

### 4.10 The next attack class — hacking the words, not the letters

We closed the *letter*-trick front with a certificate, so a smart attacker changes **words** instead. Three attacks on the union model (recall on phishing):

| budget | synonym | delete | dilution |
|---|---|---|---|
| 5 | 0.998 | 0.959 | 0.970 |
| 10 | 0.998 | 0.912 | 0.638 |
| 20 | 0.998 | 0.809 | 0.284 |
| 40 | 0.998 | 0.781 | **0.129** |

**Synonyms barely work** (the model's redundant word + letter-chunk features absorb small swaps). **Deletion** hurts moderately. **Dilution is the killer**: burying the scam in legit-sounding filler drops recall to **0.13**, because a linear word-counting model just *adds up* signals. The **letter-defence does nothing** here (0.998 → 0.997) — wrong defence for the wrong attack. **The fix:** adversarial training on the word attacks restores dilution recall **0.28 → 1.00** with no clean-mail loss. Deeper lesson: bag-of-words is inherently vulnerable to dilution; the fundamental answers are structure/length-aware features (padded scams *look* padded) and fusion.

### 4.11 Evolving the detector on modern e-mail

A 2008 model catches only ~14% of modern scams (§4.9d). Fix: feed it modern examples.

**Experiment A (clean, no new confound):** add modern scams to training, leave the normal-mail source unchanged, test on held-out modern scams:

| model | recall on modern scams | recall on 2008 scams | false-alarm on modern normal (FPR) |
|---|---|---|---|
| baseline (2008) | 0.147 | 0.998 | 0.001 |
| **evolved (+ modern scams)** | **0.980** | 0.998 | 0.007 |

Recall on modern scams jumps 0.15 → 0.98, with **no forgetting** and **no rise in false alarms**. A detector isn't finished — kept current, it works.

**Experiment B (full modern detector, confound disclosed):** modern scams + modern normal (Apache mailing lists) scores F1 **0.996** — but a source-check shows the two sources are **99% separable by provenance alone**, the exact "cheating trap" the project controls for. So Experiment A is the trustworthy number; B is provenance-inflated. Same founding principle, applied to modern data.

### 4.12 A principled defence against dilution (P1)

Adversarial training removes dilution but must know the attack. A defence that doesn't: score the **positive evidence present** — the summed phishing weights of words that appear, using *binary presence* (not TF-IDF) so filler can't lower it — and flag if the union model fires **or** the evidence clears a floor set on *clean* validation ham.

| operating point | clean recall | diluted-phishing recall | ham FPR |
|---|---|---|---|
| union-only (baseline) | 0.998 | 0.129 | 0.001 |
| + floor @ 1% budget | 0.998 | 0.357 | 0.012 |
| + floor @ 3% budget | 0.998 | 0.496 | 0.030 |
| + floor @ 5% budget | 0.998 | 0.555 | 0.051 |
| + floor @ 10% budget | 0.999 | **0.663** | 0.103 |

Without knowing the attack, the floor lifts diluted-phishing recall from 0.13 to 0.36–0.66 as you widen the false-alarm budget. It doesn't fully solve dilution (short scams carry too little evidence) — the honest point: **dilution is a structural weakness of bag-of-words**; a rule mitigates it, adversarial training removes it (given knowledge), and the durable fix is a structure-aware model or fusion.

### 4.12b From mitigation to guarantee: a monotone model that cannot be diluted (P7)

The evidence-floor above *mitigates* dilution but does not close it, and it is a post-hoc rule: it clips a trained model's weights and sets a threshold by hand. Sharper question: can we **train** a model that is provably immune to dilution — and what does that cost?

**Construction.** Constrain every weight to be non-negative (`w_j ≥ 0`) and use **binary word-presence** features, so the score is `s(x) = b + Σ_j w_j · 1[word_j present]`. Inserting words can only flip indicators 0→1, and every weight is non-negative, so **the score is monotonically non-decreasing under insertion**. A flagged email therefore *cannot* be un-flagged by adding text: dilution is defeated **by construction**. This is the second certificate in this work, covering a different attack class from the UTS-39 certificate of §4.6c — that one bounds character *substitution*, this one bounds word *insertion*. The device comes from Fleshman et al. (2018) for malware; the attack it defeats is the classical **good-word attack** of Lowd & Meek (2005), of which our dilution attack is an instance (countered heuristically by Jorgensen et al. 2008). Trained with L-BFGS-B under box constraints.

**The guarantee requires order-free features — this is not a technicality.** With word bigrams, inserting a word *between* two words destroys their pair (`verify account` → `verify now account`): the indicator falls 1→0 and, since `w ≥ 0`, the score *decreases*. Measured on a bigram variant, the score fell on **388 of 500** phishing emails and the guarantee failed outright. Only order-free bag-of-words is genuinely monotone, so the model is deliberately unigram-only — itself part of the accuracy cost.

**The certificate holds exactly:**

| Padding applied | scores decreased | % clean-flagged still flagged |
|---|---|---|
| +40 words (untruncated) | **0** | **100.0** |
| +100 words (untruncated) | **0** | **100.0** |
| +300 words (untruncated) | **0** | **100.0** |
| +40 words (MAX_CHARS cap) | 9 | 99.8 |
| +100 words (MAX_CHARS cap) | 21 | 99.6 |

Zero decreases, exactly as the proof requires. The one leak is our own preprocessing: truncating at `MAX_CHARS=2000` lets heavy padding push genuine evidence past the cut-off. We report that rather than evaluating without the cap.

**And the cost is real:**

| Model | F1 | MCC | recall | precision |
|---|---|---|---|---|
| union LR (baseline) | **0.9979** | **0.9967** | 0.9979 | 0.9979 |
| monotone unigram (w ≥ 0) | 0.9115 | 0.8689 | 0.8734 | 0.9532 |

Recall on phishing under dilution (words inserted):

| Model | +0 | +10 | +20 | +40 |
|---|---|---|---|---|
| union LR | 0.998 | 0.632 | 0.365 | **0.129** |
| monotone (w ≥ 0) | 0.869 | 0.869 | 0.869 | **0.866** |
| union **or** monotone | 0.998 | — | 0.869 | **0.866** |

Clean F1 falls 0.998 → 0.912 (MCC 0.997 → 0.869), driven by recall. Expected: a monotone model may only *accumulate* evidence of guilt — it can never use "this word suggests legitimate" to acquit a borderline email — and it has surrendered word order too. It is **not** a replacement for the union detector and we do not present it as one.

**The useful reading is the combination.** As a one-sided safety net (flag if *either* fires), the pair keeps the union model's clean performance while putting a **certified floor** under dilution: 0.866 instead of 0.129, bought with false alarms on clean mail rising 0.0012 → 0.0225. Whether that trade is acceptable is an operational decision, not a mathematical one.

**What it does not solve.** The guarantee covers *insertion* only. An attacker who **deletes** the phishing wording or rewrites it semantically still evades this model — and a guilt-only model is *especially* exposed to deletion, having no negative evidence to fall back on. Two attack classes are now closed by proof (character confusables §4.6c; word insertion, here), and the adversary is pushed toward semantic rewriting, which is materially more expensive than padding but not impossible.

### 4.13 A harder word attack: paraphrase (P2)

Does a meaning-preserving rewrite evade the model? A rule-based paraphrase (synonym-swap every known word + shuffle sentences) leaves union recall at **0.998** — unchanged. The model's redundant word + character features absorb it, so paraphrase is **not** an easy evasion; dilution remains the one effective word attack.

A small **proof-of-concept probe of the stronger, neural test** (notebook §32e) uses a **three-stage pipeline**: **(1) attack** — an LLM rewrites each phishing email, keeping the scam intent but changing the wording; **(2) validate** — a *second, independent* LLM judges whether each rewrite is still phishing, and only accepted rewrites are scored (so a rewrite that quietly stopped being a scam cannot be miscounted as an evasion); **(3) detect** — our detectors score the validated attacks (24 valid of 30). Result — the union model's recall stays at **1.0**, so even a genuine neural rewrite does not evade it; the **monotone** model of §4.12b, by contrast, gives a little ground (0.75 → 0.71), because rewording removes the positive evidence it exclusively relies on — the honest counterweight to its certificate. Caveat: small sample, and attacker and judge share a model family, so this is indicative, not definitive; a full-scale study remains future work. The frozen rewrites/verdicts are committed in `data/llm_paraphrase/` (LLM output is not seed-reproducible, so it is generated once and read back).

### 4.14 Do the findings hold on a second corpus? (P6)

Replaying the core story on **SpamAssassin** (different source/era) reproduces the same shape:

| setting | word | union |
|---|---|---|
| clean | 0.959 | 0.973 |
| homoglyph+leet | 0.340 | 0.497 |
| + normalise | 0.918 | 0.965 |
| + skeleton | 0.918 | 0.965 |

High clean recall → roughly halved under the imperceptible attack → near-full recovery after cleaning. Numbers a little lower than CEAS, but the **conclusion is corpus-independent** — which makes it a finding, not a quirk.

### 4.15 How clean can the modern detector be? (P3)

Can we remove the modern-retrain confound by stripping mailing-list boilerplate (quotes, headers, list tags, footers) from the ham? **No** — source-separability stays at **0.99 → 0.99**. The gap is **genre** (dev discussion vs scam), not formatting, so it can't be scrubbed. A fully clean modern detector needs personal-style modern ham (not public); this is exactly why the trustworthy modern result is Experiment A (§4.11), which never introduces the confound.

---

## 4b. Error Analysis

On the clean test set (5,327 emails) the union model makes only **4 false positives** (legit flagged) and **4 false negatives** (phishing missed) — FP/FN rates around 0.1–0.2%. The errors are borderline, atypical emails, not a systematic blind spot (correct vs error median length ≈ 780 vs 680 chars). The false negatives are unusual phishing (a foreign-language solicitation, a terse "5 Euros" travel scam) whose vocabulary barely overlaps the training phishing; the false positives are short legit notes whose brevity + a stray URL token nudge them over the line — matching the correlation finding that *short* messages lean phishing.

**The cost asymmetry is the point.** A false positive quarantines a legit email (analyst time, alert fatigue); a false negative delivers phishing to the inbox (one click from compromise). That is why we weight recall (F2) and pick a cost-optimal threshold. Under attack the balance shifts: character obfuscation *manufactures* false negatives — phishing caught on clean text now evades the word-sensitive model — which is exactly what the normalisation and adversarial-training defences exist to contain.

---

## 5. Adaptive Deployment and Human-in-the-Loop Learning

Everything above evaluates the detector as a fixed artefact scored on a static corpus. A deployed filter is neither: it sits in a mail path with measurable cost, and next to *people* who can tell it when it is wrong. This section reports one experiment on that human signal, the measured cost of running the model, and states plainly which parts remain untested design.

### 5.1 Learning from user reports

Targeted phishing is rarely sent to one recipient — the same message goes to many employees with small variations. If one user presses "report phishing", can the system catch the rest of that campaign? This matters because the 2008-trained detector recognises only **14%** of modern phishing (§4.9d), so any cheap source of current labels is valuable.

**Do campaigns exist in the data?** We check rather than assume. Clustering the 1,303 modern (2023–25) phishing emails yields **93 campaigns** of three or more near-duplicates, mean intra-campaign similarity **0.883**. We use *average* linkage deliberately: single-link/connected-components clustering **chains** (A~B, B~C, but A≁C) and produced a 139-member "campaign" whose members averaged only 0.50 similarity — a blob, not a campaign.

**Method.** One member per campaign is revealed as "reported"; we score only the *remaining* members. Two mechanisms: **(A) similarity propagation** — flag an email whose cosine similarity to a reported one exceeds τ, no retraining — and **(B) one-shot retraining** on the reported emails. Clustering campaigns to triage phishing is established practice (Althobaiti et al., 2023); what we add is treating it as a feedback loop and then attacking it.

**Where τ comes from, and two calibrations that failed.** Our first two attempts set τ from *legitimate* mail to hit a false-alarm budget. Both returned recall ≈ 1.0 **and** a broken control: emails with *no* near-duplicates were "detected" at 0.92–0.94, which is impossible if the mechanism works by matching campaigns. The cause is the genre confound of §4.15 — a threshold fitted on ham separates "scam-like" from "mailing-list", which says nothing about campaign membership. We therefore set τ to the 99th percentile of *different-campaign* similarity and **report** the ham false-positive rate at that τ rather than using ham to choose it. Campaigns are additionally split into calibration and evaluation halves so τ never sees an email it later scores, and the vectoriser is fit on 2008 training data only.

| | recall | FPR (CEAS ham) | FPR (modern ham) |
|---|---|---|---|
| baseline (2008 model) | 0.149 | 0.0012 | 0.0017 |
| **A: campaign-matched** | **1.000** | 0.0012 | 0.0017 |
| A: campaign-blind (ablation) | 0.329 | — | — |
| B: one-shot retrain | 0.991 | — | — |
| *control*: singletons, similarity only | 0.148 | — | — |

**The control decides it.** Recall rises 0.149 → 1.000 with no increase in false positives — at a campaign-specific τ, legitimate mail simply does not resemble a particular reported scam. But recall alone proves little, since *any* modern example helps a 2008 model. The decisive figure is the **campaign-blind** ablation: remove that campaign's own report, keep all others, and recall falls to **0.329**. Identical threshold, identical reports, one difference — so the gap between 1.000 and 0.329 is the genuine campaign effect, while the remainder (0.329 vs 0.149) is simply the value of modern data. Quoting the headline jump alone would overstate the mechanism.

**Attacking the feedback loop** (fraction of campaign emails escaping propagation):

| attack | escaping |
|---|---|
| synonym (40) | 0.000 |
| dilution +40 / +80 words | 0.149 / 0.309 |
| **homoglyph + leet** | **1.000** |
| **+ UTS-39 skeleton** | **0.034** |

Character obfuscation defeats the mechanism **completely** — every campaign email escapes, because homoglyphs destroy the shared tokens the similarity is computed from. The canonicaliser of §4.6c restores it. The certified normaliser is therefore not merely a classifier defence but a **precondition** for human-in-the-loop detection: a deployment that added a report button without canonicalising its input would ship a safety feature an attacker can disable with invisible characters.

**The report button as an attack surface.** False reports — a malicious insider, or mistaken users — raise the false-positive rate on legitimate mail from 0.0012 to **0.0055** at a 10% false-report rate and **0.0115** at 50% (mean of 10 seeds). The damage is bounded, since a bad exemplar only matches mail resembling it, but the direction is unambiguous: an unvetted feedback loop is a denial-of-service channel. This is the causative counterpart to the evasion attacks studied earlier, and the standing objection to the retraining remedy Lowd & Meek (2005) proposed for good-word attacks.

**What this is not.** We simulate an organisation using a public archive; campaigns are *our* clustering, not ground-truth labels; reports are simulated and, outside the poisoning arm, always correct. This is evidence that the mechanism works and that its failure modes are specific and measurable — not a measurement of a real deployment.

### 5.2 Deployment envelope

Where such a detector can run is decided by three numbers, so we measured them rather than speculating:

| Quantity | Measured |
|---|---|
| single-email latency, median | ≈ 3 ms |
| single-email latency, p95 | ≈ 6 ms |
| batch throughput | several hundred emails/sec/core |
| model + vectoriser on disk | ≈ 3 MB |
| full retrain on 16k emails | seconds |

A few milliseconds of latency and a couple of megabytes on disk put this model comfortably within reach of **client-side** execution — inside a desktop or mobile mail client — rather than requiring a server. That is a privacy property as much as an engineering one: if scoring happens on the device, message content never leaves it. Hundreds of emails per second per core means a single modest machine covers a mid-sized organisation's mail flow, and scaling is horizontal.

The retrain figure speaks directly to §4.11: refreshing the model costs seconds, so *how often* to retrain is a policy question, not a compute constraint — the binding resource is fresh labelled mail, which is precisely what §5.1 supplies.

A plausible integration: canonicalise every incoming message (§4.6c), score it with the content detector, combine that score with the partner URL/sender detector, and route on the fused decision — with user reports feeding back into both the similarity index and the periodic retrain, subject to the poisoning caveat above.

**What is not measured.** These are microbenchmarks on already-extracted text. A real deployment must parse MIME, strip HTML, handle attachments, encodings and threading, and cope with non-English mail — none of which is evaluated here. The URL-stripping design (§3) also means this detector is by construction only half of a deployed system. This subsection is a **costed design proposal, not a deployment result**.

### 5.3 Continual learning under drift (future work)

The natural extension is a detector that retrains continuously as the phishing distribution moves. We stop short of claiming this as a contribution, because two parts of it are already reported above and the remainder is not honestly measurable with the data available.

**What is already done.** §4.11 performs one retraining step and recovers recall on modern phishing from 0.147 to 0.980 without forgetting; §5.1 performs a second, from a single reported example per campaign; and the poisoning arm quantifies the security cost of learning from an untrusted stream. Retraining is also cheap (§5.2).

**What a proper study would require.** A cadence experiment — train on 2023, evaluate on 2024 and 2025, and repeat — needs a time-ordered stream of *both* classes. Our modern phishing is time-stamped, but the only modern legitimate mail available to us is public mailing-list traffic carrying the provenance confound of §4.15, so any false-alarm-over-time curve built on it would measure genre rather than drift. We could report only a recall curve, which is half the answer, and we prefer to state the gap rather than publish the half.

**How it would be built.** Given a time-ordered corpus of both classes: fix a deployment date, retrain on a sliding or expanding window, and measure recall and false-alarm rate as a function of elapsed time and retraining interval, with a drift detector triggering event-driven refreshes between scheduled ones. The security requirement is that any such loop be treated as untrusted input — report vetting, per-reporter reputation, quarantine before promotion to training data — following the poisoning result of §5.1.

---

## 6. Discussion

**What worked.** On clean single-corpus data a linear TF-IDF model is excellent, interpretable and cheap. The robustness experiment produced a clear, honest narrative: word features are fragile to imperceptible obfuscation; char n-grams and normalisation repair most of the damage; a naive adaptive attacker finds a gap and an *importance-guided* one widens it; and a principled Unicode canonicaliser plus iterative targeted adversarial training (§4.6c–4.6d) then close that gap with a certified guarantee — so the arms race resolves not into a solved problem but into a raised cost of attack, pushing the adversary toward a fundamentally different attack class.

**Mitigation versus guarantee.** A theme emerges from the two certified defences. Most of our defences are *empirical*: they work against the attacks we ran, and their ceiling is found by attacking them harder. Two are *structural* — the UTS-39 canonicaliser (§4.6c) and the monotone model (§4.12b) — and these have no ceiling to discover, because the attack class is excluded by construction. The price differs sharply, which is the interesting part: the Unicode certificate is nearly free (recall unchanged, same tiny FPR), whereas the monotonicity certificate costs about nine points of clean F1, because forbidding negative evidence removes information the detector was genuinely using. A guarantee is not automatically worth having; it is worth having when its price is one you can afford — which is why the monotone model is reported as a safety net beside the main detector, not as a replacement.

**Defences are complementary.** Adversarial training and normalisation fail in different ways — the former is strong only on the family it trained on, the latter generalises across families but is weaker against a defence-aware attacker. Neither is complete alone; stacked, they are robust across every family we tested. A learned defence and a principled input defence are best combined, not chosen between.

**Why classical models compete with BERT here.** The signal is dominated by *which words appear* ("verify", "account", "suspended"), which bag-of-words captures directly, so contextual modelling buys little. BERT should pull ahead on order-dependent text ("good, not bad" vs "bad, not good"), which phishing keywords rarely require.

**The role of the base rate.** The attack only *looks* dangerous under a realistic legit-majority prior. On the raw phishing-majority corpus the same attack barely moves recall, because a model that loses its features falls back to the majority class — which happens to be "phishing". A base-rate artefact, and a reminder that robustness numbers are meaningless without a realistic prior.

**Limitations.** (1) CEAS_08 is from 2008; vocabulary and tactics drift, and both the cross-corpus collapse (§4.9) and the direct temporal-drift test on 2023–2025 phishing (§4.9d, recall 0.14) show a single-corpus model does not transfer in space or time — retraining on current mail is a requirement, not a nicety. (2) We upgraded the rule-based normaliser to the full Unicode confusables set (§4.6c), which closes the character-obfuscation front with a certificate; we then probed the next attack class (word-level dilution, §4.10), patched it with adversarial training, and finally closed it by construction with the monotone model (§4.12b) — though only for *insertion*, and at a substantial accuracy cost, so a structure-aware model that resists dilution *without* surrendering negative evidence remains future work, as does a truly open-ended semantic/paraphrase attacker (to which the monotone model, having no negative evidence, would be especially exposed). (3) DistilBERT is now evaluated on the full test set but still *trained* on a CPU-bounded subset, so its comparison is indicative, not a statement about the transformer's ceiling (a GPU full-data run is future work). (4) We attacked only phishing emails; a full study would also consider attacks that make legitimate mail look like phishing. (5) The modern-phishing set is phishing-only (public modern *ham* is scarce), so its numbers are recall at a fixed operating point rather than a full re-evaluation.

**Cybersecurity relevance.** The project touches: evasion attacks, perturbations, homoglyph/leet obfuscation, white/black-box and adaptive threat models, transferability, robustness, normalisation defence; the FP/FN cost trade-off and accuracy paradox; and TF-IDF, Naive Bayes with smoothing, Complement NB for imbalance, explainability, and the transformer upgrade path.

---

## 7. Conclusion

We built a content-only phishing detector and used it to study robustness honestly. On clean single-corpus data the model is near-perfect and explainable. Under imperceptible homoglyph obfuscation a word-only model loses a third of its recall; character n-grams and a Unicode-normalisation defence restore it, and the defence even generalises to an unseen full-width attack. A naive adaptive attacker keeps a residual gap and an importance-guided one widens it — but upgrading to the full Unicode confusables set closes it with a *certified* invariance guarantee, and iterative targeted adversarial training does the same from the model side, so the arms race ends by raising the cost of attack rather than by any single fix. A cross-corpus test, and a direct test on real 2023–2025 phishing (caught only 14% of the time), show the laboratory numbers survive neither a change of corpus nor the passage of time — while that same modern phishing confirms the obfuscations we defend against are a real, emerging technique. Pushing the arms race one class further, a word-level *dilution* attack still evades the model and the character defence cannot help, but adversarial training on word edits repairs it; and *evolving* the detector on modern phishing lifts its recall on 2023–2025 scams from 0.15 to 0.98 — a detector kept current works, a frozen one does not. A further round of hardening and probing (§4.12–4.15) adds nuance: a principled, attack-agnostic evidence-floor *partially* defends dilution; a paraphrase attack does not evade the model; the collapse-and-recover finding replicates on a second corpus; and the modern-retrain confound is shown to be genre-level rather than fixable by cleaning. Finally, that partial mitigation becomes a guarantee: constraining the model to non-negative weights over binary word-presence makes its score provably non-decreasing under insertion, so no quantity of filler can un-flag a phishing email (§4.12b) — verified exactly, and bought at a real cost of nine points of clean F1, which is why it belongs beside the main detector rather than in place of it. Two attack classes are now closed by proof rather than by evidence, and the adversary is left with the costlier options of deleting or rewriting the scam itself. Stepping outside the static corpus, a single simulated user report per campaign recovers most of what temporal drift takes away, at no measurable cost in false alarms — but only once the input is canonicalised, since character obfuscation otherwise disables the mechanism outright, and only if the report channel itself is defended, since false reports degrade it. The recurring lesson is that every component we add, including the ones meant to help, becomes something an attacker can aim at.

The robust character features remain the least interpretable ones, so robustness and explainability continue to pull against each other. Together these results argue for exactly the design this project is one half of: combine a lexical content detector with an independent non-text signal, so that when an attacker defeats one channel the other still fires. Future work: a structure-aware model that defends dilution outright, a neural paraphrase attacker, a personal-style modern ham source for a fully clean modern retrain, a GPU full-data DistilBERT run, and the fusion itself. Two more follow from the attack taxonomy: an **adversarial-input detector** that flags obfuscated mail as anomalous (attractive because §4.9e shows obfuscation signatures are measurably rare in ordinary mail), and hardening against **model-extraction** for a query-accessible deployment.

---

## Executive Summary

**Problem.** Phishing email is a leading initial-access vector, and much of the deception is linguistic, so we build the *content* half of a two-part detector: read only subject + body and decide phishing vs legitimate, with explainability and — the main focus — robustness to text-obfuscation evasion.

**Data.** CEAS_08 (from the Kaggle phishing compilation) — the one large corpus with both classes from a single source, so a model cannot cheat on provenance (a corpus-ID probe confirms ~99% separability). URLs stripped (the partner's job); rebalanced to a realistic legit-majority base rate.

**Methodology.** Classical NLP pipeline: TF-IDF (word + char n-grams) → Naive Bayes / Logistic Regression, tuned on validation, evaluated once on held-out test with bootstrap CIs + McNemar. Then homoglyph/leet attacks defended by normalisation and adversarial training, evaluated against *seen*, *held-out* and *adaptive* attacks. DistilBERT as a transformer baseline.

**Key findings.** Clean, in-corpus: near-perfect (F1 0.998, MCC 0.997) and interpretable, with a handful of borderline errors. A word-only model is fragile — an *imperceptible* homoglyph attack drops recall 0.99 → 0.71. Char n-grams + normalisation restore it; normalisation even generalises to an unseen full-width attack, but an adaptive attacker keeps a residual gap. Adversarial training is complementary (strong on the trained family, weak on unseen), so the two defences stack best. DistilBERT ties/slightly trails TF-IDF and the attack does not transfer to it. Cross-corpus F1 falls to ~0.37 — but we diagnose this as a recoverable threshold shift (OOD AUC 0.78; re-thresholding recovers F1 to 0.75) plus a real vocabulary gap (OOV 7% → 20%). A stronger *importance-guided* adaptive attacker lowers the defended ceiling to union 0.87 (word 0.67), and robustness numbers are stable across 10 attack seeds. **A principled UTS #39 canonicaliser then closes that gap (recall back to 0.998, 100% certified-invariant), and iterative targeted adversarial training restores recall 0.44 → 0.999.** On **real 2023–2025 phishing** the 2008-trained model catches only 14% (concept drift), and homoglyph/zero-width obfuscation — absent in 2008 — now appears in a small but growing share, empirically motivating the whole robustness study. Beyond letters, a word-level **dilution** attack still evades the model (recall → 0.13; the letter-defence can't help), but adversarial training on word edits repairs it (→ 1.0). Finally, **evolving** the detector on modern scams lifts recall on 2023–2025 phishing from **0.15 to 0.98** with no forgetting and no extra false alarms — a detector kept current works. Further hardening: a principled evidence-floor partially defends dilution (0.13 → up to 0.66) without knowing the attack; a paraphrase attack does not evade; and the robustness findings replicate on a second corpus (SpamAssassin). Dilution is then closed by **proof** rather than patching: a monotone (non-negative-weight) model over binary word-presence cannot have its score lowered by inserted text, so padding provably cannot un-flag a phishing email — at a real cost of clean F1 0.998 → 0.912, making it a certified safety net alongside the main detector rather than a replacement.

**Relation to prior claims.** Two published claims tested on our data *held*: Complement NB's imbalance advantage (Rennie 2003) and the collapse-then-shield behaviour under visual attack (Eger 2019).

**Recommendation & conclusion.** A lexical content detector is an excellent, cheap, interpretable first line but brittle alone — to obfuscation and to distribution shift. Recommend the fused design (content + independent non-text signal) and pairing a principled input defence with adversarial training. Transfers well to related keyword-driven security text (spam, fraud) with the same caveats.

---

## References

Full BibTeX in [`references.bib`](references.bib). Key works: Althobaiti et al. (2023); Sahami et al. (1998); Rennie et al. (2003); Metsis et al. (2006); Fette et al. (2007); Cavnar & Trenkle (1994); Zhang et al. (2015); Goodfellow et al. (2015); Gao et al. (2018); Li et al. (2019); Ebrahimi et al. (2018); Eger et al. (2019); Pruthi et al. (2019); Boucher et al. (2022); Vaswani et al. (2017); Devlin et al. (2019); Sanh et al. (2019); Chicco & Jurman (2020); Salloum et al. (2022); Unicode UTS #39; Lowd & Meek (2005); Jorgensen et al. (2008); Fleshman et al. (2018).
