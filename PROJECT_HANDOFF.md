# PROJECT_HANDOFF.md — PhishFusion (Student A, Content/NLP Detector)

> **Read this file first.** It is the single, self-contained entry point for continuing this project
> in a new Claude Code conversation. Everything below reflects the **latest, verified state** (repo
> reviewed and re-run end-to-end on 2026-07-21; notebook executes with **0 errors**). Every number
> here is a **real result** produced by running the code — do not invent or "improve" numbers.

---

## 1. Project title and goal
**PhishFusion — Student A: a content-based (text-only) phishing-email detector**, built as an
academic course project (Data Science & Cyber). It reads only the email **subject + body**, decides
**phishing vs legitimate**, explains its decisions, and — the main focus — is **stress-tested for
adversarial robustness** against text-obfuscation evasion. It is one half of a two-part system: a
partner (Student B) builds a URL/sender detector; the two are later *fused* (fusion is **out of scope**
for Student A).

**The contribution is not a new model.** It is an honest, rigorous **robustness study** of a standard
model: show a near-perfect clean score is *fragile*, attack it the way a real adversary would, defend
it (one defence carries a mathematical certificate), attack it smarter, defend again, prove the threat
is real on modern data, and keep the model current. Where a defence is incomplete, that is stated
plainly — which is the core argument for fusing with a second, independent detector.

## 2. Academic / cybersecurity motivation (short)
Phishing is a top initial-access vector and is largely carried by **language** (urgency, authority,
calls to action), so NLP is a natural detector. But filters have been in a decades-long arms race with
attackers who mangle text (homoglyphs `vеrify`, leetspeak `cl1ck`) so a human still reads it but a
keyword filter fails. A detector that is excellent on clean mail but collapses on trivially obfuscated
mail is not useful — hence the emphasis on **robustness, explainability, and honest evaluation**.

## 3. Exact current status
- **Fully implemented, executed end-to-end, and written up.** Notebook = **148 cells, 0 errors**
  (§2c deeper EDA → report §4.2b; §32b → report §4.12b; §32c/§32d → report §5; §32e LLM-paraphrase → §4.13).
- **Report** = `report/main.pdf` (**29 pages**, compiles clean with Tectonic; source `report/main.tex`).
- **Results** = **45 tables** (`results/tables/T0..T40` + `T9b`/`T9c` + `T13b` + `data_hash.txt`) and **25
  figures** (`results/figures/F1..F24` + `F_imbalance`; **F13 is now used** = EDA box plots).
- Eight rounds of improvement layered on the original pipeline (v3–v6 phases, v7 monotone certificate,
  v8 human-in-the-loop + deployment, v9 LLM-paraphrase probe, v10 course grounding + deeper EDA — see
  the **v3–v10 changelogs in `docs/PROJECT_PLAN.md`** for the blow-by-blow with numbers).
- **Git repo initialised**, full project committed. **Presentation built** (a 25-slide self-contained
  HTML deck at `presentation/phishfusion_presentation.html`). Fusion with Student B still out of scope.

## 3c. Recent-sessions recap (what happened after the first commit)
The initial commit was the v6 state. Since then, in order:
- **v7 — monotone certificate (P7):** a *trained* non-negative unigram model → dilution provably cannot
  un-flag an e-mail (0 score decreases; 100% at +40/+100/+300 words). Honest cost: clean F1 0.998→0.912.
  Notebook §32b → report §4.12b. Tables T31–T34, fig F23.
- **v8 — human-in-the-loop + deployment (P8):** one user report catches the rest of a phishing campaign
  (0.149→1.0; campaign-blind ablation 0.33); character obfuscation disables it until UTS-39 is applied;
  false reports poison it. Plus a measured deployment envelope. Notebook §32c/§32d → report §5. T35–T39,
  F24. Added the "dynamic database" continual-learning idea (presentation only).
- **v9 — LLM paraphrase attack, LLM-judged (P9):** a 3-stage attack→validate→detect pipeline; union recall
  stays 1.0, the monotone model gives ground 0.75→0.71. Small probe (24/30 valid). Frozen LLM outputs
  committed under `data/llm_paraphrase/`. Notebook §32e → report §4.13. Table T40.
- **v10 — course grounding + deeper EDA:** reviewed the semester's lecture decks and connected the project
  to them (metrics/accuracy-paradox, stratified split + leakage + temporal rule, the attack taxonomy,
  normalisation-as-textbook-countermeasure, explainability levels). Deeper EDA: skewness/kurtosis (T9b),
  VIF redundancy (T9c), a Mann–Whitney test, and figure F13. Notebook §2c → report §4.2b.
- **Presentation:** the 25-slide animated deck (see `presentation/`).
- **New doc:** `docs/DECISIONS_AND_PITFALLS.md` — the bugs we caught, how, and the methodology rules.
- **Date-trap check (2026-07-21):** tested the CEAS date column for the Unix-epoch/imputed-zero trap
  (`scripts/check_date_trap.py`) — **clean** (0 of 34,342 are 1970/epoch; the ~0.3% off-2007-2009 are
  spoofed headers, already excluded by §2). Recorded in DECISIONS_AND_PITFALLS #15, IMPROVEMENTS_STORY
  Stage 12, and the notebook §2 temporal markdown.
- **Presentation grew to 31 slides:** added a motivation slide (social engineering), a datasets+cleaning
  slide, an EDA-findings slide, and three "grounded in the course" slides (pipeline map, methods/metrics,
  attack taxonomy incl. Taleb's turkey). Eyebrow numbers + counter are JS-auto, so inserts renumber safely.

## 3b. Environment & toolchain (Windows)
- **Project root:** `C:\GitHub\Data_Science\PhishFusion` (git repo, initialised 2026-07-19).
- **Python:** use the launcher **`py -3`** = Python 3.11 (`%LOCALAPPDATA%\Python\pythoncore-3.11-64`).
  It has numpy/pandas/scikit-learn/scipy, **torch (CPU-only, no CUDA)** and transformers; matplotlib,
  jupyter, nbconvert, ipykernel, pypdf are installed. (A second `python` = msys mingw has NO packages —
  don't use it.) matplotlib is 3.11: use `orientation=` / `tick_labels=` for boxplots (not `vert=`/`labels=`).
- **LaTeX:** no system LaTeX; the report PDF is compiled with the **Tectonic** single-binary engine.
  It was fetched to a session scratchpad; if absent, re-download from GitHub releases
  (`tectonic-*-x86_64-pc-windows-msvc.zip`) and run `tectonic main.tex` from `report/`.
- **Shell:** PowerShell primary; a Bash tool is also available. `cd` can reset — prefer absolute paths.
- **Reproduce:** `py -3 run_all.py` (full, incl. DistilBERT) or `PHISH_RUN_BERT=0 py -3 run_all.py`
  (classical only). **Measured runtime of the classical-only run: ~45 min** (the old "3–4 min" figure
  predates the phase-2/3/4 sections — targeted attacks, K=10 seed CIs, iterative adversarial training,
  SpamAssassin and evolve are the expensive parts). Budget over an hour for the full run.
  Classical results are **exactly** deterministic (seed 42): a re-run reproduced 31/32 tables and
  21/22 figures byte-identically. DistilBERT can vary slightly.
  Note: `PHISH_RUN_BERT=0` no longer overwrites `T7_distilbert.csv` (it used to clobber it with a
  "skipped" placeholder); the table is only written when the DistilBERT arm actually runs.

## 4. Repository structure (what each important file does)
```
PROJECT_HANDOFF.md        <- THIS FILE (entry point for a new chat)
README.md                 top-level overview + real headline results + how to run
requirements.txt          pinned deps (numpy/pandas/sklearn/scipy/matplotlib; torch+transformers optional)
run_all.py                re-executes the notebook via nbconvert (regenerates ALL tables/figures)
.gitignore                excludes raw data, caches, LaTeX build artifacts

notebooks/
  student_A_content_detector.ipynb   THE deliverable — runs top-to-bottom, 148 cells, 0 errors,
                                     numbered §0–§34 (42 headings incl. §2c/§13b/§15b/§32b–§32e);
                                     each section = markdown "what/why" + code + output +
                                     a "Analysis of the results" markdown cell.

src/                      reusable helpers imported by the notebook (edit here, not the .ipynb):
  config.py               all knobs: paths, seed=42, PHISH_FRACTION=0.35, MAX_CHARS=2000, n-gram ranges,
                          max_features=30000, BOOTSTRAP_N=1000, cost weights, BERT_N_TRAIN=3000.
  data.py                 load per-source CSVs, harmonise, STRIP URLs, clean/truncate, dedupe,
                          make_text(), sha256_of().
  normalize.py            the normalisation DEFENCE. normalize_text() = curated confusables table +
                          NFKD + de-leet (the "hand-made" baseline). normalize_skeleton() = the
                          PRINCIPLED defence built from the FULL Unicode UTS-39 confusables.txt.
  attacks.py              all attacks: homoglyph/leet/fullwidth/zerowidth/adaptive/mixed;
                          attack_importance (greedy word-importance); word_key/rank_words_by_weight/
                          perturb_words; WORD_SYNONYMS + attack_synonym/attack_wordmask/attack_insert
                          (dilution); attack_paraphrase; attack_corpus.
  stats.py                metrics() (acc/P/R/F1/F2/MCC/AUC), bootstrap_ci(), mcnemar(),
                          cost_optimal_threshold().
  explain.py              top_features(), explain_email(), instance_contributions() (faithful coef×tfidf).

scripts/                  DEV TOOLS (not the deliverable; regenerate the notebook):
  build_nb.py             SOURCE OF TRUTH for the notebook — assembles it from cell strings.
                          To change the notebook: edit here -> run it -> nbconvert -> add_analysis.py.
  add_analysis.py         inserts the 17 "Analysis of the results" markdown cells by anchor after a run.
  parse_nazario_recent.py mbox -> CSV for modern 2023-25 phishing.
  parse_modern_ham.py     mbox -> CSV for modern legitimate mail (Apache mailing lists).

report/
  main.tex                the academic report (Overleaf-ready). 29 pages.
  main.pdf                compiled report (Tectonic).
  report.md               readable Markdown MIRROR of the report (keep in sync with main.tex).
  references.bib          bibliography (all cited; + lowd2005good / jorgensen2008multiple /
                          fleshman2018nonnegative added for the P7 monotone certificate).

docs/
  PROJECT_PLAN.md         full design/plan: RQs, experiment matrix E0..E24, methodology, the PIPELINE
                          MAP (§12, folded in), §11 planned tasks (P1/P2/P3/P5/P6/P7/P8/P9 IMPLEMENTED;
                          P4 fusion out of scope), v2..v10 changelogs (the blow-by-blow dev record).
  IMPROVEMENTS_STORY.md   PLAIN-LANGUAGE "build -> hack -> fix -> evolve" narrative (Stages 0-12) =
                          the presentation speaker script.
  DECISIONS_AND_PITFALLS.md  the bugs we caught + HOW, the methodology rules, reproducibility rules.
                          (institutional memory — read to avoid repeating traps.)
  PROJECT_EXPLAINED.md    PLAIN-LANGUAGE full study guide + "likely lecturer questions & answers".
                          (The old docs/HANDOFF.md and docs/PIPELINE.md were folded into PROJECT_PLAN.md
                          and this file, then removed, to avoid duplicate/competing docs.)

presentation/
  phishfusion_presentation.html   the 25-slide talk (self-contained; open in a browser, press F).
  README.md                       how to present it + slide-by-slide outline.

data/
  raw/kaggle_phishing/    6 source CSVs (CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud, SpamAssasin).
                          NOTE: the merged phishing_email.csv was deleted (deliberately unused). gitignored.
  raw/nazario_recent/     nazario_recent.csv (1,303 modern 2023-25 phishing; COMMITTED) + raw .mbox (ignored).
  raw/modern_ham/         modern_ham.csv (3,000 modern legit emails, Apache lists; COMMITTED) + .mbox (ignored).
  llm_paraphrase/         sample/rewrites/judge.json — frozen LLM-paraphrase probe (§32e; COMMITTED, see its README).
  unicode/confusables.txt Unicode UTS-39 data for normalize_skeleton (COMMITTED).
  emails_working.csv      frozen 35%-phishing CEAS snapshot (regenerated by the notebook; gitignored; hashed).

results/tables/           T0..T40 (+T9b/T9c/T13b) + data_hash.txt  (see §8 for the map)
results/figures/          F1..F24 (+F_imbalance; F13 = EDA box plots)
artifacts/                preds_A_val/test/test_attacked.csv  (detector probabilities for the fusion step)
```

## 5. Final chosen pipeline & model (and why)
**CEAS_08 (single mixed-class corpus) → strip URLs → rebalance to 35% phishing → stratified 60/20/20
(seed 42) → TF-IDF {word 1–2 grams, char 3–5 grams `char_wb`, and their UNION} → Logistic Regression
(C=8) on the union features.**
- **CEAS_08 only** for the main experiments: it is the one large corpus with **both classes from one
  source**, which removes the corpus-confound *by construction* (a corpus-ID probe scores ~0.99, i.e.
  pooling different corpora would let a model cheat on provenance).
- **Strip URLs** → content-only (links are the partner's signal). **35% phishing** → realistic
  legit-majority prior that also makes the evasion test meaningful.
- **TF-IDF + Logistic Regression (union)** is the working model: near-perfect, fast, and **explainable**
  (a token's contribution is exactly `coef × tfidf`). Char n-grams are what give obfuscation robustness.
- **Model choice:** among MultinomialNB / ComplementNB / LogisticRegression, LogReg wins on validation
  F1 (0.994 vs ~0.97). DistilBERT is an optional transformer baseline (see §9), not the working model.

## 6. Dataset
- **Primary:** CEAS_08 (Kaggle "Phishing Email Dataset" compilation), ~39k emails, both classes, 2008.
  Working set after URL-strip + rebalance = **26,633 emails, 35% phishing** (SHA-256 in `data_hash.txt`).
- **OOD (cross-corpus test):** Nazario + Nigerian_Fraud (phishing), Enron + Ling (ham).
- **Second within-corpus testbed:** SpamAssassin (5,809 emails, 30% phishing) — used for the P6 replication.
- **Modern:** 1,303 real 2023-25 phishing (Jose Nazario live archive) + 3,000 modern legit (Apache
  mailing lists, 2024).

## 7. Preprocessing & feature extraction
- **Preprocessing:** concat subject+body; strip control chars; **replace URLs with a `URL` token**;
  truncate to `MAX_CHARS=2000`; dedupe; lowercase (in the vectorizer). Normalisation is applied only as
  a *defence arm* (kept separable for ablation), never silently baked into the base pipeline.
- **Features (TF-IDF, `sublinear_tf`):** word 1–2 grams (English stop-words, `min_df=3`); char_wb 3–5
  grams (`min_df=5`); each capped at 30k features; the **union** is the working representation.
- **Meta-features** (length, digit/upper/non-ASCII ratios, etc.) are **analysis-only**, not model inputs
  — EXCEPT the P1 dilution defence, which adds a positive-evidence structure feature as an explicit rule.

## 8. Final results & where they are stored
All tables in `results/tables/`, figures in `results/figures/`. Headline numbers (all REAL):

**Clean detection (T3):** union LogReg **F1 0.998, MCC 0.997, AUC 1.00**; MultinomialNB 0.972 /
ComplementNB 0.976. Baselines (T1): majority 0 recall; keyword F1 ~0.25.

**Robustness — recall on phishing (T5), word-only | union:** clean 0.99|1.00; homoglyph+leet
0.71|0.93; +normalise 0.99|1.00; full-width[held-out] 0.68|0.92; +normalise 0.99|1.00; adaptive
0.89|0.98; +normalise 0.96|0.99.

**Importance-guided (smart) attacker (T10–T12):** 8 targeted edits → word recall 0.74 (vs 0.98
random); defended ceiling **union 0.879 / word 0.673** (T11) — worse than the uniform attacker implied.

**Arms-race fix (T17/T18/T19):** the **UTS-39 skeleton canonicaliser** restores targeted-adaptive
recall to **0.993/0.998** (old hand-table did nothing: 0.67); **certificate = 100%** provably invariant
vs 0% (T18); **iterative targeted adversarial training** 0.44 → 0.999 (T19). Skeleton defence cost is
free (T20). Multi-seed CIs sd ≤ 0.004 (T14); calibration Brier 0.0020→0.0014 (T16).

**Cross-corpus (T6):** F1 **0.998 → 0.37** (recall 0.24, AUC still 0.78) — diagnosed as prior-shift +
vocab gap (T13/T13b: OOV 0.07→0.20; re-threshold → F1 0.75).

**Modern phishing (T21/T22):** 2008 model catches only **14%** of 2023-25 scams (drift); real homoglyph/
zero-width obfuscation ≈0% in 2008 → 1.8%/4.0% now.

**Word attacks + evolve (T23–T26):** synonym 0.998, delete → 0.78, **dilution → 0.13**; adversarial
training fixes dilution → **1.0** (T24). **Evolve:** adding modern scams lifts recall on modern scams
**0.147 → 0.98**, no forgetting (0.998), FPR 0.007 (T25). Modern mixed detector F1 0.996 but 99%
source-separable (T26 — provenance confound, disclosed).

**P9 LLM paraphrase attack, LLM-judged (T40) — small probe, in report §4.13.** The neural version of
the §4.13 paraphrase test, promised there as future work. An LLM rewrites 30 test-set phishing e-mails
(meaning-preserving); a *second, independent* LLM judges each rewrite still-phishing (24/30 accepted,
6 rewritten into non-scams and dropped — the validity filter). Scoring the 24 valid attacks: **union
recall stays 1.000** (a genuine neural rewrite does not evade it — strengthens §4.13), while the
**monotone model (§32b) drops 0.75 → 0.71** (rewording removes the positive evidence it exclusively
relies on — the honest counterweight to its certificate). Two agent instances did the rewriting and
judging separately (attacker never saw judge), but SAME model family = partial independence only;
judge is not ground truth (hand-check of the 24 is the proper validation); N is tiny. **Not
reproducible from seed** (LLM output varies) so the outputs are frozen and committed to
`data/llm_paraphrase/` (sample.json / rewrites.json / judge.json + README); the notebook (§32e) reads
them and re-scores deterministically. Generated 2026-07-20.

**P8 human-in-the-loop feedback + deployment envelope (T35–T39, F24) — now report §5.** Simulates a user pressing
"report phishing": one member per campaign is revealed, the rest are scored. Campaigns = 93 groups of
>=3 near-duplicates in the 1,303 modern (2023–25) phishing e-mails, found by **average linkage** at
cosine 0.7 (mean intra-campaign sim 0.883; single-link/connected-components CHAINS and produced a
139-member blob at intra-sim 0.50 — do not use it). **Result (T36):** baseline 0.149 → **1.000** with
a matching report; **campaign-BLIND ablation 0.329** (own report removed) — the matched-vs-blind gap
is the real campaign effect, the rest is "any modern example helps an old model". One-shot retrain
0.991. Control (singletons, sim only) 0.148. FPR unchanged from the model alone (0.0012/0.0017).
**Arm C (T37):** dilution degrades similarity slowly (0.309 evade at +80 words), synonyms do nothing,
but **homoglyph+leet evades 100%** — and the §21 UTS-39 skeleton restores it to 0.034. The certified
normaliser is a *precondition* for the feedback loop, not just a classifier defence. **Arm D (T38):**
false reports raise FPR 0.0012 → 0.0055 (10% bad) → 0.0115 (50%), averaged over 10 seeds — bounded
but real; the report button is an attack surface.
**Methodology warning worth keeping:** two earlier calibrations of tau on *ham* (CEAS-2008, then
modern Apache) both gave recall ~1.0 with a broken control (singletons "detected" at 0.92–0.94),
because any ham-fitted threshold only separates "scam-like" from "mailing-list" — the T26/T30 genre
confound. tau must come from **different-campaign** similarity; ham FPR is then *reported*, not used
to fit. Campaigns are also split calibration/evaluation so tau never sees an e-mail it scores.
Notebook §32c/§32d; report **§5** (Adaptive Deployment and Human-in-the-Loop Learning: §5.1
feedback experiment, §5.2 measured deployment envelope T39 — ~3 ms/e-mail, ~6 ms p95, several hundred/sec/core,
2.7 MB, 30 s retrain; §5.3 continual learning as **future work**, deliberately not claimed as a
contribution because §28 and §32c already do one retraining step each and no time-ordered modern
*ham* exists). Plain-language write-up `IMPROVEMENTS_STORY.md` Stage 10. Ref: althobaiti2023clustering.

**P7 monotone certificate (T31–T34, F23).** A *trained* non-negative
(monotone) unigram model: weights constrained `w >= 0` on binary word-presence, so inserting text can
only raise the score → dilution provably cannot flip a flag. Certificate is **exact**: 0 score
decreases and **100%** of clean-flagged phishing stay flagged at +40/+100/+300 inserted words (T33);
the only leak is our own `MAX_CHARS` truncation (99.8%/99.6%, 9 and 21 decreases). Dilution recall is
a **flat 0.87** vs the union model's 0.998→0.129 collapse (T32). **The honest cost:** clean F1
**0.998 → 0.912**, MCC 0.997 → 0.869 (T31) — it is a *safety net, not a replacement*. Used as an OR
with the union model it gives a certified floor of 0.87 under heavy dilution at FPR 0.0012 → 0.0225
(T34). Key subtlety: the certificate holds **only for unigram** features — with bigrams, insertion
destroys word pairs (`verify account` → `verify now account`) and the score *can* fall (measured: 388
of 500). Distinct from the §29 evidence-floor, which post-hoc *clips* a trained model's weights and
uses a hand-tuned threshold; §29's "cannot lower it" claim is itself slightly loose because it is
built on the bigram vectoriser. Notebook §32b; report §4.12b (`\label{sec:monotone}`) in both `main.tex` and `report.md`;
plain-language write-up in `IMPROVEMENTS_STORY.md` Stage 9; `PROJECT_PLAN.md` v7 changelog.
Citations added to `references.bib`: `fleshman2018nonnegative` (the non-negativity construction),
`lowd2005good` (the good-word attack our dilution attack instantiates), `jorgensen2008multiple`.

**Phase-4 hardening/probing (T27–T30):** P1 dilution **evidence-floor** (attack-agnostic) recovers
0.13 → 0.36/0.50/0.55/0.66 at FPR 1/3/5/10% (T27, F22) — partial, structural weakness. P2 **paraphrase**
does NOT evade (0.998, T28). P6 **SpamAssassin** replicates clean 0.97 → attacked 0.50 → cleaned 0.965
(T29). P3 **confound scrub** leaves separability 0.99→0.99 (T30) — genre, not formatting.

**DistilBERT (T7, P5):** now on the **full test set** (5,327; train 3,000, CPU) → **F1 0.984**, attacked
recall 0.999. Ties/slightly below TF-IDF; attack does not transfer (sub-word tokeniser). GPU full-data
run = future work.

**Error analysis:** only 4 FP / 4 FN of 5,327, all borderline.

## 9. Robustness / adversarial experiments implemented
Attacks (`src/attacks.py`): homoglyph, leetspeak, full-width (held-out), zero-width, adaptive (unmapped
confusables), mixed, **importance-guided greedy** (grey-box, DeepWordBug/TextBugger/HotFlip family), and
word-level **synonym / mask / dilution / paraphrase**. Defences (`src/normalize.py` + notebook):
character n-grams, **normalisation** (curated table AND full UTS-39 skeleton), **one-shot and iterative
adversarial training**, a **deterministic robustness certificate**, and a **positive-evidence floor** vs
dilution. Evaluation tiers are honest: *seen / held-out / adaptive / importance-guided*, multi-seed CIs,
cross-corpus, temporal drift, and a real-obfuscation measurement.

## 10. Explainability / analysis implemented
Faithful per-token contributions (`coef × tfidf`, `src/explain.py`) — global top weights + exact
per-email local explanations. Plus: EDA (balance/length/non-ASCII/discriminative tokens/temporal/
outliers), **Spearman correlation** of meta-features (phishing is shorter; length≈word-count redundant),
cost-sensitive thresholding, calibration (reliability/Brier/ECE), and a dedicated error analysis.
Explicit RQ4 trade-off: the readable signal lives in the *word* features; the robust *char* features are
not human-readable.

## 11. Papers actively tested (not just cited)
Rennie 2003 (ComplementNB under imbalance — HELD); Eger 2019 VIPER (collapse-then-shield — HELD);
Pruthi 2019 (rule-based vs principled normaliser — demonstrated: hand-table fails vs targeted attacker,
UTS-39 succeeds); Gao 2018 / Li 2019 / Ebrahimi 2018 (word-importance attack — instantiated);
Goodfellow 2015 (adversarial training generalises only to trained family — shown); morris2020textattack
(cited for the importance-guided attack). Full list in `report/references.bib`.

## 12. What is finished vs what still needs doing
**Finished:** the entire content-detector pipeline; all attacks/defences; the arms-race resolution with
the UTS-39 certificate; the **monotone dilution certificate (P7)**; **human-in-the-loop feedback +
deployment envelope (P8)**; the **LLM-paraphrase probe, LLM-judged (P9)**; **course grounding + deeper
EDA (v10)**; modern-data drift + real-obfuscation + evolve; DistilBERT (full-test); the **29-page report**
(main.tex + report.md mirror); README; the plain-language docs; the **25-slide presentation**; all result
tables/figures. Notebook runs with **0 errors (148 cells)**; git repo committed.

**Still to do:**
1. ~~`git init` + first commit~~ — **DONE.** (The `.gitignore` had a broken rule — `data/raw/` excludes
   the directory so git never descends into it and the `!.../nazario_recent.csv` re-includes were dead;
   now excludes `data/raw/*`, so the two committed modern CSVs really are tracked.)
2. ~~Build the presentation~~ — **DONE.** `presentation/phishfusion_presentation.html` (25 slides,
   self-contained). `docs/IMPROVEMENTS_STORY.md` (Stages 0–12) is the speaker script.
3. **Optional future work** (all genuinely optional, listed as future work in the report):
   a structure-aware model that defends dilution outright; a *neural* paraphrase attacker at scale; a
   personal-style modern ham source for a fully-clean modern retrain; a GPU full-data DistilBERT run;
   an adversarial-input/anomaly detector for obfuscated mail; model-extraction hardening; the fusion
   with Student B (P4 — currently out of scope).

## 13. Known limitations / risks
- CEAS_08 is 2008 → does not transfer across corpora (F1 0.37) or time (14% on modern scams). Evolve
  fixes the time gap; fusion is the answer for the corpus gap.
- **Dilution** is a structural weakness of linear bag-of-words. Three answers, in order of strength:
  the evidence-floor *partially* defends it (0.13 → ≤0.66 at a false-alarm cost); adversarial training
  fixes it but must know the attack (→1.0); the **monotone certificate (P7)** makes it provably
  impossible, but only for *insertion* and at a real accuracy cost (clean F1 0.912) — a safety net, not
  a replacement. Semantic *rewording* (the LLM probe, P9) is the one class no certificate covers.
- The modern **mixed** detector has a disclosed provenance confound (scam vs mailing-list, 99% separable)
  → the trustworthy modern result is Experiment A (ham-source-unchanged, 0.98), not the mixed F1 0.996.
- DistilBERT is CPU-trained on a subset (full-test eval only) → indicative, not the transformer's ceiling.
- Attacks target phishing only (attacker wants scams through); the reverse direction is untested.
- **Reproducibility caveat:** classical results are deterministic (seed 42) and reproduce byte-identically;
  only **DistilBERT** (slightly) and the **T39 timing benchmark** vary run-to-run. Classical-only
  `run_all.py` is **~45 min**; the full run (with BERT) is over an hour.

## 14. Important design decisions & reasoning
- Single mixed-class corpus (CEAS) to kill the confound by construction; OOD corpora only for the honest
  cross-corpus number. Strip URLs (content-only). 35% phishing prior (realistic + makes attacks visible).
- Test set opened ONCE; all tuning on validation. Recall/F2/MCC over accuracy (accuracy paradox).
- Normalisation kept as a separable defence arm, never baked in. Meta-features analysis-only (except the
  P1 rule). Attack evaluation always includes held-out + adaptive tiers so defence success isn't circular.
- The notebook is generated from `scripts/build_nb.py` (do NOT hand-edit the `.ipynb` for content changes;
  edit the builder, then rebuild + execute + `add_analysis.py`). Analysis markdown cells deliberately use
  qualitative language (no hard-coded numbers) so they survive re-runs; exact numbers live in code outputs,
  tables, and figures.

## 15. Professor / assignment requirements
Prof. Dr. Uri Itai requires: **submit via Git** (✅ done), full code + report + **presentation**
(✅ all done — `presentation/phishfusion_presentation.html`), clear structure + README (✅), reproducible
results (✅, `run_all.py`), an academic report with the standard sections incl. a ~1-page executive summary
(✅), and — critically — **actually test paper claims** ("paper said X, I got Y because Z") with critical
evaluation, correct metrics (MCC, Fβ), and error analysis (✅). All core requirements are met; the only
open items are the optional future-work directions in §12.

## 16. Instructions for the next Claude Code chat
- **Read first:** THIS file (incl. §3c recent-sessions recap), then `README.md`, then skim
  `report/report.md` (readable mirror of the PDF), `docs/PROJECT_PLAN.md` (§11 planned tasks, §12 pipeline
  map, v3–v10 changelogs), and `docs/DECISIONS_AND_PITFALLS.md` (the bugs + how we caught them).
- **Trust:** the **result tables** (`results/tables/*.csv`) and the **executed notebook outputs** as
  ground truth for numbers. `report/main.tex` and `report/report.md` are kept in sync with them.
- **Note on the plan:** `docs/PROJECT_PLAN.md`'s v2 body was written before code ran; the clarifier at
  its top says so. Trust its **changelogs (v3–v10) and the result tables**, not any leftover
  design-intent phrasing in the older v2 text.
- **To reproduce / regenerate:** from repo root, `py -3 run_all.py` (needs the raw CSVs in
  `data/raw/kaggle_phishing/` and the committed modern CSVs + `data/unicode/confusables.txt`).
  `PHISH_RUN_BERT=0 py -3 run_all.py` skips DistilBERT for a classical-only run (**~45 min** — not the
  old "3–4 min"; the phase-2/3/4 sections are the expensive part). To change the notebook: edit
  `scripts/build_nb.py` → run it → `run_all.py` → `py -3 scripts/add_analysis.py`. Compile the report
  with the Tectonic binary (see §3b above for how to get it). **Prototype changes on the frozen
  `emails_working.csv` first** before paying the full re-run.
- **Verify before trusting your own memory:** if a memory or doc names a file/function/number, check it
  against the current code and tables first.

---


