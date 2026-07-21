# Decisions & pitfalls — the "how" and "why" behind the results

This file is the project's *institutional memory*: the non-obvious bugs we caught, **how** we
caught them, and the methodological rules that prevented more. The other docs record *what* the
project does and *what* the numbers are; this one records the reasoning, so the same traps are
not fallen into again. Read it alongside `IMPROVEMENTS_STORY.md` (plain-language narrative) and
the git log (each fix has a detailed commit message).

---

## A. Traps we caught, and how

Most of these were caught by **prototyping on the frozen `emails_working.csv` before a full
re-run**, or by building a **control that must fail** and noticing it didn't. Both habits are
worth keeping.

1. **`.gitignore` silently dropped committed data.** `data/raw/` excludes the *directory*, and
   git never descends into an excluded directory, so the `!data/raw/.../nazario_recent.csv`
   re-includes were dead rules — the two modern CSVs the project calls "COMMITTED" would have
   been left out of the first commit. Fix: exclude `data/raw/*` (contents), then re-include the
   two CSVs. *Lesson: `!` re-includes don't work under an excluded directory.*

2. **A fake "balanced" data point in the CNB imbalance sweep.** The code did
   `n_pos = min(n_pos, len(pos))`; the working set is 35% phishing, so the 0.50 target consumed
   all positives and the "50%" point silently re-used the **35% subset** — which is why T3b's
   last two rows were byte-identical. Caught by reading the table, then reproduced on frozen data.
   Fix: down-sample the *majority* class instead. The fix *strengthened* the Rennie-2003
   replication: at a genuine 50/50 the CNB advantage vanishes exactly (0.976 vs 0.976), which is
   what the theory predicts. *Lesson: a silent `min()` clamp can manufacture a result.*

3. **The fast re-run destroyed a real result.** `T7_distilbert.csv` was written unconditionally,
   so the documented `PHISH_RUN_BERT=0` classical-only run overwrote the real DistilBERT numbers
   with a `"skipped"` placeholder. Fix: only write T7 when the arm actually ran.

4. **The monotone certificate failed on bigram features.** First prototype trained the
   non-negative model on the usual 1–2-gram features. Inserting a word *between* two words
   (`verify account` → `verify now account`) destroys the bigram, its indicator drops 1→0, and
   with `w ≥ 0` the score *falls* — so the "insertion can only raise the score" proof was false
   (score decreased on **388 of 500** e-mails). Only **order-free unigram** bag-of-words is truly
   monotone. Caught because the empirical "0 decreases" check did not come back 0.
   *Lesson: a proof is only as strong as the feature assumption underneath it — test it.*

5. **A silently-capped attack budget.** `attack_insert` samples filler words **without
   replacement**, so a 120-word pool capped a "+300 words" request at 120 — the certificate
   "stress test" was not actually stressing. Fix: a 400-word pool, and we *verified the real
   inserted counts* before trusting the table. *Lesson: check that the knob you turned actually
   moved.*

6. **Our own truncation leaks the certificate.** Padding an e-mail can push genuine phishing
   evidence past the `MAX_CHARS = 2000` cut-off, so a flagged e-mail can flip — the certificate is
   exact only on untruncated text (100%), and 99.6–99.8% with the cap. We **report this as its own
   rows**, not quietly evaluate without the cap. *Lesson: state the leak, don't hide it.*

7. **The human-in-the-loop threshold measured the wrong thing (twice).** Calibrating the
   similarity threshold τ on *legitimate* mail (first CEAS-2008 ham, then modern Apache ham) both
   gave recall ≈ 1.0 **with a broken control**: e-mails with no near-duplicates were "detected"
   at 0.92–0.94, which is impossible if the mechanism works by matching campaigns. The cause is
   the genre confound already documented in T26/T30 — a ham-fitted threshold separates
   "scam-like" from "mailing-list", not "this campaign" from "other campaigns". Fix: calibrate τ
   on **different-campaign** similarity and *report* ham FPR at that τ. *Lesson: a control that
   must fail is the fastest way to catch a metric that's measuring the wrong thing.*

8. **A "campaign" that was a chain.** Connected-components / single-link clustering **chains**
   (A~B, B~C, but A≁C) and produced a 139-member "campaign" whose members averaged only **0.50**
   internal similarity — a blob, not a campaign. Fix: **average-linkage** clustering (93 coherent
   campaigns, mean intra-similarity 0.88). *Lesson: verify cluster coherence, don't trust size.*

9. **τ leakage in the feedback experiment.** τ was first derived from *all* campaigns including
   the ones being evaluated. Fix: split campaigns into calibration/evaluation halves so τ never
   sees an e-mail it later scores. (Same discipline as train/test.)

10. **A noisy poisoning curve.** The false-report FPR was non-monotonic on only 93 campaigns.
    Fix: average over 10 seeds → a clean monotone curve.

11. **The LLM attack needed a validity gate.** An LLM rewrite that stops being a scam must not
    count as an "evasion" — otherwise we'd be measuring *deleted* attacks. Fix: a **second,
    independent LLM judge** validates each rewrite (it dropped 6 of 30). *Lesson: when the attack
    generator can change the label, validate the label separately.*

12. **LLM output is not seed-reproducible.** So we generate the rewrites/verdicts **once**, commit
    them to `data/llm_paraphrase/`, and have the notebook read the frozen files and re-score them
    deterministically — the same pattern as the committed modern-mail CSVs.

13. **Presentation gremlins.** Duplicate `style=` attributes (the browser keeps the first, drops
    the second, silently losing the animation delay); a decorative background line that overlapped
    the centred title on the first/last slides (moved to a bottom strip — and note the two hero
    SVGs were formatted differently, so the first fix only caught one of them).

14. **Timing numbers drift.** The deployment latency/throughput vary run-to-run (p95 has been
    5.82 and 6.68 ms across runs), so the report quotes **bands** ("~3 ms median, ~6 ms p95,
    several hundred/sec, ~2.7 MB, seconds to retrain"), not false precision.

15. **The Unix-epoch date trap — checked, and clean.** A classic EDA pitfall (flagged in the
    course EDA lecture): a *missing* date imputed to 0 parses to **1970-01-01** (Unix time 0)
    and then hides in a temporal plot as if it were a real timestamp. We tested for it directly
    with `scripts/check_date_trap.py`: **0** of 34,342 harmonised CEAS dates (and 0 of the 26,633
    working rows) fall on 1970 / the epoch — so there is **no imputed-zero contamination**. The
    ~0.3% of dates outside 2007-2009 are genuine malformed/**spoofed** RFC-822 headers (real
    strings like `Mon, 05 Aug 2086 21:34:28 -0300`), not imputation artefacts, and the notebook's
    temporal analysis (§2) already restricts the plot to 2007-2009, so they are excluded. *Lesson:
    a known trap that turns out absent is still worth checking and recording — "we looked, it's
    clean" is a result, not a non-event.*

---

## B. Methodological rules we kept (the "why")

- **Prototype on the frozen data first.** The core pipeline is deterministic (seed 42); a
  classical re-run reproduces ~31/32 tables byte-for-byte. So we prototype a change on
  `emails_working.csv`, confirm the numbers, *then* pay the ~45-minute re-run. This caught #2 and
  #4 above before they reached the report.
- **Single mixed-class corpus (CEAS_08).** Removes the corpus/provenance confound *by
  construction* — a corpus-ID probe scores ~0.99, proving pooling single-class corpora would let a
  model cheat.
- **Leakage controls, verified.** Dedup happens *before* the split; the vectoriser is fit on
  **train only** (transform on val/test); adversarial training perturbs only training rows; the
  test set is opened once; the feedback experiment splits by campaign. All confirmed by reading
  the code, not assumed.
- **Honest tiers for attacks** — *seen / held-out / adaptive / importance-guided* — so a defence's
  success is never measured against its own inverse.
- **Report the bad numbers.** Monotone clean F1 0.912 (not just the certificate), the truncation
  leak, the campaign-blind ablation (0.33, not just the headline 1.0), the LLM probe's tiny N and
  partial independence, the dilution collapse to 0.13. The project's value *is* the honesty.
- **The recurring self-check.** The "five times my own numbers fooled me" slide is not a gimmick;
  it is the accuracy paradox, the fake 50% point, the dilution collapse, the chained campaign, and
  the ham-calibrated threshold — each caught by a control or a re-read.

---

## C. Reproducibility rules

- **Classical results are deterministic** (seed 42) and reproduce byte-identically. Two things
  legitimately vary: **DistilBERT** (slightly, run-to-run) and the **T39 timing benchmark**.
- **`preds_A_*.csv` churn is cosmetic:** across runs the probabilities are bit-identical; only the
  human-readable top-k `reason` strings reorder (a tie-break). Not a model change.
- **To change the notebook:** edit `scripts/build_nb.py` → `py -3 scripts/build_nb.py` →
  `PHISH_RUN_BERT=0 py -3 run_all.py` → `py -3 scripts/add_analysis.py`. For a *markdown-only*
  change you may patch the executed `.ipynb` cell directly to avoid a re-run (the builder is still
  the source of truth — update it too).
- **LLM-derived artefacts** (`data/llm_paraphrase/*.json`) are frozen and committed because LLM
  output is not seed-reproducible; the notebook reads them rather than regenerating.
- **After any change:** cross-check every headline number against `results/tables/*.csv` in all of
  README, `report/main.tex`, `report/report.md`, and the three docs before trusting the write-up.
