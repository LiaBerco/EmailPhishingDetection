# PhishFusion — the whole project explained simply (study & presentation guide)

*Everything in one place, in plain words: the big picture, every step of the pipeline, every
attack and fix (Rounds A–H), all the results, why each choice was made, and likely lecturer questions
with answers. Every number is a real result from running the code. For the full session-by-session
narrative see `IMPROVEMENTS_STORY.md`; for the entry-point handoff see `PROJECT_HANDOFF.md`.*

---

## 1. The big picture (say this first)

**The problem.** Phishing = scam emails that pretend to be someone you trust ("your account will
be suspended — verify now") to steal passwords or money. A lot of the trick is in the **words**,
so reading the text with NLP (natural-language processing) is a natural way to catch it.

**The project.** "PhishFusion" has two halves. **I am Student A: the *content* detector** — I read
only the email's text (subject + body) and decide **scam or normal**. A partner (Student B) reads
the *links and sender*. Later the two are combined ("fused"). My job is the text half + testing it.

**The one big idea (my contribution).** I didn't invent a new model. I took a *standard, simple*
detector and did an **honest, thorough robustness study**: I showed that a near-perfect score is
**fragile**, I **attacked** it the way a real criminal would, **fixed** it (one fix even comes with
a mathematical proof), attacked it again a smarter way, fixed that too, and finally **kept it
current** on real 2023–2025 scams. Where a fix is incomplete, I say so — which is exactly the
argument for pairing my text detector with a second, independent one.

**The story in one line:** *simple detector looks perfect → it's actually fragile → attack → fix →
attack smarter → fix with a proof → prove the threat is real & the model goes stale → evolve it.*

---

## 2. The pipeline (how the detector is built, step by step)

Think of it as: **raw emails → numbers → a model → stress-test the model.**

1. **Get the data.** I use **CEAS-2008**, a set of ~39k real emails that has **both** scam and
   normal mail **from the same place**.
   - *Why this one?* If scam emails came from one source and normal emails from another, the model
     could "cheat" by spotting *which source* an email is from instead of *whether it's a scam*. I
     even proved this danger: a model can guess the source with ~99% accuracy. Using one mixed
     source removes the trap **by design**.

2. **Clean the text.**
   - **Strip links** (replace each URL with the word "URL"). *Why?* Links are the partner's job; my
     text detector should earn its score from language, not links.
   - **Rebalance to 35% scam** (make normal mail the majority). *Why?* Real inboxes are mostly
     normal, and — importantly — if scam were the majority, a broken model could "pass" by just
     guessing "scam" for everything, which would hide the attacks. This makes the later tests fair.
   - Final working set: **26,633 emails**.

3. **Split the data 60/20/20** (train / validation / test), fixed random seed.
   - *Why?* **Train** teaches the model; **validation** is used to make choices/tuning; **test** is
     opened **once**, at the very end. This stops "cheating by peeking" and keeps the final number
     honest.

4. **Turn text into numbers — TF-IDF.**
   - **TF-IDF** = count how often a word appears, but down-weight words that appear in *every* email
     (like "the") and up-weight rare, telling ones (like "verify account").
   - I build **three versions**: **word** features (vocabulary), **character-chunk** features (3–5
     letter pieces, which capture *shape*), and their **union** (both together — the working model).
   - *Why character chunks?* If an attacker writes "cl1ck", the word "click" is gone but the letter
     pieces still overlap — so character features survive letter-tricks. This matters a lot later.

5. **Train models and pick the best.**
   - **Naive Bayes** (the classic, fast spam filter) and **Logistic Regression** (a strong model
     whose weights you can *read*). I also test **Complement Naive Bayes**, a version made for uneven
     classes.
   - **Logistic Regression wins** and is the working model. *Why keep it simple?* It's accurate,
     fast, and — crucially — **explainable**: you can see exactly which words drove a decision.

6. **Measure properly.** (This is where I show I understand evaluation.)
   - **Accuracy alone lies** when classes are uneven — a do-nothing model that says "normal" every
     time looks ~65% accurate while catching **zero** scams (the "accuracy paradox").
   - So I report: **recall** (% of scams caught), **precision** (% of flagged mail that's really
     scam), **F1** (balance of the two), **F2** (like F1 but weights recall more, because *missing a
     scam is worse than a false alarm*), **MCC** (one honest score that uses all four outcomes), and
     **AUC** (how well it *ranks* scam above normal).
   - **Confidence intervals** (error bars, by "bootstrap" resampling) so I don't over-read a
     0.001 difference, and **McNemar's test** to check "is model A *really* better than B, or luck?"

7. **Explain decisions.** For this model a word's contribution is exactly its *weight × TF-IDF*, and
   these add up to the decision. So I can show the *exact* words that pushed an email toward "scam"
   — a **faithful** explanation, not a guess. (Trade-off: only the *word* features are readable; the
   robust *character* features are not — I state this honestly.)

8. **Then the main event: attack → defend** (Sections 3–4 below).

9. **Extra checks:** a transformer baseline (DistilBERT), testing on totally different email sets
   (cross-corpus), a look at the individual mistakes (error analysis), and whether the scores are
   "trustworthy numbers" (calibration).

---

## 3. The attacks and the fixes (the cybersecurity heart)

This is the part to spend the most time on. It's a back-and-forth: I attack, then defend, repeatedly.

### Round A — letter-tricks
- **Attack:** swap letters for **identical-looking** foreign letters (e.g. a Cyrillic "а" that
  looks exactly like an English "a"), and leetspeak ("cl1ck"). A human reads it fine; the detector
  sees unknown words.
- **Result:** the word-only detector **collapsed** — caught **99% → 71%** of scams. Invisible to a
  person, devastating to the filter.
- **Fix:** (1) **character chunks** (already robust), and (2) **normalisation** — clean the email
  first, turning look-alike letters back to plain ones before the detector reads it.
- **Result:** recall back to **~99%**. I also tested a trick the cleaner was *not* built for
  (full-width letters) and it still worked — a sign the defence generalises.

### Round B — a second, different defence
- **Adversarial training:** also show the model scam examples with the tricks already applied, so it
  learns them.
- **Finding:** the two defences are **complementary** — normalisation handles *new* trick families;
  adversarial training is best on the family it trained on. Neither alone is complete; together they
  cover everything. (This mirrors a known result — Goodfellow 2015.)

### Round C — a *smart* attacker (importance-guided)
- **Attack:** instead of messing with every letter, aim at the **few words the model relies on most**
  and disguise just those.
- **Result:** far more efficient — **~9 disguised words** flip a scam email; and against this smart
  attacker the "fully-defended" model dropped to **0.88** (union). *So the earlier fix looked
  complete but wasn't, once the attacker got clever.*

### Round D — fix it *properly*, with a proof
- **Fix 1 — a principled cleaner:** replace my small hand-made list of look-alike letters with the
  **full official Unicode list** of look-alikes (the same data web browsers use to catch fake
  website names).
- **Result:** recall back to **0.993 / 0.998**. The old hand-list did **nothing** against the smart
  attacker (0.67); the official list catches all its tricks.
- **Fix 2 — a certificate (a proof):** for the whole family of letter-tricks, the *cleaned* email is
  **identical** whether or not it was attacked — so the decision **cannot** change. This is true for
  **100%** of scam emails (the old list: 0%). A guarantee, not just an average.
- **Fix 3 — iterative adversarial training:** repeatedly show the model the smart attack and retrain.
  Recall **0.44 → 0.999**, beating the old one-shot method (0.975).
- **Honest end:** the *letter* attack is now closed — with a proof. So a clever attacker must switch
  to a *different kind* of trick (Round E). The fight isn't "won"; it moves.

### Round E — hack the *words*, not the letters
- **Attacks:** (a) swap important scam words for synonyms; (b) delete them; (c) **dilution** — bury
  the scam in lots of normal-sounding filler words.
- **Result:** synonyms barely worked (**0.998** — the model is robust to small word swaps);
  deletion hurt moderately; **dilution is the killer — recall 0.998 → 0.13**, because a word-counting
  model just *adds up* signals, so enough "safe" words outvote the scam words.
- The **letter-cleaner does nothing** here (0.998 → 0.997) — wrong tool for this attack.
- **Fix:** train the model on these word-tricks too → recall back to **1.00**, no loss on normal mail.
- **Honest lesson:** a word-counting model is *inherently* weak to dilution; adversarial training
  patches it, but the deeper fixes are structure-aware features (a padded scam *looks* padded) and
  the **fusion** with a second detector.

### Round F — a *second* proof (for words)
- **The idea:** instead of *patching* dilution, build a model that **cannot** be diluted. Force every
  word's weight to be **≥ 0** and only look at *which* words appear. Then adding text can only *raise*
  the scam score — so padding **provably** cannot turn a caught scam into "safe".
- **Result:** the proof holds exactly (0 score drops; 100% of caught scams stay caught at +40/+100/+300
  filler words) where the normal model collapses to 0.13.
- **The honest cost:** this "provable" model is a **worse everyday detector** — clean F1 drops
  **0.998 → 0.912**, because it may only collect *guilt* evidence. So it's a **certified safety net used
  alongside** the main model, not a replacement. (Subtlety: the proof needs *single words* — with word
  *pairs* it breaks, because inserting a word splits a pair.)
- **Two proofs now:** letters can't beat it (Round D), and *added* words can't beat it (Round F). The
  only move left is **rewording** the scam — see Round H.

### Round G — learning from people (deployment)
- **One user report → the whole campaign.** Scams are sent to many employees at once. If **one** person
  reports a scam, the system finds the near-identical copies in everyone's inbox: catch-rate on the rest
  jumps **0.15 → 1.0**, with **no extra false alarms**. (Honest check: an ablation shows the real
  campaign-specific effect is 1.0 vs **0.33** — the rest is just "modern examples help".)
- **It needs my earlier defence:** look-alike letters switch this feature **off** (100% escape) until the
  Unicode cleaner is applied (→ 3%). And **false reports poison it** (false alarms rise) — the report
  button is itself an attack surface.
- **A living defence (dynamic database):** generalise this — whenever a *new, unknown* attack is seen
  (a report, a monitor, a honeypot), validate it, add it to a growing database, and retrain. Retraining
  is cheap (~30 s), so "how often to retrain" is a *policy* choice, not a compute limit.
- **Runs anywhere:** the model is tiny and fast (~3 ms/email, ~2.7 MB) → it can run **on your phone**,
  so email content never leaves the device (privacy). It's **one layer** of a real system — it doesn't
  even use the sender's address (that's the partner's half).

### Round H — the frontier: get an AI to reword the scam
- **The one attack no proof covers:** rewrite the scam in *different words*. I tested it with a
  **three-stage pipeline**: an AI rewrites the scam → a *second, independent* AI judges "is it still a
  scam?" (so a rewrite that stops being a scam isn't miscounted) → my detectors score the valid ones.
- **Result (small probe, 24 of 30 valid):** my **main model shrugs it off (recall 1.0)** — rewording
  doesn't fool it. But the **"provable" model gives ground (0.75 → 0.71)**, exactly as expected: it only
  knows guilt-words, and rewording swaps those out. Honest counterweight to its certificate.

*The full blow-by-blow of every round (with the bugs I caught along the way) is in
`IMPROVEMENTS_STORY.md`; the traps and the "how" are in `DECISIONS_AND_PITFALLS.md`.*

---

## 4. The reality checks (does any of this matter in the real world?)

- **Different email sets (cross-corpus):** trained on 2008 CEAS, tested on completely different
  email sets → F1 crashes **0.998 → 0.37**. *Diagnosis:* it's mostly a *threshold* problem (the
  ranking is still okay, AUC 0.78; re-tuning recovers F1 to 0.75) plus unfamiliar vocabulary
  (unknown-word rate 7% → 20%). Honest: the lab number does not survive the real world.
- **Modern scams (2023–2025):** I downloaded **1,303 real recent scam emails**. The 2008 model
  catches only **14%** of them → proof the model goes **stale over time** (concept drift).
- **Are the tricks real?** I measured obfuscation in real modern scams: look-alike letters and
  invisible characters are **~0% in 2008 → ~2–4% now** → the attacks I defend against are a **real,
  growing** technique, not made up.
- **Evolve it:** retrain with modern scam examples → catching modern scams jumps **15% → 98%**, with
  **no forgetting** (still 0.998 on old scams) and **no extra false alarms** (0.7%). A detector kept
  current works; a frozen one goes blind.

---

## 5. All the results in one place

**Clean detection (the "too good" number):** F1 **0.998**, MCC **0.997**, AUC ~1.0.
**Naive Bayes:** MultinomialNB F1 0.972 / ComplementNB 0.976. **DistilBERT (full-test eval):** F1 0.984 (slightly
*below* the simple model — see Q&A).

**Robustness — recall on scams (word-only | union):**
clean 0.99 | 1.00 · homoglyph+leet 0.71 | 0.93 · +normalise 0.99 | 1.00 · full-width 0.68 | 0.92 ·
+normalise 0.99 | 1.00 · adaptive 0.89 | 0.98 · +normalise 0.96 | 0.99.

**Smart (importance-guided) attacker:** 8 targeted edits → recall 0.74 (vs 0.98 for random);
fully-defended ceiling **union 0.88 / word 0.67**.

**Principled fix (Unicode UTS-39 cleaner):** recovers to **0.993 / 0.998**; **certificate = 100%**
provably safe (old list 0%); iterative adversarial training **0.44 → 0.999**; costs no extra false
alarms.

**Word attacks:** synonym 0.998 · delete → 0.78 · **dilution → 0.13**; fixed by adversarial training
→ **1.00**.

**Reality:** cross-corpus F1 **0.998 → 0.37** (AUC still 0.78); modern scams caught **14%**; real
obfuscation **~0% → ~2–4%**; evolve **15% → 98%** (no forgetting, false alarms 0.7%); modern mixed
detector F1 0.996 but sources 99% separable (disclosed as inflated).

**Extra honesty checks:** attack numbers stable across 10 runs (wobble ≤ 0.004); cleaning normal
mail raises false alarms only 0.12% → 0.40%; scores well-calibrated in-place (error 0.0020 → 0.0014
after a small tune) but untrustworthy out-of-place.

**Error analysis (clean test):** only **4 false alarms / 4 misses** out of 5,327 — all borderline
cases, no systematic blind spot.

**Further hardening & probing:**
- **Dilution defence (attack-agnostic):** an "evidence-floor" lifts diluted-phishing recall from
  **0.13 → 0.36 / 0.50 / 0.55 / 0.66** at false-alarm budgets of 1 / 3 / 5 / 10%. Partial —
  dilution is a structural bag-of-words weakness.
- **Paraphrase attack (rule-based):** meaning-preserving rewrite does **not** evade — recall stays **0.998**.
- **Second corpus (SpamAssassin):** clean 0.97 → attacked 0.50 → cleaned **0.965** — findings replicate.
- **Confound check:** stripping mailing-list boilerplate leaves source-separability at **0.99 → 0.99**
  (genre, not formatting) — so the ham-source-unchanged modern result (0.98) is the trustworthy one.

**The later rounds (F–H):**
- **Monotone certificate (2nd proof):** provably un-dilutable; clean F1 cost **0.998 → 0.912**; used as a
  certified *safety net* OR'd with the main model (floor ~0.87 under heavy dilution, ~2% false alarms).
- **User reports:** one report → rest of the campaign **0.15 → 1.0** (campaign-blind ablation 0.33);
  obfuscation disables it (100% escape) until the Unicode cleaner is applied (→3%); false reports poison
  it (false alarms 0.0012 → 0.0115).
- **LLM reword attack (LLM-judged, small probe):** main model **1.0** (unmoved), monotone **0.75 → 0.71**.
- **Deployment:** ~3 ms/email, ~2.7 MB, ~hundreds/sec → runs client-side; retrain ~30 s.
- **Deeper EDA:** skewness/kurtosis (why Spearman + median), VIF ≈ 21 (length/word-count redundant),
  Mann–Whitney test of "scams are shorter" (p ≪ 0.001).

---

## 6. The papers I actually tested (not just cited)

- **Rennie 2003 (Complement NB):** claim — CNB helps when classes are uneven. I squeezed the scam
  class down and CNB pulled ahead of plain NB as predicted. **Held.**
- **Eger 2019 (VIPER):** claim — look-alike attacks break NLP, cleaning shields it. Reproduced the
  collapse-then-recover on my data. **Held.**
- **Pruthi 2019:** hand-list vs principled cleaner. I showed the hand-list fails against the smart
  attacker but the principled Unicode cleaner succeeds. **Demonstrated.**
- **DeepWordBug / TextBugger / HotFlip (2018–2019):** the "attack the important words" idea — this
  is exactly my smart attacker.
- **Goodfellow 2015:** adversarial training hardens against the trained attack but not others — I
  showed this, and that it must sit *alongside* normalisation.

---

## 7. Honest limitations (say these — the lecturer rewards honesty)

1. CEAS is from 2008; it doesn't transfer across email sets or across time (I show both, and evolve
   it as the fix).
2. Two attack classes are now closed by proof (letters via UTS-39; added words via the monotone
   model), but the monotone proof costs accuracy and covers *insertion* only — a structure-aware
   defence against **dilution** that keeps clean accuracy, and a truly open **semantic/rewording**
   attacker (the one class no certificate covers), are future work.
3. DistilBERT was run small (CPU) — suggestive, not the last word on transformers.
4. I only attacked scam emails (an attacker wants scams to slip through), not the reverse.
5. Modern normal mail is scarce publicly, so the modern set is scam-only for the clean test, and the
   mixed detector has a disclosed source-confound.
6. Not implemented but relevant (future work): an anomaly detector that flags obfuscated mail as
   suspicious, and hardening against **model-extraction** (an attacker cloning the detector via queries).

---

## 8. Likely lecturer questions — and simple answers

**"Why is your score so high (0.998)? Is it leakage/overfitting?"**
No. The single-source data removes the corpus-cheat (I proved the danger); the test set is touched
only once; validation carries all tuning. But the whole point is that this high score is *fragile* —
that's what the rest of the project shows.

**"Why not just use deep learning / BERT?"**
I did test DistilBERT. On this keyword-driven text it *ties or slightly trails* the simple model,
because "which words appear" is already captured by bag-of-words — the transformer's context adds
little here. And the simple model is **explainable**, which matters for security. (Nice bonus: the
letter-attack didn't transfer to BERT, because its sub-word tokeniser breaks obfuscated words up.)

**"Isn't attacking your own model and then defending it circular / rigged?"**
I guarded against exactly that: I tested attack families the defence was *not* built for (held-out),
an *adaptive* attacker that avoids the defence, and a *smart importance-guided* attacker — and I
grounded it in *real* modern scams. The defence has to earn each result.

**"Do real attackers actually use these tricks?"**
Yes — I measured it: look-alike letters and invisible characters are ~0% in 2008 scams but appear in
a few percent of 2023–2025 scams, and rising.

**"What's the biggest weakness of your detector?"**
Three, all reported honestly: (1) it doesn't generalise to other email sets (F1 → 0.37); (2) it goes
stale over time (14% on modern scams) — fixed by retraining; (3) it's vulnerable to word *dilution*.
The first and third are the strongest arguments for **fusion** with a second, independent detector.

**"Why did you use mailing lists for modern normal email?"**
It's the only public source of *recent* normal email (private inboxes aren't published). I disclosed
the catch: scam-source and mailing-list-source are 99% separable, so I lead with the clean
experiment (which keeps the normal-mail source unchanged) and flag the mixed one as inflated.

**"What's the certificate — is it really a proof?"**
Yes. After the principled cleaner, an attacked email and the clean email become *byte-for-byte
identical*, so the model *must* give the same answer. It's a deterministic guarantee for the whole
letter-trick family, not an average — true for 100% of scam emails in the test.

**"You didn't fully solve dilution — isn't that a failure?"**
No — it's an honest finding. I built a principled defence that needs *no* knowledge of the attack (an
evidence-floor) and it recovers a lot (0.13 → up to 0.66), but not everything, because short scams
carry little evidence. That shows dilution is a *structural* limit of any word-counting model — which
is exactly why the real answer is a richer model or fusion. Adversarial training *does* fully fix it,
but only once you know the attack. Reporting both is the point.

**"Did you check the results aren't specific to one dataset?"**
Yes — I replayed the whole attack-and-fix story on a second corpus (SpamAssassin) and got the same
shape (0.97 → 0.50 under attack → 0.965 after cleaning). The findings replicate.

**"What would you do next?"**
A structure-aware model that defends dilution outright; a *neural* paraphrase attacker (stronger than
my rule-based one); a personal-style modern ham source for a fully clean modern retrain; a GPU
full-data DistilBERT run; and fusing with Student B's URL/sender detector — so when an attacker
defeats one channel, the other still fires.

---

## 9. Where everything lives (if asked to show it)

- **Notebook** (runs everything end-to-end, 0 errors): `notebooks/student_A_content_detector.ipynb`
- **Report** (29 pages): `report/main.pdf` (source `report/main.tex`, readable copy `report/report.md`)
- **Presentation** (25 slides): `presentation/phishfusion_presentation.html` (open in a browser, press F)
- **Plain-language change story (Stages 0–12):** `docs/IMPROVEMENTS_STORY.md`
- **Bugs caught + the "how" + methodology rules:** `docs/DECISIONS_AND_PITFALLS.md`
- **This guide:** `docs/PROJECT_EXPLAINED.md`
- **Design & pipeline + all changelogs (v2–v10):** `docs/PROJECT_PLAN.md` (pipeline map is its §12)
- **Entry point for a new session:** `PROJECT_HANDOFF.md`
- **Results:** `results/tables/` (T0–T40 + T9b/T9c), `results/figures/` (F1–F24, incl. F13 EDA box plots)
- **Code:** `src/` (data, features, models, attacks, normalise, explain), `scripts/` (build/parse)
- **Data:** CEAS + modern scams (`data/raw/nazario_recent/`) + modern normal (`data/raw/modern_ham/`)
  + Unicode look-alike list (`data/unicode/confusables.txt`) + frozen LLM-attack files
  (`data/llm_paraphrase/`)
