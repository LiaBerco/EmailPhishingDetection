# PhishFusion — the story of the project, in simple words

*This file tells the whole journey in plain language, and for every change explains
**what** I did, **why**, and **how**. It is written to be dropped straight into the
presentation. The one-line story: **I built a scam-email detector, then tried to hack it,
then fixed it, and kept evolving it to make it stronger.***

Every number here is a real result from running the code (not made up).

---

## Stage 0 — The goal, in one breath
Read an email's text and decide: **scam (phishing) or normal?** Then do the interesting
part — try to *fool* the detector and *repair* it, over and over, like testing a lock by
trying to pick it. This is the "cyber" heart of the project.

---

## Stage 1 — Build the detector
**What:** turned each email into numbers (which words and letter-chunks it contains) and
trained a simple, explainable model to separate scam from normal.
**Why:** a simple model is fast, and you can *read* why it made a decision (which words
pushed it) — important for security work.
**How:** TF-IDF features (counts of words and 3–5 letter chunks) + Logistic Regression,
trained on the CEAS-2008 email set (the one big set that has both scam and normal mail
from the *same* place, so the model can't cheat by spotting where mail came from).
**Result:** near-perfect on clean text — F1 **0.998** (F1 = a balanced score out of 1).
**Honest note:** that score is *too* good, and the rest of the project is about showing it
is fragile.

---

## Stage 2 — Hack it (the easy way): look-alike letters
**What:** attacked scam emails by swapping normal letters for **identical-looking** foreign
letters (e.g. a Russian "а" that looks exactly like an English "a"), and "leetspeak"
(cl1ck).
**Why:** a human still reads the word, but the detector sees a different, unknown word — a
cheap, realistic way criminals dodge filters.
**How:** a small attack tool that substitutes letters.
**Result:** the word-based detector collapsed — it caught **99% → 71%** of scams. Invisible
to a person, devastating to the filter.

---

## Stage 3 — Fix it (round 1)
**What:** two repairs. (1) **Normalisation** — clean the email first, turning look-alike
letters back into plain ones before the detector reads it. (2) **Character chunks** — the
detector also looks at 3–5 letter pieces, which survive letter-swaps.
**Why:** stop the cheap trick at the door, and make the detector naturally tougher.
**How:** a text-cleaning step + the character features already in the model.
**Result:** recall back to **~99%**. I also tested attacks the cleaner was *not* built for
(full-width letters) and it still worked — a good sign it generalises. **Adversarial
training** (also showing the model scam examples with the tricks baked in) helped too.

---

## Stage 4 — Hack it harder: aim at the weak spot
**What:** instead of messing with *every* letter randomly, a smart attacker changes only
the **few most important words** the model relies on.
**Why:** far more efficient — fewer edits, same evasion. This is what a real, motivated
attacker would do.
**How:** rank the words by how much they push the "scam" decision, then disguise just the
top ones.
**Result:** with only ~9 disguised words a scam slips through; and against this smart
attacker the earlier fix left a real gap — the fully-defended detector dropped to **~0.88**.
**Lesson:** the first fix looked complete but wasn't, once the attacker got clever.

---

## Stage 5 — Fix it properly (and *prove* it)
**What:** three upgrades. (1) Replace the small hand-made list of look-alike letters with
the **full official Unicode list** of look-alikes (the same data web browsers use to catch
fake website names). (2) A **guarantee**: for the whole family of letter-tricks, the cleaned
email is *identical* whether or not it was attacked — so the decision *cannot* change. (3)
**Iterative adversarial training**: repeatedly show the model the smart attacker's tricks and
retrain.
**Why:** close the gap for good, not with another hand-patch, and be able to *prove* it.
**How:** build the cleaner from the official Unicode "confusables" file; measure the
guarantee; retrain in rounds.
**Result:** recall back to **0.99+**; the guarantee holds for **100%** of scam emails
(the old hand-list: 0%); iterative training lifted recall **0.44 → 0.999**.
**Honest end:** the *letter* attack is now closed — with a proof. So a smart attacker must
switch to a *different* kind of trick (see Stage 9). The fight isn't "won"; it moves.

---

## Stage 6 — Be honest (check the fine print)
**What/Why/How + Result (all real):**
- **Stability:** re-ran the attacks 10 times → the numbers barely move (rock-solid, not luck).
- **Cost of the defence:** cleaning *every* email slightly raises false alarms (from 0.12% to
  0.40%) — cheap, but not free; worth stating.
- **Are the scores trustworthy?** In-corpus, yes (well-calibrated); out-of-place, no — useful
  for later "fusion" with a partner detector.
- **Does it generalise?** Tested on completely different email sets: the score crashes
  (0.998 → 0.37). I diagnosed *why*: it's mostly a threshold/setting issue plus unfamiliar
  vocabulary — not total failure, but a real gap.

---

## Stage 7 — Reality check: modern scams (2023–2025)
**What:** downloaded 1,303 *real* scam emails from 2023–2025 and tested the 2008-trained
detector on them; also measured whether real modern scams use the very tricks we defend
against.
**Why:** prove the whole robustness story matters in the real world today.
**How:** parse the modern emails; measure recall and obfuscation rates.
**Result:** the 2008 detector catches only **14%** of modern scams (proof it goes stale over
time) — and modern scams **do** use look-alike letters and invisible characters
(≈0% in 2008 → ~2–4% now). So the attacks we defend against are real and growing.

---

## Stage 8 — Evolve it: teach it modern emails
**What:** fed the detector modern examples and tested whether it now catches modern scams.
**Why:** a detector frozen on old mail goes blind; keeping it current is the practical fix.
**How:** added modern scam emails (2023–2025) to the training data and re-tested on held-out
modern scams. To also check false alarms, I paired them with 6,174 modern *normal* emails from
public software mailing lists (Apache projects, 2024).
**The honest catch (measured, not hidden):** scam and mailing-list emails come from different
places, so the tool could "cheat" by spotting the *source* instead of the *scam*. I ran the
same source-check the project already uses: the two sources are **99.5% separable** — so the
trap is real. I therefore lead with a *clean* experiment that avoids it (below) and report the
mixed one only with that caveat.
**Result (real):**
- Old 2008 detector: catches **15%** of modern scams.
- After adding modern scams to training (normal-mail source left unchanged, so no new cheating):
  **98%** — a huge jump.
- It did **not forget** old scams (still **0.998** on the 2008 test), and it did **not** start
  false-alarming on modern normal mail (**0.6%** false alarms).
- **Plain lesson:** a detector isn't "done" — it needs feeding with fresh examples, and doing so
  here fixed almost all of the staleness with no downside.

---

## Stage 9 — Hack the words, not the letters
**What:** smarter attacks that change **words**, not letters: (a) swap important scam words for
plainer synonyms ("verify"→"confirm"); (b) delete the top scam words; (c) **dilution** — bury the
scam inside lots of normal-sounding filler words.
**Why:** we closed the letter-trick door (with a proof), so this is where a clever attacker goes
next. Testing it is the honest next round of the cat-and-mouse game.
**How:** find the model's most important words, then swap / delete / drown them; measure recall.
**Result (real):**
- **Synonyms barely work** (recall stays 0.998): the robust detector shrugs off small word swaps.
- **Deletion** hurts moderately (down to ~0.78 when many words removed).
- **Dilution is the killer**: padding a scam with normal-sounding words crashes recall
  **0.998 → 0.13**. A plain word-counting model just *adds up* signals, so enough "safe" words
  outvote the scam words.
- The **letter-cleaner does nothing** here (0.997) — it's a different kind of attack, as expected.
**The fix:** teach the model these word attacks (train on scam emails with the same tricks
applied). Recall recovers to **1.000**, with no loss on clean mail.
**Honest lesson & future work:** this is another round of the arms race. The deeper point is that
a simple word-counting model is *inherently* vulnerable to dilution; adversarial training patches
it, but the more fundamental answers are structure/length-aware features (a scam drowned in filler
*looks* padded) and the **fusion** with a second, independent detector — exactly the design this
project is one half of.

---

## Stage 10 — Harden and probe even further (latest round)
Four more honest checks — some fixes, some "we tried and here's the truth":

**(a) A fairer fix for dilution — without cheating.** Earlier I beat dilution by *showing* the model
the trick. Here I add a rule that needs **no** knowledge of the attack: count how much *phishing
evidence* is present (the phishing words that appear), in a way that padding can't lower. It helps —
catching diluted scams rises from **13% to 36–66%** depending on how many false alarms you allow — but
it doesn't fully solve it, because short scams don't carry enough evidence. **Honest lesson:** dilution
is a built-in weakness of any word-counting model; the real cures are showing it the attack, or using a
richer model / fusion.

**(b) A harder attack — paraphrase.** I rewrote scams with different words and shuffled sentences. The
detector **held** (still 99.8%) — its word + letter-chunk features absorb rewording. So paraphrase is
*not* an easy way in; dilution stays the one word-trick that works. (A fancier AI-rewriter is the
tougher test — future work.)

**(c) Does it all hold on a different dataset?** I replayed the whole attack-and-fix story on a
**second** email set (SpamAssassin). Same shape: high when clean, cut roughly in half by the attack,
mostly restored by cleaning. So the findings are **real, not a fluke of one dataset**.

**(d) Can the modern retrain be made fully clean?** I tried to remove the "cheating trap" by scrubbing
mailing-list junk from the modern normal emails. It **didn't** help (99% → 99%): the difference is the
*topic/genre* (tech chat vs scams), not formatting. **Honest lesson:** a fully clean modern detector
needs real personal modern email (not public), so the earlier clean experiment stays the trustworthy one.

**(e) Fairer transformer test.** I re-ran the DistilBERT comparison on the **full** test set (5,327
emails) instead of a small sample. Result barely changed: F1 **0.984** (still just below the simple
model's 0.998), and the letter-attack still didn't transfer to it (recall 0.999). So the conclusion
holds on the fair test — the simple model is as good here, and more explainable. (A GPU full-data
training run is still future work.)

---

## Stage 8 — Tidying up: three bugs found by re-reading my own work

Before building the presentation I went back over the whole repo looking for things that
disagreed with each other. I found four, and fixed them. None of them changed the story — but
finding them is part of doing this honestly.

**(a) A fake "balanced" data point.** In the Complement-NB experiment I trained on data ranging
from 5% scams up to 50% scams. But my working data only *has* 35% scams, and the code quietly
capped the request instead of failing. So the "50%" point was secretly re-using the 35% data —
the same numbers twice.
**Why it mattered:** I was claiming "the advantage disappears at balance" while pointing at a
point that was never balanced.
**How I fixed it:** above 35% there aren't enough scam emails to add, so instead I *remove*
normal emails until the mix hits the target. Now 50% really is 50%.
**Result:** the fix actually made the finding *stronger*. At a genuine 50/50 split the two models
score **exactly the same (0.976 vs 0.976)**, instead of the fake gap of 0.976 vs 0.972. That is
precisely what the theory predicts — Complement NB's help is *specifically* an imbalance
correction, so it should vanish when the data is balanced. My mistake had been hiding a clean
confirmation of the paper's claim.

**(b) A re-run could have destroyed a real result.** The fast re-run mode (skip the slow
DistilBERT part) was still *writing* the DistilBERT results file — with the word "skipped" in it.
Anyone following my own README instructions would have silently wiped a real number.
**How I fixed it:** the file is now only written when the model actually runs; otherwise it is
left alone.

**(c) Two documents disagreed with each other.** The readable Markdown copy of my report still
had the *old* conclusion, written before I built the Unicode fix. So one half of the file said
"a real defence would need the full Unicode set and would still lose", while the other half
already showed that using the full Unicode set **wins, with a proof**. The formal LaTeX report was
correct; only the Markdown mirror was stale.
**How I fixed it:** brought the Markdown discussion and conclusion in line with the real report.

**(d) The diagram was out of date.** My pipeline diagram stopped at "adversarial training" — it
was missing the whole second half of the project (the Unicode proof, the word attacks, the drift
test, the retrain). Anyone reading it would think the project was a third of its actual size.
**How I fixed it:** redrew it as three rounds of attack-and-defend, ending where the project
actually ends.

**Also:** the project is now in **Git**, as required. While setting that up I found the ignore
file had a rule that silently excluded the two modern-email data files I *meant* to include
(Git doesn't look inside a folder it has been told to skip), so those are now correctly saved.

---

## Stage 9 — My own idea: a detector that *cannot* be diluted (and what it cost me)

### Where I was before this
By this point I had closed the "letter tricks" front properly. Look-alike letters, leetspeak,
full-width letters, invisible characters — all of those were beaten, and not just beaten but beaten
with a **proof**: after cleaning the text through the official Unicode look-alike table, the detector's
answer *cannot* change, for 100% of the scam emails. That felt solid.

But one attack was still beating me: **dilution**. If an attacker takes a scam email and sprinkles
40 ordinary, innocent-sounding words through it, my detector's catch-rate falls from **99.8% to 13%**.
The reason is simple and a bit embarrassing: my model works by adding up evidence. Scam words push
"scam", normal words push "normal", and it adds them up. So an attacker doesn't need to hide the scam
at all — they just need to bury it under enough boring text to outvote it.

I had already tried one fix (Stage 7a): an "evidence floor" — look only at the scam-ish words that are
present and ignore everything else. It helped, lifting 13% up to somewhere between 36% and 66%
depending on how many false alarms I was willing to accept. But I was never happy with it, for two
honest reasons:
1. it was a **patch bolted onto the outside** of the model — I took the trained model, threw away its
   negative weights by hand, and picked a cut-off by trial and error;
2. it **never actually solved** the attack. 66% at best still means a third of diluted scams get through.

### What I asked myself
Instead of patching a model that *can* be diluted, could I **build a model that cannot be diluted at
all** — and prove it, the same way I proved the Unicode defence?

### The idea, in plain words
Force every word's weight to be **zero or positive** — never negative. In other words, build a model
that is only allowed to collect *evidence of guilt*, and is never allowed to treat a word as evidence
of innocence. Also, only look at *which* words appear, not how many times or in what order.

Then something nice follows automatically. Adding words to an email can only ever *add* evidence of
guilt — because there is no such thing as innocent evidence in this model. **So the score can only go
up, never down.** Padding an email with innocent filler literally cannot make this model change its
mind from "scam" to "safe". That's not a defence that happens to work in my tests; it's a
**guarantee**, true by the way the model is built.

That gives me a *second* proof, covering a *completely different* kind of attack from the first one:
- proof #1 (Stage 5): you cannot beat me by **changing letters**;
- proof #2 (this stage): you cannot beat me by **adding words**.

**Where it comes from:** the trick of forcing weights to be positive to get a guarantee against
"adding stuff" attacks comes from a paper on malware detection (Fleshman et al., 2018). And the attack
I'm defending against turns out to be a classic — the "good word attack" on spam filters (Lowd &
Meek, 2005). I found this out *after* I had already built my dilution attack myself, which was a
useful lesson: my attack wasn't new, but it was real enough that people published it 20 years ago.

### The mistake I made first (worth telling)
My first version **failed the proof**, and figuring out why taught me the most interesting thing in
this stage.

I built it on my usual features, which include **word pairs** ("verify account" counted as one
feature). But if an attacker inserts a word *in between* — "verify **now** account" — then the pair
"verify account" is **destroyed**. That's a piece of guilt evidence *disappearing*, so the score can
go **down** after all, and the guarantee breaks. I measured it: on 500 emails, the score dropped on
**388 of them**. The proof was simply false.

The fix was to use **single words only**, with no word pairs — because then word order doesn't matter,
and inserting text can never destroy anything that was already there. After that change the guarantee
held perfectly. **The lesson:** a proof is only as good as the assumptions underneath it, and mine had
a hidden assumption I hadn't noticed until I tested it.

### The results — including the bad ones

**The good news: the guarantee is real and exact.**

| padding added | scores that went down | scam emails that stayed caught |
|---|---|---|
| +40 words | **0** | **100%** |
| +100 words | **0** | **100%** |
| +300 words | **0** | **100%** |

Zero. Not "almost zero" — zero, exactly as the maths says. And side by side with the old model:

| padding added | old model catches | new certified model catches |
|---|---|---|
| none | 99.8% | 86.9% |
| +10 words | 63.2% | 86.9% |
| +20 words | 36.5% | 86.9% |
| +40 words | **12.9%** | **86.6%** |

The old model falls off a cliff. The new one is a **flat line** — the attack simply does nothing to it.

**The bad news, and it's genuinely bad: it is a much worse detector on normal email.**

| model | F1 | MCC | recall | precision |
|---|---|---|---|---|
| my main model (union) | **0.998** | 0.997 | 0.998 | 0.998 |
| certified monotone model | **0.912** | 0.869 | 0.873 | 0.953 |

That is a big drop, and I'm not going to dress it up. It makes sense: I took away two of the model's
abilities. It can no longer use "this word suggests the email is *fine*" to let a borderline email off,
and it can no longer use word order. I paid for the guarantee with accuracy. **This model is not a
replacement for my main detector** and I won't present it as one.

**One more honest crack: my own truncation breaks the proof slightly.** I cut every email at 2000
characters for speed. So a really determined attacker can pad an email so much that real scam evidence
gets pushed *past* the cut-off and is never seen. Then the guarantee no longer applies — measured, it
drops from 100% to about **99.6%**. Small, but it's a genuine hole caused by my own pipeline, so I
report it separately instead of quietly testing without the cut-off.

### What I actually recommend, given all that
Don't replace — **combine**. Run both models and flag an email if *either* one says scam. Then:
- on clean email you keep the strong model's quality;
- under heavy dilution the catch-rate has a **certified floor of about 87% instead of 13%**;
- the price is more false alarms on normal mail: roughly **0.1% → 2%**.

Whether 2% false alarms is worth it is an **operational decision, not a maths one** — it depends how
much an analyst's time costs versus a missed scam. My job is to state the trade honestly, not to
pretend it's free.

### What this still does *not* fix
The guarantee only covers **adding** text. An attacker who **deletes** the scam wording, or **rewrites**
it in different words, still gets through — and in fact a "guilt evidence only" model is *especially*
weak against deletion, because it has no innocent evidence to fall back on. So the honest summary is:
two kinds of attack are now closed by proof, and the attacker has been pushed toward rewriting the
meaning — which is much more expensive for them than padding was, but it is not nothing.

---

## Stage 10 — Letting people help the detector (and finding out how to break that too)

### Where I was before this
My detector was a **frozen object**. It learned once, in 2008, and that was that. And I had
already proved it was badly out of date: it catches only **14%** of real 2023–25 scams.

But a detector in a real company doesn't sit alone — it sits next to **people**. And there is one
signal every mail system already collects and mine was ignoring completely: someone clicking
**"report this email as suspicious"**.

This matters because targeted attacks aren't sent to one person. An attacker who goes after a
company sends the *same* scam, with small variations, to lots of employees. So the question is:

> If **one** employee reports **one** scam email, can the system automatically catch the same scam
> in everyone else's inbox?

### First: do these "campaigns" actually exist, or am I imagining them?
I checked before building anything. Grouping the 1,303 modern scam emails by how similar their text
is, I found **93 groups of 3 or more** near-identical emails. So yes — real repeated campaigns, in
real data.

**A mistake I caught here.** My first grouping method produced a "campaign" of 139 emails, which
looked great until I measured how similar those 139 actually were to each other: only about **0.50**
on a 0–1 scale. It wasn't one campaign — it was a **chain** (A is like B, B is like C, but A is
nothing like C). I switched to a grouping method that can't chain, and got 93 genuinely tight groups
(internal similarity **0.88**). Fewer campaigns, but real ones.

### The two ways to use a report
- **(A) Similarity matching:** if a new email looks very like a reported one, flag it. No retraining,
  instant.
- **(B) Retraining:** add the reported email as a new training example and refit the model.

### The two failed attempts that taught me the most
To use method (A) I need a cut-off: *how* similar counts as "similar enough"? My first two attempts
set that cut-off using **normal (non-scam) email** — a standard approach, aiming for a chosen
false-alarm rate.

Both gave me a fantastic-looking answer: **100% of the campaign caught.** And both were **wrong**.

The giveaway was a control I had built in on purpose: emails with **no** near-duplicates at all.
Those should get *no* benefit from this method — nothing to match them to. But they were being
"detected" at **92–94%**. That's impossible if the method really works by matching campaigns.

The reason turned out to be something my own project had already documented: my normal-email source
(Apache developer mailing lists) is about **99% distinguishable** from scam email just by topic and
style. So a cut-off tuned on it doesn't measure "is this the same campaign?" — it measures "is this a
scam-ish email at all?" I was accidentally measuring the wrong thing, in a way that flattered me.

**The fix:** set the cut-off using *other scam campaigns* instead of normal email — so it has to
distinguish **this campaign** from **other campaigns**, which is the actual question. Then report the
false-alarm rate on normal mail separately, instead of using it to set the cut-off.

### The test that decides whether the result is real
Catching lots of emails proves nothing on its own — feeding a 2008 model *any* modern examples helps.
So I ran the experiment twice:
- **matched:** the campaign's own report is available
- **blind:** that one report is removed, all the others kept

Same cut-off, same reports, one difference. Any gap is caused by campaign membership and nothing else.

### The results

| | caught |
|---|---|
| old model alone | **15%** |
| **with a matching report (A)** | **100%** |
| with reports from *other* campaigns only (blind) | **33%** |
| retraining on the reports (B) | **99%** |
| control: emails with no near-duplicates | 15% |

**And false alarms didn't rise at all** — the false-alarm rate on normal mail stayed at the model's
own level (about 0.1%). Normal email just doesn't look like a specific reported scam.

**Read honestly:** the jump from 15% to 100% is real, but the *campaign* effect is the gap between
**100% and 33%** — not the whole jump. The blind arm still beats the baseline because any modern
example helps an outdated model. Quoting only "15% → 100%" would oversell what the mechanism does.

### Then I attacked my own idea

| attack on later campaign emails | how many escape |
|---|---|
| nothing | 0% |
| swap words for synonyms | 0% |
| bury it in 40 filler words | 15% |
| bury it in 80 filler words | 31% |
| **look-alike letters (homoglyphs)** | **100%** |
| **the same, after Unicode cleaning** | **3%** |

**The letter trick switches the whole system off.** 100% escape — because if you replace the letters,
the text no longer *looks* similar to anything, so there is nothing to match. And the fix is the
Unicode cleaner I already built in Stage 5: it brings escapes back down to 3%.

That's my favourite result in this stage. It means my earlier defence isn't just a nice extra — it is
a **requirement** for this feature to work at all. A company that added a "report" button without
cleaning the text first would have a safety feature an attacker could disable with invisible characters.

### And the ugly side: the report button can be abused
What if people report *normal* emails as scams — a malicious employee, or just mistakes?

| share of reports that are false | false alarms on normal mail |
|---|---|
| 0% | 0.1% |
| 10% | 0.6% |
| 50% | 1.2% |

So the damage is real (about **5× worse** at 10% bad reports) but **bounded** — one bad example only
affects mail that resembles it. Still, the honest conclusion is that a "report" button is not just a
feature, it's an **attack surface**: an unchecked feedback loop lets users, or an attacker pretending
to be one, degrade the filter for everyone.

### What this is *not*
I simulated a company using a public email archive. The "campaigns" are my own grouping, not labels
someone gave me. The reports are simulated and (except in the poisoning test) always correct. So this
shows the **mechanism works and how it fails** — it is not a measurement of a real deployed system.

---

## Stage 11 — The last kind of attack: get an AI to reword the scam

### Where I was before this
I had closed two whole kinds of attack **with proofs**: you can't beat the detector by changing
*letters* (Stage 5), and you can't beat it by *adding* words (Stage 9). But both proofs leave the
same door open on purpose: what if the attacker **rewrites the scam in different words**?

I had already tried a weak version of this back in Stage 9b — swap a few known words for synonyms,
shuffle the sentences. It did nothing (the detector still caught 99.8%). But that's a toy. The real
modern threat is that an attacker can now ask an **AI to rewrite every scam email** — fluently, in
seconds, for pennies. In 2008 that was impossible; today it's a button.

### The idea
Use one AI to **rewrite** the scam emails (the attacker), and my detector to try to catch the
rewrites. Simple enough — but it has a trap.

### The trap, and how I avoided it
If the AI rewrites an email so much that it's **no longer a scam**, then "the detector missed it"
proves nothing — I didn't disguise the attack, I *deleted* it. So I need something to check that each
rewrite is *still* a scam before I count it.

My solution: a **second, separate AI as a judge**. One AI rewrites; a different one reads each rewrite
and rules "is this still phishing, yes or no?" Only the rewrites the judge accepts count as real
attacks. This is a known technique — "AI as a judge" — used here to check labels, not to detect.

I kept them **independent**: the rewriter never saw the judge. (Honest limitation: they're the same
*kind* of AI, so it's partial independence, not two totally different systems — I say so.)

### How I kept it reproducible
AI answers change every time you ask — that would break my "runs the same every time" rule. So I did
it **once**, saved the rewrites and the judge's verdicts to files, and committed them
(`data/llm_paraphrase/`). The notebook now just *reads* those files and re-scores them with the
detector, which is fully repeatable. Same trick I used for the modern-email data.

### The results (30 scam emails; judge kept 24 as still-scams, dropped 6)

| | my main model catches | the "provable" model catches |
|---|---|---|
| original scams | 100% | 75% |
| after AI rewrote them | **100%** | **71%** |

**Two honest lessons:**
- **My main detector shrugged it off** — still 100%, even against a real AI rewrite (not a synonym
  swap). Its mix of word *and* letter-chunk features is just too redundant to fool by rewording. So
  the reassuring Stage 9b result holds up under a much tougher test.
- **My "provable" model is the one that bleeds** (75% → 71%). And that's the *honest* result I wanted:
  the model I proved you can't beat by *adding* words is exactly the one most exposed to *changing*
  words — because it's only allowed to collect guilt-words, and rewording swaps those guilt-words out.
  A proof is only as strong as the attack it's about.

So the whole story closes into one line: **letters — proved shut. Added words — proved shut. Reworded
words — my strong model resists, my provable model doesn't, and doing it at scale is the real open
problem.**

### What this is *not*
Only 30 emails (24 after judging) — this is a **taster, not a full measurement**. The judge is an AI,
not a human, so ideally I'd double-check its 24 calls by hand. And attacker and judge are the same
family of AI. All stated. The point is the *shape* of the result and that I built the attack-check-
detect pipeline honestly — not the exact percentages.

---

## Stage 12 — Grounding it in the course, and a deeper EDA

Two things here: I made the data-exploration part more thorough, and I tied every important
decision back to the methods we were actually taught, so it's clear I chose them on purpose.

**A deeper EDA.** Before I only showed that scam emails *correlate* with being shorter. Now I
also:
- measured the **shape** of each simple feature (skewness = how lopsided, kurtosis = how
  heavy-tailed). It turns out the ratio features are very lopsided and heavy-tailed — which is
  the real reason I used a **rank-based** correlation (Spearman) and the **median**, not the
  average. That wasn't an arbitrary choice; the data's shape forces it.
- **tested** the "scam emails are shorter" claim with a proper statistical test (Mann–Whitney U),
  instead of just eyeballing a chart. It's real (p is essentially zero).
- **measured redundancy** with the VIF number: email length and word-count score ~21 (anything
  over 10 means "these two say the same thing"), which is exactly why I keep them as *diagnostics*
  and don't feed both to the model.
- added a **box-plot** figure (F13) showing the three most telling features split by scam vs
  normal, so the story is visual as well as numeric.
- **checked the dates for a classic trap.** The EDA lecture warns that a *missing* date quietly
  filled in with a 0 turns into "1 January 1970" (the computer's zero-point) and then sneaks into
  a time chart as if it were real. I tested for it directly (`scripts/check_date_trap.py`): there
  are **none** — zero 1970 dates in all 34,342 emails. The tiny 0.3% of odd dates are genuinely
  *spoofed* email headers (years like 2086), which the temporal chart already leaves out. So the
  time analysis is clean — I looked, and the trap simply isn't there.

**Connecting to what we learned.** I went back over the course material and made the links
explicit — not as decoration, but where it justifies a real choice:
- the **accuracy paradox**, F-scores, MCC and AUC (why accuracy alone lies on imbalanced data);
- **stratified** train/test split with a fixed seed, and the **no-leakage** rule ("train is
  always before test; only use what you'd know at prediction time") — which is exactly why my
  temporal-drift and campaign experiments are split the careful way they are;
- the standard **attack taxonomy** — my letter/word attacks are *evasion*, the false-report
  problem is *poisoning*, the smart attacker is *targeted / grey-box*, and I test *transfer* to
  BERT — so a marker can see where each piece fits;
- **normalising text / converting homoglyphs** is the textbook defence for exactly the attack I
  run, and **adversarial training** is the other one;
- **explainability levels** (model-level = the weights, instance-level = one email's reasons),
  and why a simple linear model gives both directly, with no need for LIME/SHAP approximations.

I also named, honestly, the relevant ideas I did **not** do (model-stealing/extraction, and an
anomaly-detector that flags obfuscated mail as suspicious) as future work.

## The one-line takeaway for the talk
> A simple detector can look perfect and still be fragile. I attacked it, fixed it, attacked
> it smarter, and fixed it with a *proof* — twice, for two different kinds of attack — and where
> a fix falls short, or costs accuracy, I say so. The result isn't a "solved" detector but an
> *honestly hardened* one, plus clear evidence for why the real system needs a second,
> independent detector alongside it.

*(The two proofs: you cannot beat it by changing letters (Stage 5), and you cannot beat it by
adding words (Stage 9) — the second one bought at a real cost in accuracy, which is stated.)*
