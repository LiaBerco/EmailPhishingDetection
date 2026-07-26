# LLM paraphrase attack — frozen artefacts

These three files are the **frozen outputs** of a one-off LLM-based paraphrase experiment
(notebook §32e). Unlike the rest of the project, LLM outputs are **not reproducible from a
seed** — they vary between calls — so the outputs are committed and the notebook *reads* them,
exactly as the modern Nazario/Apache data is committed rather than re-fetched. Re-running the
notebook re-scores these frozen texts with the detector (which *is* deterministic); it does not
re-generate them.

| file | what it is | produced by |
|---|---|---|
| `sample.json` | 30 phishing e-mails drawn from the **test** split (seed 42) | `src.config` split, deterministic |
| `rewrites.json` | a meaning-preserving paraphrase of each (`id` → `rewrite`) | an LLM agent (the "attacker") |
| `judge.json` | for each rewrite, whether it is **still phishing** (`phishing`/`legit`/`unsure`) | a *separate* LLM agent (the "judge") |

## Provenance and honest caveats
- **Generated on 2026-07-20** by two independent Claude agent instances: one rewrote, one judged.
  The rewriter never saw the judge and vice-versa, so the validity label is not the attacker
  grading its own work — but both are the **same model family**, so this is *partial*, not true
  cross-model, independence. State this wherever the result is quoted.
- **The judge is not ground truth.** It is a second model with its own biases; a human spot-check
  of the 24 "phishing" verdicts is the proper validation and is left as a small follow-up.
- **N = 30 (24 valid after the filter).** This is a proof-of-concept probe, not a headline number.
- The paraphrase is a *pure semantic rewrite* — no homoglyphs, no padding — so it isolates the
  "reword the scam" attack class that both project certificates (§21 UTS-39, §32b monotone)
  explicitly exclude.
