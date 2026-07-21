# Insert an "Analysis" markdown cell after each section's results.
# Anchor-based: find the code cell that contains a unique substring, insert the
# analysis right after it. Only adds markdown cells; never touches code or outputs.
import nbformat as nbf

NB = r"C:\GitHub\Data_Science\PhishFusion\notebooks\student_A_content_detector.ipynb"
nb = nbf.read(NB, as_version=4)
H = "### \U0001F50E Analysis of the results\n\n"

# (anchor substring in a code cell) -> analysis markdown. Order-of-magnitude, not
# exact numbers, so it stays valid if the notebook is re-run.
ITEMS = [
("DATA_SHA = D.sha256_of", H +
"The rebalancing did what we wanted: legitimate mail is now the majority and phishing the minority, "
"which mirrors a real inbox far better than the raw corpus (where phishing was actually the larger "
"class). Two consequences carry through the whole notebook. First, because we kept a *single* corpus, "
"any score the model earns has to come from phishing-ness, not from telling corpora apart. Second, the "
"phishing share is a **knob, not a fact of nature** — so the headline metrics should always be read "
"together with this base rate; the same model on a differently balanced set would move the numbers "
"even though the model itself is unchanged."),

("F2_discriminative", H +
"The class split is uneven but not extreme — the minority class is still well represented — so ordinary "
"metrics remain meaningful as long as we do not lean on accuracy alone. The two length distributions "
"overlap heavily, which already says that *length by itself* is a weak signal; the discriminating power "
"has to come from **which words appear**. The non-ASCII panel is the one to remember: clean mail is "
"almost entirely plain ASCII, so an email suddenly full of exotic look-alike characters is anomalous — "
"exactly the surface the homoglyph attack exploits and the normalisation defence folds away. Finally, "
"the most discriminative tokens match intuition (finance/urgency words lean phishing, ordinary "
"conversational words lean legit), a reassuring sign that the labels are trustworthy."),

("T9_correlation", H +
"The clearest monotonic signal is **length**: in this corpus phishing emails tend to be *shorter* than "
"legitimate ones (both length and word-count correlate negatively with the label), which matches "
"intuition --- a scam is often a short, urgent message, whereas legitimate mailing-list traffic is long "
"and threaded. The character-ratio features carry only weak association on their own. But note the "
"ceiling: even the strongest meta-feature is a *moderate* correlation, nowhere near the near-perfect "
"separation the TF-IDF model reaches --- confirming that the real signal lives in **which words appear**, "
"not in coarse surface statistics. The matrix also exposes clear **redundancy**: length and word-count "
"move together almost perfectly, so feeding both to a model would split their importance across "
"near-duplicate features and hurt interpretability. That is one reason we keep these meta-features as "
"diagnostics rather than model inputs and let the sparse lexical features do the work."),

("corpus-ID accuracy", H +
"The probe reaches almost perfect accuracy at telling the corpora apart, against a one-in-three chance "
"baseline — i.e. the corpora are essentially separable on writing style alone. The lesson is blunt: had "
"we glued phishing from one corpus onto legitimate mail from another, a classifier could look brilliant "
"while learning nothing about phishing, only about **provenance**. This one number turns *'we should "
"avoid the confound'* from an assertion into evidence, and is the whole justification for training and "
"testing inside a single mixed-class corpus."),

("tr_idx, tmp_idx = tts", H +
"There is little to interpret in the split numbers beyond confirming that all three parts kept the same "
"class ratio (stratification worked). The important point is procedural: **validation** carries every "
"tuning and model-selection decision, and the **test** set is opened exactly once, at the very end. "
"Every headline metric therefore reflects data the model was never optimised on, which is what makes it "
"an honest estimate."),

("Xtr[kind] = v.fit_transform", H +
"Each configuration produces a high-dimensional sparse space (on the order of tens of thousands of "
"features). The word view and the character view encode different *kinds* of information: word features "
"know **vocabulary**, character features know **shape**. On clean text this distinction barely matters — "
"but it becomes the entire story under attack, because a mangled word still shares letter-chunks with "
"the original, so the character view is the one that survives obfuscation."),

("T1_baselines", H +
"The majority classifier is the accuracy paradox in one line: it posts a respectable accuracy while "
"catching **zero** phishing — a detector that detects nothing, with recall and MCC pinned at the floor. "
"The keyword list does only marginally better, and its correlation with the truth is essentially noise. "
"The takeaway is that accuracy on its own is meaningless here; from now on we read recall, F1 and MCC, "
"and we expect any real model to clear these baselines by a wide margin."),

("T2_feature_ablation", H +
"All three feature sets are near-perfect on clean text, and the differences between them are within (or "
"close to) the confidence intervals — so we should **not** over-read which one is 'best' by a thousandth "
"of an F1. The honest reading is that on clean, single-corpus phishing the lexical signal is so strong "
"that the choice of representation hardly matters. That is a *clean-data* statement only: the three "
"feature sets will separate dramatically once we attack them, which is precisely why we build all three "
"now."),

("F3_confusion_clean", H +
"The linear model and the two Naive Bayes variants are all strong, but Logistic Regression is clearly "
"ahead, and McNemar's test says that gap is **real** rather than sampling luck (a very small p-value). "
"Complement NB edges plain Multinomial NB — the direction the literature predicts, which we probe "
"properly in the next section. The confusion matrix shows errors in the low single-digit percent range "
"in *both* directions, so the model is not quietly trading recall for precision; it is simply accurate "
"on clean mail. We keep Logistic Regression as the working model because it is both the strongest here "
"and the one we can later read off as explanations."),

("T3b_imbalance_sweep", H +
"This is a small replication of a classical claim, and it holds. When phishing is common the two Naive "
"Bayes variants are practically indistinguishable; as we squeeze the phishing class down to a small "
"minority, **Complement NB pulls clearly ahead** of Multinomial NB. In other words the advantage of "
"Complement NB is not a fixed bonus — it *grows with the imbalance*, exactly the mechanism the original "
"paper describes. A useful corollary: citing Complement NB is only justified when the data is genuinely "
"skewed; on balanced data the choice barely matters."),

("best C =", H +
"The setting chosen on validation transfers to the test set with almost no gap between the two — the "
"signature of a model that is **generalising rather than memorising**. If it were overfitting we would "
"see validation flatter itself while test lagged behind. Because the search only ever touched validation "
"data, the reported test number stays an honest estimate."),

("F4_feature_weights", H +
"Two levels of explanation, both faithful to the linear model. The **global** view (the weight chart) "
"shows the tokens that carry the most pull overall — urgency, credentials and money on the phishing "
"side, ordinary conversational words on the legit side. The **local** view is the stronger one: for a "
"single email we can print the *exact* per-token contribution (coefficient times TF-IDF), which sums to "
"the model's log-odds — so this is the model's own arithmetic, not a keyword guess or a LIME-style "
"approximation. The catch, and the answer to our fourth research question, is that this legibility lives "
"entirely in the **word** features; the character n-grams we rely on for robustness have no readable "
"meaning individually. The more we lean on the robust representation, the less of this explanation we "
"get — **robustness and interpretability pull in opposite directions**."),

("T4_threshold", H +
"Because the model is already near-perfect, moving the threshold barely changes the outcome — but the "
"*direction* is the point. Weighting a miss more heavily than a false alarm pushes the chosen threshold "
"below the default 0.5, buying a little extra recall at a tiny precision cost. On a weaker or more "
"adversarial model this lever matters far more; here it mainly shows that the "
"**false-negative-is-worse-than-false-positive** logic of a SOC can be encoded explicitly instead of "
"left at an arbitrary default. The ROC and PR curves hugging the top corner just confirm there is little "
"clean-data headroom left to win."),

("F5_robustness_bars", H +
"This is the heart of the project, and the **shape** of the result matters more than any single figure. "
"On clean text every model catches essentially all phishing. The moment we apply an *imperceptible* "
"homoglyph attack, the **word-only** model loses a large chunk of its recall — on the order of a quarter "
"to a third of phishing now slips through, with no visible change to a human reader. That is the "
"vulnerability of a filter that trusts exact word matches.\n\n"
"Two defences answer it, in different ways. **Character n-grams** never fall far in the first place — the "
"union model stays high throughout — because sub-word chunks survive letter swaps. **Normalisation** "
"repairs the word model almost completely, folding the look-alikes back to plain text before the model "
"ever sees them.\n\n"
"The two honest stress-tests are what stop this from being a rigged demo. Against a **held-out** attack "
"family the defence was never tuned for, normalisation *still* recovers the model — not because we "
"hand-listed those characters, but because its generic Unicode folding catches them. That is genuine "
"generalisation. But the **adaptive** attacker, who deliberately uses look-alikes the normaliser does "
"not know, keeps recall meaningfully below the clean level even after normalisation. So the conclusion "
"is nuanced rather than triumphant: character features and normalisation together remove *most* of the "
"risk, but a defence-aware attacker always keeps a residual foothold — **no silver bullet**, only a "
"raised cost of attack."),

("F8_adversarial_training", H +
"Two defences, two different failure modes — and they line up to cover each other. **Adversarial "
"training** is excellent on the exact family it was trained on (the homoglyph attack, which it now "
"catches almost perfectly) but it **generalises poorly**: on a held-out attack family it never saw, its "
"recovery is only partial, because it memorised specific look-alike characters rather than the general "
"idea of obfuscation. **Normalisation** is the mirror image — it was not tuned for that held-out family "
"either, yet it recovers there, because its Unicode folding is a general rule rather than a learned "
"pattern, while it leaves a little more on the table against the adaptive attacker. Put together, the "
"two reach high recall across every family, each covering the other's blind spot. This is the standard "
"caution about adversarial training in miniature: it hardens a model against the perturbation you "
"trained on and little else, so it belongs **alongside** a principled input defence, not instead of "
"one."),

("T7_distilbert", H +
"Two things stand out. First, the transformer does **not** beat the tuned TF-IDF model on clean text — "
"it lands slightly behind. That is the expected outcome when the signal is 'which words appear': "
"bag-of-words already captures it, so contextual modelling has little left to add, and its advantage "
"would only surface on text where word *order* carries meaning. Second, and more interesting for a "
"security project, the homoglyph attack barely transfers to DistilBERT — its recall on phishing stays "
"essentially unchanged under an attack that badly hurt the word model. The reason is structural: the "
"transformer's sub-word tokeniser shatters an obfuscated word into pieces, the same reason our character "
"n-grams were robust. So an attack designed against a lexical model does not automatically carry over to "
"a differently-tokenised model — a concrete example of **limited attack transferability**."),

("T6_cross_corpus", H +
"This is the sobering number and the most important caveat in the project. Everything above is "
"*in-corpus* — train and test drawn from the same source. The moment we test on genuinely different "
"corpora, performance falls off a cliff: F1 drops by well over half, driven by a **recall collapse**, "
"because the model has effectively learned one corpus's vocabulary and dialect of phishing rather than "
"phishing in general. This is distribution shift, not a bug, and it is exactly why the abstract quotes "
"this number rather than the flattering in-corpus one. It is also the strongest argument for the "
"project's larger design: a single text model is not enough on its own, which is why it is meant to be "
"**fused** with an independent, non-text detector."),
]

# find insertion positions (index of the code cell containing each anchor), then
# insert from the bottom up so earlier indices stay valid.
insertions = []
for anchor, text in ITEMS:
    pos = next((i for i, c in enumerate(nb.cells)
                if c.cell_type == "code" and anchor in "".join(c.source)), None)
    if pos is None:
        raise SystemExit(f"anchor not found: {anchor!r}")
    insertions.append((pos, text))
for pos, text in sorted(insertions, reverse=True):
    nb.cells.insert(pos + 1, nbf.v4.new_markdown_cell(text))

nbf.write(nb, NB)
print(f"inserted {len(insertions)} analysis cells; notebook now has {len(nb.cells)} cells")
