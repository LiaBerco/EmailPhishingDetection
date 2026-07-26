"""Central configuration — paths, column names, seeds, feature grids.

Everything that a reader might want to change lives here (and is also mirrored in
the CONFIG cell at the top of the notebook). Keeping it in one place is what the
brief asked for: "make dataset path and column names configurable at the top".
"""
from pathlib import Path

# ---- reproducibility -------------------------------------------------------
RANDOM_SEED = 42

# ---- paths -----------------------------------------------------------------
# Project root = the folder that contains this src/ package.
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "kaggle_phishing"
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (DATA_DIR, ARTIFACTS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- dataset schema --------------------------------------------------------
# The Kaggle compilation ships one CSV per source corpus. They do not all share
# the same columns, so we harmonise down to: source, date?, subject, body, label.
LABEL_COL = "label"          # 1 = phishing/spam, 0 = legitimate
TEXT_COLS = ["subject", "body"]

# name -> role in this project (see docs/PROJECT_PLAN.md section 2)
CORPORA = {
    "CEAS_08.csv":        {"role": "primary",  "mixed_class": True},   # main testbed
    "SpamAssasin.csv":    {"role": "secondary", "mixed_class": True},
    "Nazario.csv":        {"role": "ood_phish", "mixed_class": False},
    "Nigerian_Fraud.csv": {"role": "ood_phish", "mixed_class": False},
    "Enron.csv":          {"role": "ood_ham",   "mixed_class": True},
    "Ling.csv":           {"role": "ood_ham",   "mixed_class": True},
    # phishing_email.csv is the merged file (no source/date) -> deliberately unused.
}

# The corpus used for the main in-corpus experiments (E1-E8). It is the only
# corpus that contains BOTH classes from a single provenance, so training on it
# cannot be "cheating" by learning which corpus an email came from.
PRIMARY_CORPUS = "CEAS_08.csv"

# ---- split -----------------------------------------------------------------
TEST_SIZE = 0.20
VAL_SIZE = 0.20              # of the whole -> 60/20/20 train/val/test

# ---- realistic base rate ---------------------------------------------------
# Real inboxes are mostly legitimate mail. CEAS_08 is 56% phishing (phishing is
# the majority), which is unrealistic and also hides evasion attacks: a model that
# loses its signal falls back to the majority class, so if that class is "phishing"
# the attack looks harmless. We therefore subsample phishing to a legit-majority
# prior, which is both more realistic and makes the robustness test meaningful.
PHISH_FRACTION = 0.35       # target phishing share of the working dataset

# ---- text / feature settings ----------------------------------------------
MAX_CHARS = 2000            # truncate very long bodies (bounds attack/vectorise time)
WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (3, 5)
WORD_MAX_FEATURES = 30000
CHAR_MAX_FEATURES = 30000
MIN_DF = 3

# ---- evaluation ------------------------------------------------------------
BOOTSTRAP_N = 1000          # resamples for confidence intervals
DEFAULT_THRESHOLD = 0.50
# cost of a miss vs a false alarm (FN >> FP for phishing) -> used to pick an
# operating point in the cost-sensitive evaluation.
COST_FN = 5.0
COST_FP = 1.0
