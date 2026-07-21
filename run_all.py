"""Reproduce every result: execute the notebook end-to-end and regenerate all
tables (results/tables/) and figures (results/figures/) from the raw data.

Usage:
    python run_all.py                # run everything, including DistilBERT
    PHISH_RUN_BERT=0 python run_all.py   # skip the (slow, CPU) DistilBERT arm

This is a thin wrapper around `jupyter nbconvert --execute` so that the notebook
stays the single source of truth for the pipeline.
"""
import subprocess
import sys
from pathlib import Path

NB = Path(__file__).parent / "notebooks" / "student_A_content_detector.ipynb"


def main():
    if not NB.exists():
        sys.exit(f"notebook not found: {NB}")
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook", "--execute", "--inplace",
        "--ExecutePreprocessor.timeout=2400",
        str(NB),
    ]
    print("running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
