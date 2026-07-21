"""Date-quality check: is the CEAS `date` header contaminated by the classic
Unix-epoch / imputed-zero trap?

A well-known EDA pitfall (see docs/DECISIONS_AND_PITFALLS.md): when missing dates
are silently imputed to 0 they parse to **1970-01-01** (Unix time 0) and then
masquerade as a real timestamp in any temporal plot. This script checks for that
directly, on both the harmonised CEAS corpus and the frozen working set, and prints
the breakdown of any dates that fall outside the plausible 2007-2009 window.

Result (run 2026-07-21): **clean** -- 0 epoch/1970 dates in 34,342 harmonised
emails (and 0 in the 26,633-row working set). The ~0.3% of dates outside 2007-2009
are genuine malformed/spoofed RFC-822 headers (real strings like
'Mon, 05 Aug 2086 21:34:28 -0300'), not imputed nulls -- and the notebook's temporal
analysis (Section 2) already restricts the plot to 2007-2009, so they are excluded.

Run:  py -3 scripts/check_date_trap.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C, data as D  # noqa: E402


def report(name: str, dates_raw: pd.Series) -> None:
    print(f"\n===== {name}  (n={len(dates_raw):,}) =====")
    s = dates_raw.astype(str)
    empty = (s.str.strip() == "") | s.str.lower().isin(["nan", "nat", "none"])
    dts = pd.to_datetime(dates_raw, errors="coerce", utc=True)
    yr = dts.dt.year
    is_1970 = (yr == 1970)
    is_epoch_day = (dts.dt.strftime("%Y-%m-%d") == "1970-01-01")
    plausible = yr.between(2007, 2009)
    impl = dts.notna() & ~plausible

    print(f"empty/blank date strings         : {int(empty.sum())} ({empty.mean():.2%})")
    print(f"parseable to a date              : {int(dts.notna().sum())} ({dts.notna().mean():.2%})")
    print(f"year == 1970 (epoch / imputed?)  : {int(is_1970.sum())}")
    print(f"exactly 1970-01-01               : {int(is_epoch_day.sum())}")
    print(f"plausible 2007-2009              : {int(plausible.sum())} ({plausible.mean():.2%})")
    print(f"parsed but implausible (off 07-09): {int(impl.sum())} ({impl.mean():.2%})")
    if impl.sum():
        print("  implausible-year breakdown:")
        for y, c in yr[impl].value_counts().sort_index().items():
            tag = "   <-- Unix epoch / imputed-zero" if y == 1970 else ""
            print(f"    {int(y)}: {int(c)}{tag}")
        print("  sample raw implausible strings:")
        for v in dates_raw[impl].astype(str).head(5).tolist():
            print(f"    {v!r}")

    verdict = "CLEAN (no epoch/imputed dates)" if is_1970.sum() == 0 else "CONTAMINATED"
    print(f"  --> epoch-trap verdict: {verdict}")


def main() -> None:
    df = D.load_all([C.PRIMARY_CORPUS])
    report("CEAS_08 (harmonised, deduped) - raw date column", df["date"])

    wp = C.DATA_DIR / "emails_working.csv"
    if wp.exists():
        w = pd.read_csv(wp, dtype=str, keep_default_na=False, na_filter=False)
        if "date" in w.columns:
            report("emails_working.csv - date column", w["date"])


if __name__ == "__main__":
    main()
