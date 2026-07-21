"""Parse recent Nazario phishing mbox files (2023-2025) into a harmonised CSV.

The raw mbox files are downloaded from Jose Nazario's live archive:
    https://monkey.org/~jose/phishing/phishing-2023  (and -2024, -2025)
Save them to data/raw/nazario_recent/phishing-<year>.mbox, then run this script.
It extracts the visible text (prefers text/plain, else HTML with tags stripped),
strips URLs and cleans exactly as the main pipeline does (src/data.py), and writes
    data/raw/nazario_recent/nazario_recent.csv
which the notebook loads for the temporal-drift / real-obfuscation experiments.

This is the *modern* phishing signal (2023-2025) used to test whether a 2008-trained
detector still works and whether real phishing uses the obfuscations we defend against.
"""
from __future__ import annotations
import sys, mailbox, re
from email.header import decode_header, make_header
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D

RECENT_DIR = ROOT / "data" / "raw" / "nazario_recent"
YEARS = (2023, 2024, 2025)
_TAG = re.compile(r"<[^>]+>")


def _decode_header(s: str) -> str:
    try:
        return str(make_header(decode_header(s or "")))
    except Exception:
        return s or ""


def _body_text(msg) -> str:
    """Best-effort visible text: prefer text/plain, else HTML with tags stripped."""
    plain, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct not in ("text/plain", "text/html"):
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                txt = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            except Exception:
                continue
            if ct == "text/plain":
                plain += " " + txt
            else:
                html += " " + txt
    else:
        try:
            payload = msg.get_payload(decode=True)
            txt = payload.decode(msg.get_content_charset() or "utf-8", errors="replace") if payload else str(msg.get_payload())
        except Exception:
            txt = ""
        (html := txt) if msg.get_content_type() == "text/html" else (plain := txt)
    return plain if plain.strip() else _TAG.sub(" ", html)


def main() -> None:
    rows = []
    for y in YEARS:
        path = RECENT_DIR / f"phishing-{y}.mbox"
        if not path.exists():
            print(f"  (skip) missing {path}")
            continue
        for msg in mailbox.mbox(str(path)):
            subj = D.clean_text(D.strip_urls(_decode_header(msg.get("subject", ""))))
            body = D.clean_text(D.strip_urls(_body_text(msg)))
            if len(subj) + len(body) == 0:
                continue
            rows.append({"source": "Nazario_recent", "year": y, "date": msg.get("date", ""),
                         "subject": subj, "body": body, "label": 1})
    modern = pd.DataFrame(rows)
    modern["text"] = D.make_text(modern)
    modern = modern[modern["text"].str.len() > 0].reset_index(drop=True)
    out = RECENT_DIR / "nazario_recent.csv"
    modern.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out} with {len(modern)} emails  by year: {modern.year.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
