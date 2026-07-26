"""Parse modern legitimate ("ham") email from public Apache mailing-list archives
into a harmonised CSV, to pair with the modern phishing set for the evolve/retrain
experiment (Stage 8).

Download the raw mbox files first, e.g.:
    https://lists.apache.org/api/mbox.lua?list=users@maven.apache.org&date=2024-06
saved to data/raw/modern_ham/<list>_<YYYY-MM>.mbox , then run this script. It extracts
the visible text, strips URLs and cleans exactly as src/data.py, de-duplicates, and writes
    data/raw/modern_ham/modern_ham.csv

NOTE on the confound: mailing-list mail is a *different source* from the Nazario phishing,
so a detector trained on this mix can partly learn provenance rather than phishing-ness.
The notebook measures this directly (a source-separability probe) and treats the
ham-source-unchanged experiment as the clean result. See docs/IMPROVEMENTS_STORY.md.
"""
from __future__ import annotations
import sys, mailbox, re
from email.header import decode_header, make_header
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as D

HAM_DIR = ROOT / "data" / "raw" / "modern_ham"
_TAG = re.compile(r"<[^>]+>")


def _decode_header(s: str) -> str:
    try:
        return str(make_header(decode_header(s or "")))
    except Exception:
        return s or ""


def _body_text(msg) -> str:
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
    for mb in sorted(HAM_DIR.glob("*.mbox")):
        for msg in mailbox.mbox(str(mb)):
            subj = D.clean_text(D.strip_urls(_decode_header(msg.get("subject", ""))))
            body = D.clean_text(D.strip_urls(_body_text(msg)))
            if len(subj) + len(body) == 0:
                continue
            rows.append({"source": "Apache_lists", "subject": subj, "body": body, "label": 0})
    ham = pd.DataFrame(rows)
    ham["text"] = D.make_text(ham)
    ham = ham[ham["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    out = HAM_DIR / "modern_ham.csv"
    ham.to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out} with {len(ham)} legitimate emails")


if __name__ == "__main__":
    main()
