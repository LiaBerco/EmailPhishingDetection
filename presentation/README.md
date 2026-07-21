# Presentation

`phishfusion_presentation.html` — the full **25-slide** talk for this project. A single,
self-contained HTML file (no internet needed, no dependencies): open it in any modern
browser and present.

## How to present
- **Open** the file in Chrome/Edge/Firefox.
- Press **F** for fullscreen (renders at 16:9, the intended size).
- **Navigate:** → / Space / click to advance · ← to go back · click a dot or use Home/End to jump.
- **Theme:** a "theme" button toggles light/dark; the default follows your OS.
- **Slide 9 is a live demo** — type an email and toggle the homoglyph attack and the
  normalise defence to watch the verdict flip on stage. (Clicking inside the demo does not
  advance the slide.)

## What's on it (order)
1 title · 2 the project + fusion with Gilad · 3 method (build→attack→defend→repeat) ·
4 pipeline · 5 why Logistic Regression · 6 explainability · 7 clean results ("is it real?") ·
8–10 attack 1 (homoglyph → live demo → normalise) · 11–12 attack 2 (adaptive → UTS-39
certificate) · 13–14 attack 3 (dilution → monotone certificate) · 15–16 reality checks
(cross-corpus, temporal drift) · 17 evolve · 18 "what changed after each defence" (the
arms-race chart) · 19 model summary · 20 LLM-as-judge · 21 human-in-the-loop · 22 dynamic
database · 23 deployment · 24 future work · 25 closing.

Every number on the slides is a real result from the executed notebook.

## Notes
- The design uses the same palette as the report figures (teal = defence, coral = attack,
  gold = a certified proof), so the deck and the report read as one system.
- It transitions between slides (content glides in/out); animation respects
  "reduced motion" if the viewer has that set.
- The talk narration lives in [`../docs/IMPROVEMENTS_STORY.md`](../docs/IMPROVEMENTS_STORY.md)
  (Stages 0–12), which is effectively the speaker script.
- A private copy is also hosted as a Claude artifact; the local file here is the source of truth.
