# TakeMeter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fine-tuned DistilBERT classifier that labels Formula 1 Reddit comments as `analysis` / `hot_take` / `reaction` / `joke`, compare it against a Groq zero-shot baseline, and ship the dataset, code, and evaluation report.

**Architecture:** Local Python handles data collection (PRAW), hand-labeling, and stratified splitting → producing one committed CSV. Fine-tuning, the Groq baseline, and metrics run in the user's Colab notebook (free T4 GPU), which consumes the CSV and emits `evaluation_results.json` + `confusion_matrix.png`. Docs (`planning.md`, `README.md`) live at the repo root.

**Tech Stack:** Python 3, `praw`, `pandas`, `scikit-learn`, `python-dotenv` (local); HuggingFace `transformers` + `datasets`, Groq API (`llama-3.3-70b-versatile`) (Colab).

**Legend:** 🧪 = TDD code task · 🛠️ = manual/local procedure · ☁️ = Colab procedure.

---

## File structure

```
ai201-project3-takemeter/
├── planning.md                      # spec (already written)
├── README.md                        # graded writeup (filled across tasks)
├── requirements.txt
├── .gitignore
├── .env.example                     # creds template (real .env is gitignored)
├── data/
│   ├── pool.csv                     # raw scraped candidate comments
│   └── takemeter_dataset.csv        # 200 labeled rows + split column (THE deliverable)
├── scripts/
│   ├── collect.py                   # PRAW collector
│   └── make_splits.py               # stratified 70/15/15 split
├── notebook/
│   ├── label_map.py                 # label↔id mapping for the notebook
│   └── groq_prompt.md               # exact zero-shot baseline prompt
├── results/
│   ├── evaluation_results.json      # downloaded from Colab
│   └── confusion_matrix.png         # downloaded from Colab
└── tests/
    ├── test_collect.py
    └── test_make_splits.py
```

---

## Task 0: Repo & environment setup 🛠️

**Files:**
- Create: `.gitignore`, `requirements.txt`, `.env.example`, `README.md` (skeleton), folders `data/ scripts/ notebook/ results/ tests/`

- [ ] **Step 1: Initialize git and folders**

Run (PowerShell, from project root):
```powershell
git init
New-Item -ItemType Directory -Force data, scripts, notebook, results, tests | Out-Null
```

- [ ] **Step 2: Create `.gitignore`**

```
.env
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

- [ ] **Step 3: Create `requirements.txt`**

```
praw==7.7.1
pandas
scikit-learn
python-dotenv
pytest
```

- [ ] **Step 4: Create `.env.example`**

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=takemeter:v1 (by u/your_username)
```

- [ ] **Step 5: Create `README.md` skeleton** (headings only — each gets filled by a later task)

```markdown
# TakeMeter — F1 Discourse Classifier

## Community
## Labels & definitions
## Data
### Source
### Labeling process
### Label distribution
### Three hard-to-label examples
## Model & training
## Baseline (Groq zero-shot)
## Evaluation report
### Metrics
### Confusion matrix
### Error analysis (3+ examples)
### What the model learned vs. what I intended
## How to run
```

- [ ] **Step 6: Create local venv and install**

Run:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Expected: installs complete without error; `python -c "import praw, pandas, sklearn"` prints nothing (success).

- [ ] **Step 7: Commit**

```powershell
git add .gitignore requirements.txt .env.example README.md planning.md docs/
git commit -m "chore: scaffold TakeMeter repo and planning docs"
```

---

## Task 1: Reddit API credentials 🛠️ (prerequisite for Task 3)

**Files:** Create: `.env` (NOT committed)

- [ ] **Step 1: Create a Reddit app**

Go to https://www.reddit.com/prefs/apps → "create another app" → type **script** → set redirect URI to `http://localhost:8080`. Copy the client id (under the app name) and the secret.

- [ ] **Step 2: Write `.env`** (copy `.env.example`, fill real values)

```
REDDIT_CLIENT_ID=abcd1234
REDDIT_CLIENT_SECRET=wxyz5678
REDDIT_USER_AGENT=takemeter:v1 (by u/your_username)
```

- [ ] **Step 3: Verify connectivity**

Run:
```powershell
python -c "import os,praw; from dotenv import load_dotenv; load_dotenv(); r=praw.Reddit(client_id=os.environ['REDDIT_CLIENT_ID'],client_secret=os.environ['REDDIT_CLIENT_SECRET'],user_agent=os.environ['REDDIT_USER_AGENT']); print(r.subreddit('formula1').display_name)"
```
Expected: prints `formula1`. (If it errors on auth, re-check id/secret.)

---

## Task 2: Collection script 🧪

**Files:**
- Create: `scripts/collect.py`
- Test: `tests/test_collect.py`

- [ ] **Step 1: Write the failing test**

`tests/test_collect.py`:
```python
from scripts.collect import clean_text, is_filterable

def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world\n\n") == "hello world"

def test_is_filterable_drops_deleted_short_and_urls():
    assert is_filterable("[deleted]") is True
    assert is_filterable("[removed]") is True
    assert is_filterable("nice") is True            # too short (<15)
    assert is_filterable("https://i.imgur.com/x") is True
    assert is_filterable("") is True

def test_is_filterable_keeps_real_comment():
    assert is_filterable("Leclerc lost that on strategy, not pace.") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_collect.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.collect'`.

- [ ] **Step 3: Write `scripts/collect.py`**

```python
"""Collect candidate F1 comments into data/pool.csv for hand-labeling."""
import csv
import os
import re

WS = re.compile(r"\s+")


def clean_text(text):
    return WS.sub(" ", text or "").strip()


def is_filterable(text):
    """True = drop before a human ever sees it (junk, not a label decision)."""
    if not text:
        return True
    if text.lower() in ("[deleted]", "[removed]"):
        return True
    if len(text) < 15:
        return True
    if text.startswith("http") and " " not in text:
        return True
    return False


# Pull each label from where it's densest (see planning.md §4).
# (subreddit, time_filter, num_submissions, comments_per_submission)
SUBREDDIT_SOURCES = [
    ("F1Technical", "year", 25, 15),   # analysis-dense
    ("formula1", "month", 30, 15),     # hot_take / joke / reaction
]
# Paste a few live race-thread / post-race discussion URLs for reaction-dense comments:
SUBMISSION_URLS = [
    # "https://www.reddit.com/r/formula1/comments/XXXXXX/...",
]


def collect(out_path="data/pool.csv"):
    import praw
    from dotenv import load_dotenv
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )
    seen, rows = set(), []

    def add(c, sub):
        body = clean_text(getattr(c, "body", ""))
        if c.id in seen or is_filterable(body):
            return
        seen.add(c.id)
        rows.append({"id": c.id, "text": body, "source_sub": sub,
                     "score": c.score, "permalink": "https://reddit.com" + c.permalink})

    for sub, tf, nsub, ncom in SUBREDDIT_SOURCES:
        for sm in reddit.subreddit(sub).top(time_filter=tf, limit=nsub):
            sm.comments.replace_more(limit=0)
            for c in sm.comments[:ncom]:
                add(c, sub)
    for url in SUBMISSION_URLS:
        sm = reddit.submission(url=url)
        sm.comments.replace_more(limit=0)
        for c in sm.comments[:50]:
            add(c, sm.subreddit.display_name)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "source_sub", "score", "permalink"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} comments to {out_path}")


if __name__ == "__main__":
    collect()
```

Also create empty `scripts/__init__.py` and `tests/__init__.py` so imports resolve:
```powershell
New-Item -ItemType File scripts/__init__.py, tests/__init__.py | Out-Null
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_collect.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/collect.py scripts/__init__.py tests/test_collect.py tests/__init__.py
git commit -m "feat: PRAW collector with text cleaning and junk filtering"
```

---

## Task 3: Build the candidate pool & sanity-read 🛠️

**Files:** Create: `data/pool.csv`

- [ ] **Step 1: (Optional) add 2–4 race-thread URLs** to `SUBMISSION_URLS` in `scripts/collect.py` for reaction-dense comments (find them via r/formula1 → search "Race Thread").

- [ ] **Step 2: Run the collector**

Run: `python scripts/collect.py`
Expected: prints `Wrote N comments to data/pool.csv` with N ≥ 400 (gives margin to pick a balanced 200).

- [ ] **Step 3: Sanity-read 30–40 rows** of `data/pool.csv`. Confirm the four labels actually apply cleanly. **If a label barely appears or boundaries feel wrong, revise `planning.md` §2–3 BEFORE labeling 200.** (This is the brief's "read before you commit" gate.)

- [ ] **Step 4: Commit the pool**

```powershell
git add data/pool.csv scripts/collect.py
git commit -m "data: collect candidate comment pool"
```

---

## Task 4: Annotate 200 comments 🛠️ (the core deliverable)

**Files:** Create: `data/takemeter_dataset.csv`

- [ ] **Step 1: Create the labeling file** with this exact header and copy chosen rows from `pool.csv`:

```
id,text,label,source_sub,split,notes
```
Leave `split` blank (Task 5 fills it). Put text in quotes if it contains commas.

- [ ] **Step 2: Label ~200 rows**, applying `planning.md` decision rules. Targets: ~50 per label, **hard floor ≥40 each**. `label` ∈ {`analysis`,`hot_take`,`reaction`,`joke`}. Drop (don't label) anything that fits none — keep `other` at 0%.

- [ ] **Step 3: Record hard cases.** For every genuinely borderline comment, write one line in `notes` explaining which rule decided it. You need **≥3** of these for the README.

- [ ] **Step 4: Verify the dataset** (counts + valid labels)

Run:
```powershell
python -c "import pandas as pd; d=pd.read_csv('data/takemeter_dataset.csv'); print(len(d),'rows'); print(d['label'].value_counts()); assert d['label'].isin(['analysis','hot_take','reaction','joke']).all(); assert d['label'].value_counts().min()>=40"
```
Expected: ~200 rows, each label ≥40, no `AssertionError`.

- [ ] **Step 5: Commit**

```powershell
git add data/takemeter_dataset.csv
git commit -m "data: 200 labeled F1 comments"
```

---

## Task 5: Stratified split 🧪

**Files:**
- Create: `scripts/make_splits.py`
- Test: `tests/test_make_splits.py`

- [ ] **Step 1: Write the failing test**

`tests/test_make_splits.py`:
```python
import pandas as pd
from scripts.make_splits import add_stratified_split

def _toy():
    labels = ["analysis", "hot_take", "reaction", "joke"] * 10  # 40 rows, balanced
    return pd.DataFrame({"id": range(40), "text": ["x"] * 40, "label": labels})

def test_every_row_gets_a_split():
    out = add_stratified_split(_toy())
    assert out["split"].isin(["train", "val", "test"]).all()
    assert out["split"].notna().all()

def test_splits_are_disjoint_no_leakage():
    out = add_stratified_split(_toy())
    by = {s: set(out[out.split == s]["id"]) for s in ["train", "val", "test"]}
    assert by["train"].isdisjoint(by["test"])
    assert by["train"].isdisjoint(by["val"])
    assert by["val"].isdisjoint(by["test"])

def test_each_split_has_all_four_labels():
    out = add_stratified_split(_toy())
    for s in ["train", "val", "test"]:
        assert set(out[out.split == s]["label"]) == {"analysis", "hot_take", "reaction", "joke"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_make_splits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.make_splits'`.

- [ ] **Step 3: Write `scripts/make_splits.py`**

```python
"""Add a stratified 70/15/15 train/val/test split column."""
import pandas as pd
from sklearn.model_selection import train_test_split


def add_stratified_split(df, seed=42):
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=seed)
    df = df.copy()
    df["split"] = None
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/takemeter_dataset.csv")
    df = add_stratified_split(df)
    df.to_csv("data/takemeter_dataset.csv", index=False)
    print(df.groupby(["split", "label"]).size())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_make_splits.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Apply to the real dataset**

Run: `python scripts/make_splits.py`
Expected: prints a split×label table; test split totals ~30, each label present in every split.

- [ ] **Step 6: Commit**

```powershell
git add scripts/make_splits.py tests/test_make_splits.py data/takemeter_dataset.csv
git commit -m "feat: stratified 70/15/15 split + applied to dataset"
```

---

## Task 6: Fine-tune DistilBERT in Colab ☁️

**Files:** Create: `notebook/label_map.py`

- [ ] **Step 1: Write `notebook/label_map.py`** (paste into the notebook's label-map cell)

```python
label_map = {"analysis": 0, "hot_take": 1, "reaction": 2, "joke": 3}
id2label = {v: k for k, v in label_map.items()}
label2id = label_map
```

- [ ] **Step 2: Set up the runtime.** In your copied starter notebook: Runtime → Change runtime type → **T4 GPU**. Add `GROQ_API_KEY` via the 🔑 Secrets panel (needed in Task 7).

- [ ] **Step 3: Upload `data/takemeter_dataset.csv`** and load it, using the existing `split` column to build train/val/test `datasets.Dataset`s (do **not** re-split — that would risk leakage).

- [ ] **Step 4: Set training args — the documented hyperparameter decision.** Use `num_train_epochs=4`, `learning_rate=2e-5`, `per_device_train_batch_size=16`, `eval_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`. Rationale (for README): ~140 train rows → overfitting is the main risk, so few epochs + keep the best-by-validation checkpoint instead of training long.

- [ ] **Step 5: Train.** Run the training cell.
Expected: training completes in ~5–15 min; validation accuracy prints per epoch. Record the best val accuracy for the README.

- [ ] **Step 6: Save the fine-tuned model's test-set predictions** (a list of predicted labels for the 30 test rows) — Task 8 needs them alongside the baseline's.

---

## Task 7: Groq zero-shot baseline ☁️

**Files:** Create: `notebook/groq_prompt.md`

- [ ] **Step 1: Write `notebook/groq_prompt.md`** (the exact prompt; paste into the baseline cell as an f-string on `{text}`)

```text
You are classifying the REGISTER of a comment from Formula 1 fan discussion into exactly one of four labels. Reply with ONLY the label in lowercase, nothing else.

Labels:
- analysis: a structured argument backed by specific, verifiable evidence (lap times, tire strategy, technical/regulation reasoning, statistics). The reasoning is the point.
- hot_take: a bold, confident opinion or judgment asserted without genuine evidence (rankings, overrated/underrated, predictions stated as fact). Any evidence is vague or decorative.
- reaction: an immediate emotional response to a specific event, with little or no argument.
- joke: a comment whose primary intent is humor — meme, pun, bit, sarcasm, copypasta.

Rules:
- If an emotional or ranty tone wraps a real, evidenced argument, it is still analysis (substance over tone).
- If the comedic construction is clearly the point, it is joke; if it is sincere emotion that merely happens to be hyperbolic, use the sincere label.

Comment:
\"\"\"{text}\"\"\"

Label:
```

- [ ] **Step 2: Classify the 30 test comments** with `llama-3.3-70b-versatile` (`temperature=0`) using the prompt above.

- [ ] **Step 3: Normalize each response** — lowercase, strip whitespace/punctuation, keep the first token that exactly matches one of the four labels; if none matches, record `"unknown"` (counts as wrong). Verify all 30 produced a prediction.

---

## Task 8: Evaluation & outputs ☁️

**Files:** Create: `results/evaluation_results.json`, `results/confusion_matrix.png`

- [ ] **Step 1: Compute metrics for BOTH models** on the same 30 test rows: overall accuracy, and per-class precision/recall/F1 via `sklearn.metrics.classification_report(..., output_dict=True)`.

- [ ] **Step 2: Build a 4×4 confusion matrix** for the fine-tuned model (`ConfusionMatrixDisplay`), label the axes with the four class names, and save:
```python
plt.savefig("confusion_matrix.png", bbox_inches="tight")
```

- [ ] **Step 3: Write `evaluation_results.json`** containing both models' accuracy and per-class reports, e.g.:
```python
import json
json.dump({"finetuned": ft_report, "baseline_groq": groq_report}, open("evaluation_results.json", "w"), indent=2)
```

- [ ] **Step 4: Download** `confusion_matrix.png` and `evaluation_results.json` from Colab into the local `results/` folder.

- [ ] **Step 5: Commit**

```powershell
git add results/evaluation_results.json results/confusion_matrix.png notebook/
git commit -m "results: fine-tuned vs Groq baseline evaluation"
```

---

## Task 9: Write the README 🛠️

**Files:** Modify: `README.md`

- [ ] **Step 1: Fill every section** using results from Tasks 4–8. Required content (from the brief), with nothing left blank:
  - **Community** — 2–3 sentences (reuse `planning.md` §1).
  - **Labels & definitions** — the four labels + one-sentence definitions + the decision rules.
  - **Data → Source** — which subreddits/threads, that it's comments, collected via PRAW.
  - **Data → Labeling process** — how you applied the rules; single annotator.
  - **Data → Label distribution** — the actual counts per label (from Task 4 Step 4).
  - **Data → Three hard examples** — 3 real comments from your `notes` column + what you decided and why.
  - **Model & training** — started from `distilbert-base-uncased`; the training approach; the **epochs** hyperparameter decision + rationale (Task 6 Step 4).
  - **Baseline** — Groq `llama-3.3-70b-versatile`, zero-shot, same test set.
  - **Evaluation report** — accuracy for BOTH models; per-class P/R/F1; the confusion matrix image; **3 specific wrong predictions** with your why-analysis; reflection on **learned vs. intended** (lead with the tone-vs-substance failure mode from `planning.md`).
  - **How to run** — venv + `pip install -r requirements.txt`, `.env` setup, `python scripts/collect.py`, `python scripts/make_splits.py`, and the Colab steps.

- [ ] **Step 2: Verify completeness** — re-read the brief's "Evaluation report" bullet and confirm each item has a corresponding filled subsection. No "TBD".

- [ ] **Step 3: Commit**

```powershell
git add README.md
git commit -m "docs: complete README with evaluation report"
```

---

## Task 10: Stretch — error-pattern analysis (optional)

> Update `planning.md` §9 (check the box) before starting.

- [ ] **Step 1:** Group the fine-tuned model's wrong test predictions by (true label → predicted label) and by surface features (has CAPS? has emoji? length < 50 chars?).
- [ ] **Step 2:** State one *systematic* pattern with evidence (e.g., "k of m evidenced rants were misread as hot_take/reaction; all had CAPS or profanity" — the predicted tone-over-substance failure).
- [ ] **Step 3:** Add a "Systematic error pattern" subsection to the README; commit.

---

## Self-review (against planning.md)

- **§1 Community** → Task 9 Step 1. ✔
- **§2 Labels / §3 Rules** → encoded in `planning.md`, surfaced in README (Task 9) and Groq prompt (Task 7). ✔
- **§4 Data (unit, ≥40/label, sourcing, filtering, CSV schema, 70/15/15, no leakage)** → Tasks 2–5; leakage guarded by `make_splits` test + reuse of `split` column in Colab (Task 6 Step 3). ✔
- **§5 Fine-tune (distilbert, epochs decision)** → Task 6. ✔
- **§6 Baseline (Groq zero-shot, same test set)** → Task 7. ✔
- **§7 Eval (accuracy, per-class P/R/F1, 4×4 CM, 3 errors, reflection, output files)** → Task 8 + Task 9. ✔
- **§8 Deliverables / repo structure** → Task 0 + matches file-structure section. ✔
- **§9 Stretch** → Task 10 (error-pattern). ✔
- **Placeholder scan:** none (stretch checkboxes intentional). ✔
- **Type/name consistency:** `clean_text`/`is_filterable`/`collect` (Task 2) and `add_stratified_split` (Task 5) used consistently in tests and `__main__`; `label_map` keys match the four labels everywhere. ✔
