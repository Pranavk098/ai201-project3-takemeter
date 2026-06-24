# TakeMeter — Classifying the Register of F1 Discourse

TakeMeter is a fine-tuned text classifier that labels a Formula 1 Reddit comment by its **register**: `analysis`, `hot_take`, `reaction`, or `joke`. It is built on `distilbert-base-uncased`, fine-tuned on 200 hand-curated comments, and benchmarked against a zero-shot Groq Llama-3.3-70B baseline.

🎥 **Demo video (≈3 min): https://www.loom.com/share/3e76942efa334f9aa0f122d9e751a76a**

> This README is the final report and stands on its own. [planning.md](planning.md) holds the deeper design notes (full label rules, edge-case reasoning, AI tool plan).

---

## Community

**Formula 1 fans on Reddit** — mainly [r/formula1](https://reddit.com/r/formula1) (race threads, news, opinions, memes) with [r/F1Technical](https://reddit.com/r/F1Technical) for the technical end. F1 is a strong fit for a discourse-quality task because the *same event* spawns four genuinely different registers in the same thread: a strategy breakdown, a "fire the pit wall" hot take, a despairing reaction, and a meme. Regulars distinguish these instantly, so the labels are grounded in real community norms.

**Framing caveat:** because one label (`joke`) is not a "take," TakeMeter classifies *register/type*, not quality on a single axis. Quality is the implied ordering `analysis` > `hot_take` > `reaction` > `joke`.

---

## Labels & definitions

| Label | One-sentence definition |
|---|---|
| `analysis` | A structured argument backed by **specific, verifiable evidence** (lap times, tyre strategy, technical/regulation reasoning, statistics) — the reasoning is the point. |
| `hot_take` | A **bold, confident opinion or judgment asserted without genuine evidence** (rankings, overrated/underrated, predictions stated as fact); any evidence is vague or decorative. |
| `reaction` | An **immediate emotional response to a specific event**, with little or no argument and no debatable position. |
| `joke` | A comment whose **primary intent is humor** — meme, pun, bit, sarcasm, copypasta. |

**Two example posts per label:**
- `analysis` — *"Leclerc lost that on strategy not pace — he was ~0.3s/lap quicker on mediums but Ferrari left him out 4 laps too long, so he came back into traffic and the undercut was gone."* / *"Red Bull's high-speed advantage is the anti-dive front suspension — it holds ride height under braking, protecting underfloor load."*
- `hot_take` — *"Verstappen is the most overrated champion in history. Put him in a midfield car and he's nothing."* / *"Hamilton is washed, he should retire. Embarrassing to watch."*
- `reaction` — *"OH MY GOD WHAT A MOVE BY LANDO INTO TURN 1, I'M SHAKING"* / *"not again Ferrari… I genuinely can't watch this team anymore 😭"*
- `joke` — *"Ferrari's strategy dept was last seen taking orders from a magic 8-ball."* / *"BREAKING: Stroll qualifies P18, blames the car, the wind, and the concept of time itself."*

**Decision rules** (the boundaries that actually matter):
1. **`hot_take` vs `analysis`** — strip the opinion/rant framing; if the remaining evidence is *load-bearing* → `analysis`, if *decorative* → `hot_take` (substance over tone: a furious rant with a real argument is still `analysis`).
2. **`reaction` vs `hot_take`** — a debatable verdict ("that should've been a penalty") → `hot_take`; pure venting, even hyperbolic ("worst stop ever") → `reaction`.
3. **`joke` vs `reaction`/`hot_take`** — if the comedic construction is the point → `joke`; if it's sincere emotion that's merely hyperbolic → the sincere label; **50/50 → default to the sincere label.**

---

## Data

### Source
Public Reddit comments scraped via **Apify** (`automation-lab/reddit-scraper`) across three runs: r/F1Technical + r/formula1 top-of-year posts; r/formula1 "Race Discussion" threads (hot-take dense); and r/formula1 "Race Thread" threads (reaction dense). Public content only. *(PRAW was the original plan; Reddit's Responsible Builder Policy blocked API-app creation, so we pivoted to Apify.)* Raw scrapes were cleaned/deduped (bots, deleted, <15-char, links removed) into a 630-comment candidate pool, then curated to a balanced 200.

**Final source split:** 189 from r/formula1, 18 from r/F1Technical.

### Labeling process
Labels were assigned by applying the definitions and decision rules above. **Annotation disclosure:** the 200 labels were **AI-pre-labeled (Claude, applying the rules) and reviewed by the annotator**; 7 genuinely ambiguous comments were left blank for a human decision and excluded from training/eval. This is disclosed because the baseline is also an LLM (see [AI usage](#ai-usage)).

### Label distribution
Curated to be exactly balanced so per-class metrics are interpretable and no class can dominate (the brief's hard ceiling is no label > 70%):

| Label | Count |
|---|---|
| analysis | 50 |
| hot_take | 50 |
| reaction | 50 |
| joke | 50 |
| *(blank — ambiguous, excluded)* | 7 |
| **Total labeled** | **200** |

The notebook/pipeline splits this 70/15/15 → **train 140, val 30, test 30**, stratified (7–8 per class in test).

### Three hard-to-label examples
1. **`hot_take` vs `analysis`** — *"Apart from that time he demanded a journalist leave a press conference… but it's back to acting like it's Saint Verstappen again."* Cites a specific event, which *looks* like evidence. **Decided `hot_take`** (rule #1): the event is decorative — scoring a rhetorical point, not reasoning toward a checkable claim.
2. **`reaction` vs `hot_take`/`joke`** — *"Max being the first F1 driver to win in a Ferrari this year is diabolical."* **Decided `reaction`** (rule #2/#3): an amused in-the-moment exclamation with no debatable position and no constructed punchline.
3. **`analysis` vs `hot_take` — left BLANK** — *"Except that's not corruption. Corruption is knowingly using your position of power for personal gain… nothing indicates Masi's doings were acts of corruption."* A calm, reasoned argument but with **no specific verifiable F1 evidence**, so it fails the load-bearing test for `analysis` yet is too measured for a clean `hot_take`. Neither rule fires cleanly → blank.

---

## Model & training

- **Base model:** `distilbert-base-uncased` with a fresh 4-class head, fine-tuned locally on CPU (PyTorch + 🤗 Transformers).
- **Key hyperparameter decision — epochs (and LR):** The brief's default (3–4 epochs, lr 2e-5) **under-fit** this tiny dataset: after 4 epochs the train loss had barely moved (1.38 → 1.28; random for 4 classes is ln 4 ≈ 1.386) and `reaction` collapsed to **F1 = 0.00**. Because the loss curve showed clear *under*-fitting (not over-fitting), the fix was *more* training, not less. I raised it to **10 epochs, lr 4e-5, with 10% warmup and 0.01 weight decay**. Train loss then fell to 0.09 and `reaction` recovered to F1 = 0.71. Final config: **10 epochs · lr 4e-5 · batch 16 · max_len 256 · seed 42.**

---

## Baseline (Groq zero-shot)

`llama-3.3-70b-versatile`, **true zero-shot** (temperature 0). **How results were collected:** [scripts/run_baseline.py](scripts/run_baseline.py) sends each of the 30 test comments to the Groq API with the prompt below, then parses each response by taking the first exact label match (lowercased); an unmatched response is counted as wrong. **0 of 30 were unparseable.** The prompt gives the label definitions and rules but **no example posts**, keeping it genuinely zero-shot:

```text
You are classifying the REGISTER of a comment from Formula 1 fan discussion into exactly one of four labels. Reply with ONLY the label in lowercase, nothing else.

Labels:
- analysis: a structured argument backed by specific, verifiable evidence (lap times, tire strategy, technical/regulation reasoning, statistics). The reasoning is the point.
- hot_take: a bold, confident opinion or judgment asserted without genuine evidence (rankings, overrated/underrated, predictions stated as fact). Any evidence is vague or decorative.
- reaction: an immediate emotional response to a specific event, with little or no argument.
- joke: a comment whose primary intent is humor - meme, pun, bit, sarcasm, copypasta.

Rules:
- If an emotional or ranty tone wraps a real, evidenced argument, it is still analysis (substance over tone).
- If the comedic construction is clearly the point, it is joke; if it is sincere emotion that merely happens to be hyperbolic, use the sincere label.

Comment:
"""{comment}"""

Label:
```

---

## Evaluation report

### Overall accuracy & per-class metrics (test set, n = 30)

| | Accuracy | Macro-F1 |
|---|---|---|
| **Groq zero-shot (baseline)** | **0.80** | **0.78** |
| Fine-tuned DistilBERT | 0.73 | 0.74 |

**Per-class F1 (precision / recall / F1):**

| Label | Baseline (Groq) | Fine-tuned |
|---|---|---|
| analysis | 0.73 / 1.00 / **0.84** | 0.75 / 0.75 / 0.75 |
| hot_take | 0.60 / 0.43 / 0.50 | 0.86 / 0.86 / **0.86** |
| reaction | 0.86 / 0.86 / **0.86** | 0.71 / 0.71 / 0.71 |
| joke | 1.00 / 0.88 / **0.93** | 0.62 / 0.62 / 0.62 |

**Headline result (reported honestly): the zero-shot 70B LLM beats the fine-tuned model (0.80 vs 0.73).** Against the success criteria in planning.md §6 — macro-F1 ≥ 0.70 ✓ (0.74), `analysis` recall ≥ 0.60 ✓ (0.75) — but **"beat the baseline by ≥ 0.10" failed**: fine-tuning scored *below* the baseline. For a subjective 4-way task with only 140 training rows, fine-tuning did not earn its keep. The two models also have **opposite weak spots**: the baseline is weakest on `hot_take` (F1 0.50), while fine-tuning *fixed* `hot_take` (0.86) but became weakest on `joke` (0.62).

### Confusion matrix — fine-tuned model (rows = true, cols = predicted)

| true ↓ / pred → | analysis | hot_take | reaction | joke |
|---|---|---|---|---|
| **analysis** | **6** | 1 | 0 | 1 |
| **hot_take** | 1 | **6** | 0 | 0 |
| **reaction** | 0 | 0 | **5** | 2 |
| **joke** | 1 | 0 | 2 | **5** |

(A supplementary image is committed at [results/confusion_matrix.png](results/confusion_matrix.png).)

### Error analysis (3 of the 8 misses, with the pattern)

I asked an LLM to surface themes across all 8 wrong predictions, then verified each against the confusion matrix and the raw text. The dominant theme it flagged — and which the matrix confirms — is **`joke ↔ reaction` confusion: 4 of the 8 errors** (reaction→joke ×2, joke→reaction ×2). A second theme is **technical vocabulary pulling predictions toward `analysis`**. (One hypothesis I *discarded*: "short posts cause the errors." Length correlates with the joke/reaction confusion, but error #1 below is long and still wrong — vocabulary and intent matter more than length alone.)

1. **`hot_take` → `analysis` (confidence 0.97, confidently wrong)** — *"Once you get stuck into the technical side of F1, you realise no journalist can explain how upgrades work… it boils down to adding downforce or 'flow conditioning'."*
   **Which boundary:** analysis↔hot_take. **Why hard:** it's saturated with F1 jargon ("upgrades," "downforce," "flow conditioning") but makes *no evidenced argument* — it's an unsupported opinion about journalists. **Labeling vs data:** I labeled it consistently as `hot_take` (rule #1: the jargon is decorative). The model still missed it, so this is a **data/boundary problem**: it learned "jargon ⇒ analysis" instead of "evidence ⇒ analysis." **Fix:** add more `hot_take` examples that *use* technical vocabulary, to teach that words ≠ evidence.
2. **`joke` → `reaction` (confidence 0.51)** — *"Bottle flip❌ F1 car flip✅"*
   **Which boundary:** joke↔reaction. **Why hard:** the humor lives entirely in the ❌/✅ meme format, not the words; stripped of format it looks like a terse emotional fragment. **Fix:** more short, format-driven meme jokes so the model sees that structure.
3. **`reaction` → `joke` (confidence 0.60)** — *"That is absolutely bananas that they didn't go."*
   **Which boundary:** joke↔reaction. **Why hard:** genuine incredulity expressed with playful diction ("bananas") reads like a quip; the post is short and carries no argument either way. **Fix:** more examples separating sincere hyperbole from comedic intent — exactly the rule-#3 tie-breaker, which is the hardest call in the taxonomy.

### Sample classifications (fine-tuned model + confidence)

| Comment (truncated) | True | Predicted | Confidence |
|---|---|---|---|
| "Franz Tost and Christian Horner don't get enough credit… Pierre Gasly's P4 in Bahrain… Toro Rosso built around the engine" | analysis | **analysis** | 0.97 ✓ |
| "He is always talking about Max on this dumb podcast. Unfortunately he is a disgraced driver and a loser" | hot_take | **hot_take** | 0.93 ✓ |
| "Gasly's car is going to remember it has a Mercedes engine now." | joke | **joke** | 0.74 ✓ |
| "Everybody happy! What a great story" | reaction | **reaction** | 0.73 ✓ |
| "Once you get stuck into the technical side of F1… no journalist can explain upgrades…" | hot_take | **analysis** | 0.97 ✗ |

**Why the first prediction is reasonable:** the comment makes a structured causal argument with *specific, checkable* facts (Gasly's P4 in Bahrain in Toro Rosso's first Honda year; the team choosing to build around the engine) — that is precisely the evidenced reasoning the `analysis` definition targets, and the model is correctly, highly confident (0.97).

### Confidence calibration (stretch)
Confidence **is** meaningful — accuracy rises monotonically with it on the test set:

| Confidence bin | n | Accuracy |
|---|---|---|
| 0.90–1.00 | 12 | 0.83 |
| 0.70–0.89 | 5 | 0.80 |
| 0.50–0.69 | 9 | 0.67 |
| < 0.50 | 4 | 0.50 |

Aggregated, **high-confidence (≥0.80) predictions are 0.79 accurate vs 0.69 for low-confidence (<0.80)**, so the score is usable as a triage signal. The one caveat: two errors are *confidently* wrong (0.92–0.97), both the jargon→`analysis` trap — confidence is well-calibrated in aggregate but can still be fooled by heavy F1 vocabulary.

### What the model learned vs. what I intended

I defined the labels by **communicative intent and structure**: `analysis` = evidenced reasoning, `hot_take` = unevidenced verdict, `reaction` = sincere emotion, `joke` = comedic intent. The model instead learned **lexical proxies**: *technical jargon → analysis*, and *brevity + playful diction → joke/reaction* (two registers it routinely conflates). It captured the `hot_take`/`analysis` split better than expected (F1 0.86) — but only via vocabulary, which is why a jargon-laden opinion fools it. It essentially **failed to learn the `joke`/`reaction` intent distinction**, the part of my taxonomy that most requires understanding *comedic construction* rather than surface words — and the part 140 examples are least able to teach. The gap, in one line: **my definitions are about why a comment was written; the model's decision boundary is about which words it contains.**

---

## Demo

**▶️ [Watch the demo (≈3 min)](https://www.loom.com/share/3e76942efa334f9aa0f122d9e751a76a)** — it shows live classification (via `app.py`), narrates one correct and one incorrect prediction, and walks through this evaluation report. The same content is reproducible from source:
- **3–5 posts with label + confidence:** the [Sample classifications](#sample-classifications-fine-tuned-model--confidence) table, or run `python scripts/predict.py "<comment>"`.
- **Correct, explained:** the analysis 0.97 example (specific checkable evidence → high-confidence correct).
- **Incorrect, explained:** the hot_take→analysis 0.97 miss (jargon fooled it) in [Error analysis](#error-analysis-3-of-the-8-misses-with-the-pattern).
- **Key metrics summary:** the accuracy / macro-F1 table at the top of the evaluation report.

## Deployed interface (stretch)

[`app.py`](app.py) is a **Gradio web app**: paste any F1 comment and it shows the predicted label with a confidence bar across all four classes. Run `python app.py` → opens a local URL. ([`scripts/predict.py`](scripts/predict.py) is the CLI equivalent.) Verified example — input *"Verstappen is the most overrated champion in history"* → **hot_take 0.62** (joke 0.20, reaction 0.14, analysis 0.05).

## Inter-annotator reliability (stretch)

**Two people independently labeled the same 32 comments** (8 per class) blind; `python scripts/iaa.py` measures their agreement with Cohen's kappa.

- **Result: 46.9% agreement, Cohen's κ = 0.28** ("fair"), n = 32, 17 disagreements.
- **Where they disagreed:** `reaction`↔`hot_take` is the biggest split (8 of 17); then the `analysis` boundary on *passionate but evidenced* posts (5 — e.g. the Honda rant and the brake-marbles breakdown: one annotator read venting, the other a real argument); then `joke` vs the rest (4).
- **Interpretation (honest):** even two humans agree only ~47%. That low κ is itself a finding — `analysis` is reasonably shareable, but the `hot_take` / `reaction` / `joke` distinctions are **genuinely subjective**: reasonable people draw them differently. This directly explains why the fine-tuned model's errors cluster on `joke`↔`reaction`, and why neither model clears ~0.80 — a large part of the ceiling is in the *task itself*, not the model.

---

## Spec reflection

- **How the spec helped:** the precise decision rules doubled as *predictions*. planning.md flagged `analysis`↔`hot_take` and the "tone vs. substance" trap as the hardest boundaries *before any training* — and the model's confident jargon→analysis errors confirmed it exactly. The spec's insistence on a stratified, ≥-balanced dataset also produced the clean 50/50/50/50 split that made per-class metrics trustworthy.
- **How the implementation diverged:** the spec planned **PRAW** collection and **hand-labeling**; in practice Reddit blocked API-app creation (→ pivoted to **Apify**) and labeling was done as **AI-pre-labeling + human review** at the user's request. Both divergences are disclosed here and in planning.md; the annotation change is the more consequential one and is the reason the AI-usage disclosure is prominent.

---

## AI usage

This project used Claude (Opus 4.8, via Claude Code) substantially. Specific instances:

1. **Label design & stress-test.** I directed Claude to generate ~8 boundary posts and classify them under my draft definitions. It produced cases that exposed a weak spot — `reaction` vs `hot_take` for *evaluative venting* — which led me to **sharpen rule #2** (debatable verdict → `hot_take`; hyperbolic venting → `reaction`).
2. **Annotation (disclosed).** Claude **pre-labeled all 200 comments** by applying my rules, leaving 7 ambiguous ones blank. These are **AI-suggested labels subject to human review**. Because the baseline is itself an LLM, this is a real limitation: unreviewed LLM labels would make the evaluation partly "does DistilBERT imitate the LLM that labeled it." Human review is what restores them as ground truth.
3. **Error-pattern analysis.** I gave Claude the 8 misclassifications and asked for common themes. It proposed the `joke`↔`reaction` cluster and the jargon→`analysis` trap; I **verified both against the confusion matrix** and discarded a weaker "length-only" hypothesis (see Error analysis).
4. **Code.** Claude wrote the scrape/clean/split/baseline/train/eval scripts (`scripts/`); the hyperparameter change (4→10 epochs) was a decision made after reading the under-fit loss curve.

---

## How to run

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate groq matplotlib

# secrets (gitignored): create .env with
#   GROQ_API_KEY=gsk_...

python scripts/run_baseline.py     # Groq zero-shot baseline -> results/baseline.json
python scripts/finetune_eval.py    # fine-tune + eval -> results/evaluation_results.json, confusion_matrix.png, ft_model/
python scripts/predict.py "Leclerc lost that on strategy not pace, he was 0.3s/lap quicker..."   # classify any comment
python app.py                      # Gradio web interface: paste a comment -> label + confidence
# inter-annotator (two-human study): python scripts/iaa.py   # compares annotator1 vs annotator2 in data/iaa_blind.csv
```

The data pipeline (`build_pool.py` → label → `build_dataset.py`) and the committed `data/takemeter_dataset.csv` reproduce the dataset; the original Colab notebook workflow is equivalent (label map in `notebook/label_map.py`, prompt in `notebook/groq_prompt.md`).

## Repository layout

```
planning.md                     design notes (labels, rules, AI plan, hard cases)
data/takemeter_dataset.csv      200 labeled comments (+7 blank) — the dataset
data/pool*.csv                  raw scraped candidate pools (provenance)
app.py                          Gradio web interface (deployed-interface stretch)
scripts/                        collect, build_pool, build_dataset, data_split, run_baseline, finetune_eval, predict, iaa
data/iaa_blind.csv              32-example blind sheet for inter-annotator reliability
results/evaluation_results.json baseline + fine-tuned metrics
results/confusion_matrix.png    fine-tuned confusion matrix (image)
notebook/                       label_map.py + groq_prompt.md for the Colab workflow
```
