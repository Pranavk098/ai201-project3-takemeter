# TakeMeter — Planning

*Project 3, CodePath Applications of AI. Last updated: 2026-06-22.*

TakeMeter is a fine-tuned text classifier that labels the **register of Formula 1 discourse** on Reddit. Given a comment, it predicts exactly one of four labels: `analysis`, `hot_take`, `reaction`, or `joke`. This document is the full project spec — written before any data was collected.

> **Note on examples:** posts shown below are *illustrative templates* written to mark each boundary. The real 200 annotated examples are scraped comments (Milestone 3).

---

## 1. Community

**Community:** Formula 1 fans on Reddit — primarily [r/formula1](https://reddit.com/r/formula1) (the main hub: race threads, news, opinions, memes), with [r/F1Technical](https://reddit.com/r/F1Technical) as a denser source of genuine analysis.

**Why I chose it, and why it's a good fit for classification:** F1 is one of the few communities whose discourse *naturally* spans four very different registers, and it produces all of them every single race weekend:

- A weekly **live event** cadence floods threads with in-the-moment emotional **reactions** (crashes, overtakes, penalties, results).
- A deep **technical/strategic layer** (tyre degradation, undercut/overcut, aero, regulations) produces evidence-backed **analysis**.
- Intense **driver/team tribalism** (Verstappen vs. Hamilton, Ferrari's strategy reputation) produces confident, unevidenced **hot takes**.
- A famously **meme-heavy culture** produces **jokes** in enormous volume.

That four-way spread is what makes the discourse varied enough to be interesting: the same event (say, a Ferrari pit blunder) generates a strategy breakdown, a "fire the pit wall" hot take, a despairing reaction, and a magic-8-ball meme — often in the same thread. Distinguishing these is something regulars do instinctively, which means the labels are grounded in real community norms.

**Framing caveat (stated honestly):** because one label (`joke`) is not a "take," TakeMeter classifies *register/type*, not quality on a single axis. Quality is the *implied ordering* `analysis` > `hot_take` > `reaction` > `joke`.

---

## 2. Labels

Four labels. Each definition is a complete sentence, followed by two example posts.

### `analysis`
**Definition:** A comment that makes a structured argument backed by specific, verifiable evidence — such as lap-time or sector comparisons, tyre-strategy reasoning, technical/aero/regulation explanation, or historical/statistical comparison — where the reasoning is the point and the claim could be checked.
- *"Leclerc lost that on strategy, not pace — he was ~0.3s/lap quicker on mediums but Ferrari left him out 4 laps too long, came back into traffic, and the undercut was gone."*
- *"Red Bull's high-speed advantage is the anti-dive front suspension — it holds ride height under braking, protecting underfloor load, so there's far less pitch than Ferrari on the onboards."*

### `hot_take`
**Definition:** A comment that asserts a bold, confident opinion or judgment — a ranking, an overrated/underrated claim, a "should be fired/should retire," or a prediction stated as fact — without genuine supporting evidence, where any evidence present is vague, cherry-picked, or decorative.
- *"Verstappen is the most overrated champion in history. Put him in a midfield car and he's nothing."*
- *"Hamilton is washed, he should retire. Embarrassing to watch."*

### `reaction`
**Definition:** A comment that is an immediate emotional response to a specific event — excitement, despair, anger, awe — with little or no argument and no debatable position, simply venting a feeling in the moment.
- *"OH MY GOD WHAT A MOVE BY LANDO INTO TURN 1, I'M SHAKING"*
- *"not again Ferrari… I genuinely can't watch this team anymore 😭"*

### `joke`
**Definition:** A comment whose primary intent is humor — a meme, pun, bit, copypasta, sarcastic shitpost, or running gag — where being funny, not informing or judging or sincerely emoting, is the point.
- *"Ferrari's strategy dept was last seen taking orders from a magic 8-ball."*
- *"BREAKING: Stroll qualifies P18, blames the car, the wind, and the concept of time itself."*

### Decision rules (boundary resolution)
Applied whenever a comment could plausibly fit two labels:

1. **`hot_take` vs `analysis`** — Strip the opinion *and emotional/rant framing*. If the remaining evidence is **load-bearing** (genuinely establishes the claim) → `analysis`. If it's **decorative** (cherry-picked, vague, propping up a pre-formed verdict) → `hot_take`. *Substance over tone: a furious rant with a real argument underneath is still `analysis`.*
2. **`reaction` vs `hot_take`** — A comment is `hot_take` only if it asserts a **debatable verdict/position** (a ranking, a "should have," a should-be-penalized claim). Pure emotional exclamation — *including hyperbolic superlatives like "worst ever"* with no argument — is `reaction`. (*"worst pit stop ever"* = venting → `reaction`; *"that should've been a penalty"* = verdict → `hot_take`.)
3. **`joke` vs `reaction`/`hot_take`** — If the *comedic construction is clearly the point* (bit, punchline, meme format, exaggeration-for-laughs) → `joke`. If it's sincere emotion/opinion that merely happens to be hyperbolic → the sincere register. **Tie-breaker:** when genuinely 50/50, default to the sincere category, *not* `joke`.
4. **Fits none** (pure questions, news/info dumps, off-topic) → **filtered out at collection**, never labeled, keeping the `other` bucket at 0%.

---

## 3. Hard edge cases

The genuinely ambiguous post *types* I expect during annotation, and how I'll handle each:

| Ambiguous type | Example | Handling rule |
|---|---|---|
| **Technically-sound rant** (analysis ↔ hot_take) | *"I'm SO sick of people saying Leclerc threw it. He pitted lap 18 on the undercut, came out P3, then the wall double-stacked Sainz for no reason, costing ~5s…"* | Rule #1: strip the anger; the specific causal argument survives → **`analysis`**. |
| **Evaluative venting** (reaction ↔ hot_take) | *"Worst pit stop I've ever seen, clown show"* vs *"that should've been a penalty, stewards are blind"* | Rule #2: hyperbolic venting with no debatable position → **`reaction`**; an arguable verdict → **`hot_take`**. |
| **Humorous hyperbole / sincere stress** (joke ↔ reaction) | *"I have aged 10 years in 3 laps watching this."* | Rule #3 tie-breaker: sincere emotion via exaggeration → **`reaction`** unless it's a constructed bit. |
| **Sarcasm carrying an opinion** (joke ↔ hot_take) | *"Ah yes, brilliant strategy Ferrari, galaxy-brained stuff."* | Rule #3: punchline-structured sarcasm → **`joke`**, even though it encodes a real take. |
| **Fact recitation** (analysis ↔ other) | *"DRS only works within 1s at the detection point."* | A fact with no claim/argument is **not** `analysis` → filter as `other`. |

**Documented hard cases from annotation (Milestone 3).** Three genuinely difficult comments and the decision made:

1. **`hot_take` vs `analysis`** — *"Apart from that time a few months ago where he demanded a journalist leave a press conference because he said something Verstappen didn't like… but it's back to acting like it's Saint Verstappen again."* It cites a specific event, which looks like evidence. **Decided `hot_take`:** by rule #1, the event is *decorative* — used to score a rhetorical point against the "Saint Verstappen" narrative, not as load-bearing reasoning toward a checkable claim.
2. **`reaction` vs `hot_take`/`joke`** — *"Max being the first F1 driver to win in a Ferrari this year is diabolical."* Reads as a jab, but is it an argued verdict or a bit? **Decided `reaction`:** by rule #2/#3 it's an amused in-the-moment exclamation with no debatable position and no constructed punchline — it's venting wry delight, not asserting a take.
3. **`analysis` vs `hot_take` — left BLANK for review** — *"Except that's not corruption. Corruption is knowingly using your position of power for personal or collective gain… and until now there hasn't been anything to indicate that Masi['s] doings were acts of corruption."* A calm, reasoned definitional argument — but the reasoning is purely conceptual with **no specific, verifiable F1 evidence**, so it fails the load-bearing test for `analysis`, yet it's far too measured to be a clean `hot_take`. Neither rule fires cleanly, so it is one of the 7 comments left **blank** for a second pass. (The other blanks are flagged with reasons in the dataset's `notes` column.)

---

## 4. Data collection plan

- **Source:** public Reddit comments — r/F1Technical (analysis-dense, 18 examples) and r/formula1 (everything else, 189 examples). Public content only; no authenticated or private channels.
- **Collection method:** **Apify** (`automation-lab/reddit-scraper`). *PRAW was the original plan, but Reddit's "Responsible Builder Policy" blocked API-app creation and public `.json` endpoints return 403, so we pivoted to Apify.* Three runs built the pool: (1) r/F1Technical + r/formula1 top-of-year posts; (2) r/formula1 "Race Discussion" threads (hot-take dense); (3) r/formula1 "Race Thread" threads (reaction dense — the scarcest register). ~$0.9 of Apify credits. Pipeline: raw scrape → [scripts/build_pool.py](scripts/build_pool.py) (clean/dedupe/filter) → label → [scripts/build_dataset.py](scripts/build_dataset.py) (join text + labels → final CSV). The PRAW [scripts/collect.py](scripts/collect.py) is retained for reproducibility if creds become available.
- **Unit:** one comment = one example.
- **How many per label:** target **~50 each (200 total)**, with a **hard floor of ≥40 (20%) per label**. (The milestone's ceiling is no label above 70%; my target is far stricter so no class dominates.)
- **Sourcing strategy (stratified so balance is built in, not hoped for):**

  | Label | Where it's densest |
  |---|---|
  | `analysis` | r/F1Technical top comments; r/formula1 strategy/technical threads |
  | `hot_take` | r/formula1 "unpopular opinion" / driver-ranking / tier-list threads |
  | `reaction` | r/formula1 live race-thread comments around key moments |
  | `joke` | r/formula1 top-voted funny comments |

- **If a label is underrepresented after 200:** do a **targeted top-up** — return to that label's densest source (e.g., more r/F1Technical threads for `analysis`, more race threads for `reaction`) and collect only that class until it clears the ≥40 floor, then re-count. I will not down-sample the majority classes below ~50; I add to the minority.
- **Pre-collection validation gate:** read 30–40 collected comments first and confirm the labels apply cleanly *before* annotating all 200; revise §2–3 if not.
- **CSV format:** a **single** file `data/takemeter_dataset.csv` with columns **`text, label, notes`** (plus `id, source_sub` for provenance). **No pre-split / no `split` column** — the Colab notebook performs the 70/15/15 train/val/test split automatically.

---

## 5. Evaluation metrics

Accuracy alone is insufficient for this task: with four classes on a subjective boundary, a model can post a respectable accuracy while quietly collapsing the rare, high-value `analysis` class into `hot_take`. So the report uses a metric *set*, each chosen for a reason specific to TakeMeter:

- **Overall accuracy** — a baseline sanity figure and the headline comparison vs. the Groq baseline, but never the sole metric.
- **Macro-averaged F1 (primary metric)** — weights every class equally, so the model cannot win by nailing the high-frequency registers (`joke`, `reaction`) while failing `analysis`. For a *discourse-quality* tool, getting the rare substantive class right matters as much as the common ones — macro-F1 captures exactly that.
- **Per-class precision / recall / F1** — because the cost of confusions is asymmetric. I care most about **recall on `analysis`** (a useful tool must not silently drop genuine substance) and **precision on `analysis`** (it must not flatter low-quality posts as analysis).
- **4×4 confusion matrix** — diagnostic, not just a score: it shows *which* registers get confused. I predict the hot confusions will be `analysis`↔`hot_take` (the tone-vs-substance boundary) and `joke`↔`reaction` (humor-vs-sincerity).

Both the fine-tuned model and the Groq zero-shot baseline are scored on the identical held-out test set.

---

## 6. Definition of success

Concrete, objectively checkable thresholds on the held-out test set:

- **Primary:** macro-F1 **≥ 0.70**.
- **Relative (did fine-tuning earn its keep?):** the fine-tuned model beats the Groq zero-shot baseline's macro-F1 by **≥ 0.10**. If it doesn't, fine-tuning added little and I'll say so.
- **Per-class floor:** **recall ≥ 0.60 on `analysis`** — the hardest, highest-value class.

**"Good enough for deployment" in a real community tool:** higher bar — **macro-F1 ≥ 0.75 and `analysis` precision ≥ 0.70**, so that posts the tool *promotes* as analysis are usually genuinely analysis (precision matters most when you act on the positive label). Below that bar, it's only fit as a **human-in-the-loop assist** (suggesting a label for a moderator to confirm), not an autonomous classifier.

**Honesty caveat:** this is a subjective task labeled by a single annotator, so even two humans wouldn't agree 100% — a realistic performance ceiling is well below 1.0. Inter-annotator agreement (stretch feature) would let me contextualize that ceiling; without it, I treat ~0.70 macro-F1 as a strong result, not a disappointing one.

---

## 7. AI Tool Plan

This project has little code to generate; AI tools help in three specific places.

- **Label stress-testing — DONE (used Claude).** Before annotating, I had the assistant generate ~8 boundary posts and classify each with these definitions. It validated most boundaries and exposed one weak spot — `reaction` vs `hot_take` for *evaluative venting* — which I fixed by sharpening **rule #2** (debatable verdict → `hot_take`; hyperbolic venting → `reaction`). Outcome recorded in §3.
- **Annotation assistance — REVISED DECISION: LLM pre-labeling, human review.** The original plan was to hand-label all 200. In practice, at the user's request, **Claude (Opus 4.8) pre-labeled every comment** by applying the §2 definitions and §3 rules, leaving 7 genuinely ambiguous cases blank. **These are AI-*suggested* labels and require the annotator's review/correction before the dataset is treated as ground truth** — this is mandatory, not optional. **Known risk being accepted:** because the Groq baseline (Milestone 4) is also an LLM, LLM-origin labels make the evaluation partly a measure of "does DistilBERT imitate an LLM"; human review is what restores the labels as genuine human-intended ground truth, so skimming the review would invalidate the comparison. Inter-annotator agreement (stretch) would directly quantify how much the human pass changed.
- **Failure analysis — PLANNED (will use Claude).** After evaluation, I'll give the assistant the list of wrong test predictions (`text`, true label, predicted label) and ask it to propose *systematic* patterns (e.g., "evidenced rants misread as `hot_take` via CAPS/profanity"). **Verification:** I will not accept a pattern on the AI's word — I'll re-read every cited example myself and confirm the pattern holds in the confusion-matrix counts before it goes in the writeup.

### AI usage disclosure
Claude (Opus 4.8, via Claude Code) was used to: brainstorm the label taxonomy and edge-case rules, run the label stress-test, scrape the data (Apify), build the cleaning/merge pipeline, draft this `planning.md`, and **pre-label all 200 examples** (see Annotation assistance above). Planned future use: failure-pattern analysis (verified manually). **The current labels in `data/takemeter_dataset.csv` are AI-suggested and pending human review;** the 7 blank rows await a human decision. This disclosure must be carried into the README's AI-usage section.

---

## 8. Model & approach (context for later milestones)

- **Fine-tune** `distilbert-base-uncased` + 4-class head in the Colab starter notebook (free T4 GPU).
- **Key hyperparameter decision → epochs.** With ~140 training rows, overfitting is the dominant risk, so train for a *low* epoch count (≈4) and keep the best-by-validation checkpoint rather than training long.
- **Baseline:** Groq `llama-3.3-70b-versatile`, true zero-shot (label definitions + rules in the prompt, no example posts), same test set.

---

## 9. Deliverables & repo structure

```
ai201-project3-takemeter/
├── planning.md                      # this file
├── README.md                        # graded writeup (filled in later milestones)
├── requirements.txt
├── data/takemeter_dataset.csv       # 200 labeled comments (text,label,notes,+provenance) — single file, no split
├── scripts/collect.py               # PRAW collector
├── notebook/label_map.py            # label↔id mapping for the notebook
├── notebook/groq_prompt.md          # zero-shot baseline prompt
└── results/                         # evaluation_results.json + confusion_matrix.png (from Colab)
```

---

## 10. Stretch features (update this section before starting each)

- [ ] **Inter-annotator reliability** — 1 other person labels 30+ examples; report Cohen's κ + disagreement analysis. *(Also contextualizes the §6 performance ceiling.)*
- [ ] **Confidence calibration** — does a 90%-confident prediction beat a 60%-confident one?
- [ ] **Error-pattern analysis** *(most natural fit)* — a *systematic* pattern, e.g. sound rants misread as `hot_take` via tone cues.
- [ ] **Deployed interface** — input a comment → show predicted label + confidence.

---

## 11. Open assumptions

- The Colab starter notebook performs the 70/15/15 split; if its split proves non-stratified and a class is starved in the test set, restore stratified splitting (removed from `scripts/`, recoverable from git history).
- Fine-tuning runs in the user's Colab notebook (this environment has no GPU); local deliverables are the dataset, collector, prompt, notebook config, and docs.
