# TakeMeter — Planning

*Project 3, CodePath Applications of AI. Last updated: 2026-06-22.*

A fine-tuned text classifier that labels the **register of F1 discourse** on Reddit. Given a comment, it predicts one of four labels: `analysis`, `hot_take`, `reaction`, or `joke`.

---

## 1. Community & why these labels matter

**Community:** Formula 1 fans on Reddit — primarily [r/formula1](https://reddit.com/r/formula1) (the main hub: race threads, news, opinions, memes) with [r/F1Technical](https://reddit.com/r/F1Technical) as a denser source of genuine analysis.

F1 discourse moves between very different registers depending on the moment: a strategy post-mortem reads nothing like a live-thread scream of joy, which reads nothing like a meme about Ferrari's pit wall. Regulars instinctively distinguish a *substantive breakdown* from a *spicy assertion* from an *in-the-moment reaction* from a *bit* — but those distinctions are exactly what's hard to pin down in writing. That's the task.

**Important framing:** because one of the labels (`joke`) is not a "take," TakeMeter classifies *register/type*, not quality on a single axis. Quality is the implied ordering: `analysis` > `hot_take` > `reaction` > `joke`. The README states this explicitly.

---

## 2. Label taxonomy (4 labels)

> The examples below are **illustrative templates** written to show each boundary. The real 200 annotated examples come from scraped comments in Milestone 2.

### `analysis`
**Definition:** A structured argument backed by *specific, verifiable evidence* — lap-time/sector comparisons, tire-strategy reasoning (undercut/overcut, stint length, deg), technical/aero/regulation explanation, or historical/statistical comparison. The reasoning is the point; the claim is checkable.

- ✅ *"Leclerc lost that on strategy, not pace — he was ~0.3s/lap quicker on mediums but Ferrari left him out 4 laps too long, came back into traffic, and the undercut was gone."*
- ✅ *"Red Bull's high-speed advantage is the anti-dive front suspension — it holds ride height under braking, protecting underfloor load. Way less pitch than Ferrari on the onboards."*
- ❓ *Uncertain:* *"DRS gives a straight-line boost when you're within a second."* — true and technical, but a well-known fact with no argument or situation-specific claim. **Decision:** pure common-knowledge info with no claim → filter as `other` at collection, **not** `analysis`. Analysis requires reasoning toward a claim, not reciting a fact.

### `hot_take`
**Definition:** A bold, confident *opinion or judgment* asserted **without** genuine supporting evidence. The claim might be true, but it declares rather than argues; any evidence is vague, cherry-picked, or decorative. Usually evaluative — rankings, overrated/underrated, "should be fired," predictions stated as fact.

- ✅ *"Verstappen is the most overrated champion in history. Put him in a midfield car and he's nothing."*
- ✅ *"Hamilton is washed, he should retire. Embarrassing to watch."*
- ❓ *Uncertain:* *"Norris wins the title this year, easy."* — a confident prediction with no evidence. **Decision:** an unevidenced confident claim is a `hot_take` even when it's a prediction rather than a rating.

### `reaction`
**Definition:** An immediate *emotional response to a specific event* — excitement, despair, anger, awe — with little or no argument. Venting a feeling in the moment.

- ✅ *"OH MY GOD WHAT A MOVE BY LANDO INTO TURN 1, I'M SHAKING"*
- ✅ *"not again Ferrari… I genuinely can't watch this team anymore 😭"*
- ❓ *Uncertain:* *"That penalty was an absolute joke."* — pure in-the-moment venting (→ `reaction`) but also an evaluative verdict on the stewards (→ `hot_take`). **Decision:** if it's a bare exclamation of displeasure with no position to argue → `reaction`; if it asserts a defensible verdict ("the stewards were wrong because…") → `hot_take`. This bare version leans `reaction`. Genuinely hard — flagged as a documented edge case.

### `joke`
**Definition:** A post whose *primary intent is humor* — meme, pun, bit, copypasta, sarcastic shitpost, running gag. The point is to be funny, not to inform, judge, or sincerely emote.

- ✅ *"Ferrari's strategy dept was last seen taking orders from a magic 8-ball."*
- ✅ *"BREAKING: Stroll qualifies P18, blames the car, the wind, and the concept of time itself."*
- ❓ *Uncertain:* *"Yeah Stroll definitely earned that seat on merit 🙄"* — sarcasm that encodes a real opinion (pay driver). **Decision:** if it's structured as a bit/punchline → `joke`; if it's the author's plainly-meant opinion delivered drily → `hot_take`. Default to `joke` when comedic construction is clearly the point.

---

## 3. Decision rules (boundary resolution)

These are the rules applied when a comment could plausibly fit two labels.

1. **`hot_take` vs `analysis`** — Strip the opinion *and emotional/rant framing*. If the remaining evidence is **load-bearing** (genuinely establishes the claim) → `analysis`. If it's **decorative** (cherry-picked, vague, propping up a pre-formed verdict) → `hot_take`. *Substance over tone: a furious rant with a real argument underneath is still `analysis`.*
2. **`reaction` vs `hot_take`** — Does it assert an evaluative *position* you could agree/disagree with? → `hot_take`. Is it just venting feeling about a just-happened event? → `reaction`.
3. **`joke` vs `reaction`/`hot_take`** — If the *comedic construction is clearly the point* (bit, punchline, meme format, exaggeration-for-laughs) → `joke`. If it's sincere emotion/opinion that merely happens to be hyperbolic → the sincere register. **Tie-breaker:** when genuinely 50/50, default to the sincere category, *not* `joke`.
4. **Fits none** (pure questions, news/info dumps, off-topic) → **filtered out at collection**, never labeled, keeping the `other` bucket at 0%.

### Worked ambiguous example (the technically-sound rant)
> *"I'm SO sick of people saying Leclerc threw it. He did NOT. He pitted lap 18 on the undercut, came out P3, then the wall pitted Sainz a lap later onto the same medium for no reason, double-stacked them and cost Charles ~5s in the pit lane. The hard was 0.8s/lap slower in those temps so the overcut was dead anyway."*

Could be `hot_take` (angry, accusatory) or `analysis` (cites specifics). **Applying rule #1:** strip the anger and a specific, verifiable causal argument remains → **`analysis`**. This is also a predicted model failure mode (the model will lean on CAPS/profanity and may call it `hot_take`/`reaction`) and is reserved for the README's "model vs. intent" reflection.

### Mutual-exclusivity & exhaustiveness check
- **Exclusive:** the rules above assign exactly one label to each borderline case. ✔
- **Exhaustive:** `analysis`/`hot_take`/`reaction`/`joke` plus collection-time filtering of non-takes is expected to cover ≥90% of comments without a catch-all. ✔ (Validated against 30–40 real comments before committing — see M2.)

---

## 4. Data collection & annotation (Milestone 2)

- **Unit:** one comment = one example.
- **Target:** ~200 total, ~50 per label, **hard floor ≥40 (20%) per label**.
- **Sourcing (stratified so balance is built in):**

  | Label | Where it's densest |
  |---|---|
  | `analysis` | r/F1Technical top comments; r/formula1 strategy/technical threads |
  | `hot_take` | r/formula1 "unpopular opinion" / driver-ranking / tier-list threads |
  | `reaction` | r/formula1 live race thread comments (around key moments) |
  | `joke` | r/formula1 top-voted funny comments (r/formuladank is mostly image memes — thin on text) |

- **Collection method:** **PRAW** (Python Reddit API Wrapper). A small local script pulls a pool of comments per subreddit/thread into a CSV; comments are then hand-labeled and balanced. Free; needs Reddit API creds (client_id/secret from reddit.com/prefs/apps). *Easily swappable for manual collection or an Apify actor if creds are a hassle.*
- **Filtering at collection:** drop pure questions, news/info dumps, off-topic → `other` stays at 0%.
- **Validation step:** read 30–40 collected comments first to confirm the labels apply cleanly *before* annotating all 200; revise definitions if not.
- **CSV schema:** `id, text, label, source_sub, split, notes` — `notes` records *why* hard cases were labeled (feeds the README's "3 difficult examples").
- **Splits:** stratified **70/15/15 → 140 train / 30 val / 30 test**, each split ~25% per class. **Test set held out until final eval** (guards against leakage inflating accuracy).

---

## 5. Fine-tuning (Milestone 3)

- **Base model:** `distilbert-base-uncased` + 4-class classification head.
- **Approach:** HuggingFace `transformers` `Trainer` in the Colab starter notebook (free T4 GPU).
- **Key hyperparameter decision → number of epochs.** With only ~140 training examples, **overfitting is the dominant risk**, so train for a *low* epoch count (≈4) and select the best checkpoint by validation performance rather than training long. (Defaults: lr 2e-5–5e-5, batch 16, max_len 256.) Documented with rationale in the README.

---

## 6. Baseline comparison (Milestone 4)

- **Model:** Groq `llama-3.3-70b-versatile`, **true zero-shot**.
- **Prompt:** contains the 4 label *definitions* and decision rules but **no example posts**; must return exactly one label. Run on the **same 30 test comments** as the fine-tuned model.

---

## 7. Evaluation report (Milestone 5)

For both models on the same test set:
- Overall **accuracy**.
- **Per-class precision / recall / F1**.
- **4×4 confusion matrix** (`confusion_matrix.png`).
- **≥3 misclassified examples** with analysis of *why*.
- **Reflection: what the model learned vs. what I intended** — anchored on the tone-vs-substance failure mode (does it over-rely on CAPS/profanity/emoji as surface cues?).
- Outputs: `evaluation_results.json`, `confusion_matrix.png`.

---

## 8. Deliverables & repo structure

```
ai201-project3-takemeter/
├── planning.md                 # this file
├── README.md                   # community, labels, data, distribution, 3 hard examples, eval report
├── data/takemeter_dataset.csv  # 200 labeled comments + split column
├── scripts/collect.py          # PRAW collection script (M2)
├── scripts/baseline_groq.py    # zero-shot baseline (M4) — or run in notebook
├── results/evaluation_results.json
└── results/confusion_matrix.png
```

---

## 9. Stretch features (update this section before starting each)

- [ ] **Inter-annotator reliability** — 1 other person labels 30+ examples; report Cohen's κ + disagreement analysis.
- [ ] **Confidence calibration** — does a 90%-confident prediction beat a 60%-confident one?
- [ ] **Error-pattern analysis** *(most natural fit)* — find a *systematic* pattern, e.g. "sound rants get misread as hot_take via tone cues."
- [ ] **Deployed interface** — input a comment → show predicted label + confidence.

---

## 10. Open assumptions

- PRAW chosen as default collector; falls back to manual/Apify if Reddit creds are blocked.
- Fine-tuning runs in the user's Colab notebook (this environment can't run a GPU); deliverables here are the dataset, scripts, prompts, notebook config, and docs.
- Split ratio 70/15/15 assumed; may adjust to match the starter notebook's expected input format.
