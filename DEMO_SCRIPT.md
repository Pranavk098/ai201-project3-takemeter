# TakeMeter — Demo Video Script & Prep (target 3:00–3:30)

Everything you need to record the demo in one take. The model outputs below are **verified** — they will match what appears on screen.

---

## 1. Pre-recording checklist

1. Open a terminal in the project folder with the venv active:
   ```
   .venv\Scripts\activate
   python app.py
   ```
   Wait for `Running on local URL: http://127.0.0.1:7860` (or whatever port it prints) and open that URL in your browser.
2. In a second window, open **README.md** (rendered on GitHub, or in a Markdown preview) scrolled to the **Evaluation report** section.
3. Screen-record both the browser (for the live demo) and the README (for the walkthrough). 1080p, show the confidence bars clearly.
4. Optional fallback if the browser misbehaves: `python scripts/predict.py "<comment>"` prints the label + confidence in the terminal.

> The 4 posts you'll classify are the **example buttons** built into the app — just click them, no typing needed.

## 2. The four demo posts and their verified outputs

| # | Post (click the example) | Model output | Verdict |
|---|---|---|---|
| 1 | "Leclerc lost that on strategy not pace — he was 0.3s/lap quicker on mediums but Ferrari left him out 4 laps too long and the undercut was gone." | **analysis 0.93** | ✅ correct (narrate this one) |
| 2 | "Verstappen is the most overrated champion in history, put him in a midfield car and he's nothing." | **hot_take 0.90** | ✅ correct |
| 3 | "NOOO not again Ferrari I genuinely can't watch this team anymore" | **reaction 0.78** | ✅ correct |
| 4 | "Ferrari's strategy department was last seen taking orders from a magic 8-ball." | **hot_take 0.48** | ❌ incorrect — should be `joke` (narrate this one) |

---

## 3. Full script (narration + screen actions)

### [0:00–0:25] Intro
*(Screen: the app open in the browser, or the README title.)*

> "Hi — this is **TakeMeter**, a text classifier that labels the *register* of Formula 1 Reddit comments into four types: **analysis**, **hot_take**, **reaction**, and **joke**. I defined those labels, hand-curated 200 comments, and fine-tuned DistilBERT on them. Let me show it classifying some posts live."

### [0:25–1:10] Live demo — a correct prediction (narrate in depth)
*(Screen: click example #1. The label bars appear.)*

> "First post: *'Leclerc lost that on strategy not pace — he was 0.3 seconds a lap quicker on mediums but Ferrari left him out four laps too long and the undercut was gone.'* The model says **analysis** with **0.93 confidence** — and that's right. This makes a specific, *checkable* argument: a lap-time delta, a strategy timeline, a concrete cause. That evidence-based reasoning is exactly what my `analysis` label means, so a high-confidence `analysis` call is the model working as intended."

### [1:10–1:40] Two more correct predictions (quick)
*(Screen: click example #2, then #3.)*

> "Next: *'Verstappen is the most overrated champion in history…'* — **hot_take, 0.90**. Correct: a bold verdict with no supporting evidence. And *'NOOO not again Ferrari, I can't watch this team anymore'* — **reaction, 0.78**. Correct again: pure in-the-moment emotion, no argument."

### [1:40–2:20] An incorrect prediction (narrate what went wrong)
*(Screen: click example #4.)*

> "Now a failure. *'Ferrari's strategy department was last seen taking orders from a magic 8-ball.'* That's clearly a **joke** — but the model predicts **hot_take**, and only at **0.48 confidence**. Two things stand out. First, that low confidence shows the model itself is unsure. Second, it latched onto the critical-of-Ferrari *sentiment* and read it as an opinion — it missed that the *intent* is comedic. This is my model's core weakness: it keys on vocabulary and tone, not on comedic construction, so jokes that sound like takes slip through."

### [2:20–3:15] Evaluation report walkthrough
*(Screen: switch to README → Evaluation report.)*

> "Here's how it scored on the 30-comment test set. My honest headline result: the **zero-shot Llama-70B baseline got 0.80 accuracy**, and my **fine-tuned model got 0.73** — the big zero-shot model actually *beats* fine-tuning on only 140 training examples.
>
> The confusion matrix shows the dominant problem: **`joke` and `reaction` account for four of the eight errors** — both are short and emotional, and separating humor from sincere emotion in a few words is genuinely hard, even for me as the annotator. Per class, the model actually *nailed* `hot_take` at F1 0.86; `joke` is weakest at 0.62. And confidence is meaningful — high-confidence predictions are right about 83% of the time versus 50% for low-confidence ones."

### [3:15–3:30] Closing
*(Screen: the 'what the model learned vs intended' section.)*

> "The big-picture gap: I defined my labels by **intent** — *why* a comment was written — but the model learned **vocabulary** — *which words* it contains. That's why technical jargon drags it toward `analysis` and why it can't reliably catch a joke. Thanks for watching."

---

## 4. Tips
- Read the posts off the screen so the viewer can follow along; keep your pace relaxed — this script is ~470 words, which lands around 3:15 at a normal speaking rate.
- If you go long, trim the two quick correct predictions (section [1:10–1:40]) to one.
- Make sure the **confidence numbers/bars are legible** in the recording — that's an explicit grading requirement.
- Upload (YouTube unlisted / Loom / Drive) and put the link at the top of your README and in your submission.
