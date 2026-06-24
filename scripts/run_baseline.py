"""Milestone 4 baseline: Groq llama-3.3-70b-versatile, zero-shot, on the locked test set."""
import json
import os
import sys
import time

from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, "scripts")
from data_split import LABELS, load_splits

PROMPT = '''You are classifying the REGISTER of a comment from Formula 1 fan discussion into exactly one of four labels. Reply with ONLY the label in lowercase, nothing else.

Labels:
- analysis: a structured argument backed by specific, verifiable evidence (lap times, tire strategy, technical/regulation reasoning, statistics). The reasoning is the point.
- hot_take: a bold, confident opinion or judgment asserted without genuine evidence (rankings, overrated/underrated, predictions stated as fact). Any evidence is vague or decorative.
- reaction: an immediate emotional response to a specific event, with little or no argument.
- joke: a comment whose primary intent is humor - meme, pun, bit, sarcasm, copypasta.

Rules:
- If an emotional or ranty tone wraps a real, evidenced argument, it is still analysis (substance over tone).
- If the comedic construction is clearly the point, it is joke; if it is sincere emotion that merely happens to be hyperbolic, use the sincere label.

Comment:
"""{text}"""

Label:'''


def normalize(resp):
    r = (resp or "").strip().lower()
    for lab in LABELS:
        if lab in r:
            return lab
    return "unknown"


def classify(client, text):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": PROMPT.format(text=text)}],
                temperature=0,
                max_tokens=8,
            )
            return r.choices[0].message.content
        except Exception as e:  # rate limit / transient
            wait = 5 * (attempt + 1)
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)
    return ""


def main():
    load_dotenv()
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    test = load_splits()["test"]
    y_true = list(test["label"])
    texts = list(test["text"])

    preds, unparseable = [], 0
    for i, t in enumerate(texts):
        lab = normalize(classify(client, t))
        if lab == "unknown":
            unparseable += 1
        preds.append(lab)
        print(f"[{i+1}/{len(texts)}] true={y_true[i]:9s} pred={lab}")
        time.sleep(0.4)

    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, labels=LABELS, output_dict=True, zero_division=0)
    os.makedirs("results", exist_ok=True)
    json.dump({"model": "llama-3.3-70b-versatile (zero-shot)", "accuracy": acc,
               "report": report, "y_true": y_true, "y_pred": preds, "texts": texts,
               "unparseable": unparseable, "n": len(texts)},
              open("results/baseline.json", "w"), indent=2)
    print(f"\nBASELINE accuracy: {acc:.3f} | unparseable: {unparseable}/{len(texts)} "
          f"({unparseable/len(texts)*100:.0f}%)")
    print(classification_report(y_true, preds, labels=LABELS, zero_division=0))


if __name__ == "__main__":
    main()
