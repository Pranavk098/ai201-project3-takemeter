"""Classify F1 comments with the fine-tuned model (label + confidence).

Usage:
    python scripts/predict.py "your comment here" "another comment"
    python scripts/predict.py        # runs a few built-in demo comments
Requires results/ft_model/ (produced by finetune_eval.py).
"""
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "results/ft_model"

DEMOS = [
    "Leclerc lost that on strategy not pace - he was 0.3s/lap quicker on mediums but Ferrari left him out 4 laps too long and the undercut was gone.",
    "Verstappen is the most overrated champion in history, put him in a midfield car and he's nothing.",
    "NOOO not again Ferrari I genuinely can't watch this team anymore",
    "Ferrari's strategy department was last seen taking orders from a magic 8-ball.",
]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    texts = sys.argv[1:] or DEMOS
    for t in texts:
        inp = tok(t, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**inp).logits[0], dim=-1)
        i = int(probs.argmax())
        print(f"{probs[i]:.2f}  {model.config.id2label[i]:9s}  {t[:80]}")


if __name__ == "__main__":
    main()
