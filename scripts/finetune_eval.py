"""Milestones 5: fine-tune distilbert-base-uncased, evaluate on the locked test set,
write evaluation_results.json (fine-tuned + baseline) and confusion_matrix.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

sys.path.insert(0, "scripts")
from data_split import LABELS, load_splits

MODEL = "distilbert-base-uncased"
MAX_LEN = 256
EPOCHS = 10
LR = 4e-5
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


def main():
    sp = load_splits()
    tok = AutoTokenizer.from_pretrained(MODEL)

    def make(df):
        ds = Dataset.from_dict({"text": list(df["text"]),
                                "labels": [label2id[l] for l in df["label"]]})
        return ds.map(lambda b: tok(b["text"], truncation=True,
                                    padding="max_length", max_length=MAX_LEN), batched=True)

    train_ds, test_ds = make(sp["train"]), make(sp["test"])
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id)

    args = TrainingArguments(
        output_dir="results/ft", num_train_epochs=EPOCHS, learning_rate=LR,
        per_device_train_batch_size=16, warmup_ratio=0.1, weight_decay=0.01,
        logging_steps=10, seed=42, save_strategy="no", report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)
    trainer.train()

    out = trainer.predict(test_ds)
    logits = out.predictions
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    conf = probs.max(axis=1)
    y_pred = [id2label[i] for i in np.argmax(logits, axis=1)]
    y_true = [id2label[i] for i in out.label_ids]
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(cmap="Blues", xticks_rotation=45)
    plt.title("Fine-tuned DistilBERT - Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=120)

    trainer.save_model("results/ft_model")
    tok.save_pretrained("results/ft_model")

    result = {"finetuned": {"model": MODEL, "epochs": EPOCHS, "lr": LR, "batch_size": 16,
                            "max_len": MAX_LEN, "accuracy": acc, "report": report,
                            "y_true": y_true, "y_pred": y_pred,
                            "confidences": [round(float(c), 4) for c in conf],
                            "texts": list(sp["test"]["text"])}}
    if os.path.exists("results/baseline.json"):
        b = json.load(open("results/baseline.json"))
        result["baseline_groq"] = {"model": b["model"], "accuracy": b["accuracy"],
                                   "report": b["report"], "y_pred": b["y_pred"]}
    json.dump(result, open("results/evaluation_results.json", "w"), indent=2)

    print(f"\nFINE-TUNED accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    if "baseline_groq" in result:
        print(f"baseline (Groq) accuracy: {result['baseline_groq']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
