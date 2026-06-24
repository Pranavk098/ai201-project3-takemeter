"""Shared stratified 70/15/15 split so the baseline and the fine-tuned model are
scored on the IDENTICAL test set. Blank-label rows (ambiguous, pending review) are dropped.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["analysis", "hot_take", "reaction", "joke"]
CSV = "data/takemeter_dataset.csv"


def load_splits(seed=42):
    df = pd.read_csv(CSV)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)  # drop the 7 blanks
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=seed)
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


if __name__ == "__main__":
    sp = load_splits()
    for name, d in sp.items():
        print(name, len(d), dict(d["label"].value_counts()))
