"""Inter-annotator reliability: two people label the same 32 comments independently.

data/iaa_blind.csv has columns: idx, text, annotator1, annotator2
Reports percent agreement + Cohen's kappa between the two annotators and lists disagreements.
"""
import sys
from collections import Counter

import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABELS = ["analysis", "hot_take", "reaction", "joke"]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    df = pd.read_csv("data/iaa_blind.csv")
    df["a1"] = df["annotator1"].astype(str).str.strip()
    df["a2"] = df["annotator2"].astype(str).str.strip()
    df = df[df["a1"].isin(LABELS) & df["a2"].isin(LABELS)]
    if len(df) < 30:
        print(f"Only {len(df)} rows have valid labels in both columns (need >=30).")
        return
    agree = (df["a1"] == df["a2"]).mean()
    kappa = cohen_kappa_score(df["a1"], df["a2"], labels=LABELS)
    print(f"n={len(df)}  percent agreement={agree:.1%}  Cohen's kappa={kappa:.3f}")

    dis = df[df["a1"] != df["a2"]]
    pairs = Counter(zip(dis["a1"], dis["a2"]))
    print(f"\n{len(dis)} disagreements; pairs (annotator1/annotator2):",
          {f"{x}/{y}": n for (x, y), n in pairs.most_common()})
    for _, r in dis.iterrows():
        print(f"  A1={r['a1']:9s} A2={r['a2']:9s} | {r['text'][:80]}")


if __name__ == "__main__":
    main()
