"""Inter-annotator reliability: a second annotator vs the AI-suggested labels.

Workflow:
  1. Open data/iaa_blind.csv and fill the `your_label` column INDEPENDENTLY
     (one of: analysis / hot_take / reaction / joke) without looking at data/iaa_key.csv.
  2. Run: python scripts/iaa.py
It reports percent agreement + Cohen's kappa and lists every disagreement.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

LABELS = ["analysis", "hot_take", "reaction", "joke"]


def main():
    blind = pd.read_csv("data/iaa_blind.csv")
    key = pd.read_csv("data/iaa_key.csv")
    m = blind.merge(key, on="idx")
    m = m[m["your_label"].astype(str).str.strip().isin(LABELS)]
    if len(m) < 30:
        print(f"Only {len(m)} rows have a valid label in `your_label`. "
              f"Fill all of data/iaa_blind.csv (>=30) first.")
        return
    human = m["your_label"].str.strip()
    ai = m["ai_label"]
    agree = (human.values == ai.values).mean()
    kappa = cohen_kappa_score(human, ai, labels=LABELS)
    print(f"n={len(m)}  percent agreement={agree:.1%}  Cohen's kappa={kappa:.3f}")
    dis = m[human.values != ai.values]
    print(f"\n{len(dis)} disagreements:")
    for _, r in dis.iterrows():
        print(f"  you={str(r['your_label']).strip():9s} ai={r['ai_label']:9s} | {r['text'][:80]}")


if __name__ == "__main__":
    main()
