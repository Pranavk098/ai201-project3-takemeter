"""Join labels.csv (id,label,notes) with comment text from the pools into the
final single-file dataset: data/takemeter_dataset.csv (text,label,notes,source_sub).
"""
import pandas as pd

POOLS = ["data/pool.csv", "data/pool2.csv", "data/pool3.csv"]
LABELS = ["analysis", "hot_take", "reaction", "joke"]


def main():
    # id -> (text, source_sub); first pool that has the id wins (text is identical per id)
    lookup = {}
    for p in POOLS:
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            lookup.setdefault(r["id"], (r["text"], r["source_sub"]))

    labels = pd.read_csv("data/labels.csv")
    missing = [i for i in labels["id"] if i not in lookup]
    if missing:
        raise SystemExit(f"{len(missing)} labeled ids not found in pools: {missing[:10]}")

    rows = []
    for _, r in labels.iterrows():
        text, sub = lookup[r["id"]]
        rows.append({
            "text": text,
            "label": "" if pd.isna(r["label"]) else r["label"],
            "notes": "" if pd.isna(r["notes"]) else r["notes"],
            "source_sub": sub,
        })
    out = pd.DataFrame(rows, columns=["text", "label", "notes", "source_sub"])
    out.to_csv("data/takemeter_dataset.csv", index=False)

    labeled = out[out["label"] != ""]
    counts = labeled["label"].value_counts()
    print(f"wrote {len(out)} rows -> data/takemeter_dataset.csv "
          f"({len(labeled)} labeled, {len(out) - len(labeled)} blank for review)")
    print(counts.to_string())
    print("max class share:", round(counts.max() / counts.sum() * 100, 1), "%")
    print("min class count:", counts.min())
    print("by source_sub:", out["source_sub"].value_counts().to_dict())


if __name__ == "__main__":
    main()
