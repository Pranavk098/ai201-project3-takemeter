import pandas as pd
from scripts.make_splits import add_stratified_split


def _toy():
    labels = ["analysis", "hot_take", "reaction", "joke"] * 10  # 40 rows, balanced
    return pd.DataFrame({"id": range(40), "text": ["x"] * 40, "label": labels})


def test_every_row_gets_a_split():
    out = add_stratified_split(_toy())
    assert out["split"].isin(["train", "val", "test"]).all()
    assert out["split"].notna().all()


def test_splits_are_disjoint_no_leakage():
    out = add_stratified_split(_toy())
    by = {s: set(out[out.split == s]["id"]) for s in ["train", "val", "test"]}
    assert by["train"].isdisjoint(by["test"])
    assert by["train"].isdisjoint(by["val"])
    assert by["val"].isdisjoint(by["test"])


def test_each_split_has_all_four_labels():
    out = add_stratified_split(_toy())
    for s in ["train", "val", "test"]:
        assert set(out[out.split == s]["label"]) == {"analysis", "hot_take", "reaction", "joke"}
