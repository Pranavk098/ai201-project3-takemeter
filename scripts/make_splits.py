"""Add a stratified 70/15/15 train/val/test split column."""
import pandas as pd
from sklearn.model_selection import train_test_split


def add_stratified_split(df, seed=42):
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=seed)
    df = df.copy()
    df["split"] = None
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/takemeter_dataset.csv")
    df = add_stratified_split(df)
    df.to_csv("data/takemeter_dataset.csv", index=False)
    print(df.groupby(["split", "label"]).size())
