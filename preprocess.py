import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("raw-data/re_dataset.csv", encoding="latin-1")

df = df[["Tweet", "HS"]].copy()
df.columns = ["text", "label"]

df["text"] = df["text"].str.replace(r"\\n", " ", regex=True)
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
df["text"] = df["text"].str.strip()

print("Missing/empty texts:", df["text"].isna().sum() + (df["text"] == "").sum())

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

train_df.to_csv("data_train.csv", index=False)
val_df.to_csv("data_val.csv", index=False)
test_df.to_csv("data_test.csv", index=False)

print("\nSaved: data_train.csv, data_val.csv, data_test.csv")