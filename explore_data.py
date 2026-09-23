import pandas as pd

df = pd.read_csv("raw-data/re_dataset.csv", encoding="latin-1")  

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nLabel distribution (HS):")
print(df["HS"].value_counts())