import json
import pandas as pd
from pathlib import Path

folder = Path("data/raw/google_books")

rows = []

for file_path in folder.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        rows.append(json.load(f))

df = pd.DataFrame(rows)

print(f"Loaded {len(df)} saved Google Books records")
print(df.head())
print(df.columns.tolist())

df.to_csv("data/raw/nyt_google_enriched.csv", index=False)

print("Saved: data/raw/nyt_google_enriched.csv")