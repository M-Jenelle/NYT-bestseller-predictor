import pandas as pd

df = pd.read_csv("data/raw/nyt_bestsellers_multiple_weeks.csv")

# fix isbn10 float issue
df["primary_isbn10"] = (
    df["primary_isbn10"]
    .astype("string")
    .str.replace(r"\.0$", "", regex=True)
    .replace({"nan": pd.NA, "None": pd.NA, "0": pd.NA})
)

print(df['primary_isbn10'].head())

print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.dtypes)

# Basic counts
print(f"Total rows: {len(df)}")
print(f"Unique titles: {df['title'].nunique()}")
print(f"Unique ISBNs: {df['primary_isbn13'].nunique()}")

# Books that appear most often in your collected data
print("\nBooks that appear most often:")
print(
    df.groupby(["title", "author"])
    .size()
    .sort_values(ascending=False)
    .head(20)
)

# Longest-running bestsellers based on NYT weeks_on_list
print("\nLongest-running bestsellers:")
print(
    df.groupby(["title", "author"])["weeks_on_list"]
    .max()
    .sort_values(ascending=False)
    .head(20)
)

# Create a safer book ID
df["book_id"] = df["primary_isbn13"]

df["book_id"] = df["book_id"].fillna(
    df["title"].astype(str) + " - " + df["author"].astype(str)
)

# Deduplicate books: keep one row per unique book
df_unique = (
    df.sort_values("weeks_on_list", ascending=False)
    .drop_duplicates(subset="book_id", keep="first")
    .reset_index(drop=True)
)

# Count how many different NYT lists each book appeared on
lists_per_book = (
    df.groupby("book_id")["list_name"]
    .nunique()
    .reset_index()
)

lists_per_book.columns = ["book_id", "num_lists_appeared"]

# Add num_lists_appeared to the deduplicated data
df_unique = df_unique.merge(lists_per_book, on="book_id", how="left")

print(f"\nUnique books ready for Step 2: {len(df_unique)}")

df_unique.to_csv("data/raw/nyt_unique_books.csv", index=False)

print("Saved cleaned unique books file to data/raw/nyt_unique_books.csv")