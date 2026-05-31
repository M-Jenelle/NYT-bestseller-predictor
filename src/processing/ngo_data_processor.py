import pandas as pd
import numpy as np
from pathlib import Path


Path("data/processed").mkdir(parents=True, exist_ok=True)

# load all three datasets
nyt_google = pd.read_csv("data/raw/nyt_google_enriched.csv")
open_lib = pd.read_csv("data/raw/open_library_enriched.csv")

print(f"NYT + Google rows: {len(nyt_google)}")
print(f"Open Library rows: {len(open_lib)}")
print(f"\nNYT + Google columns:\n{nyt_google.columns.tolist()}")
print(f"\nOpen Library columns:\n{open_lib.columns.tolist()}")

# Clean ISBN merge key
for df in [nyt_google, open_lib]:
    df["primary_isbn13"] = (
        df["primary_isbn13"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

# Drop Open Library duplicate ISBN rows to prevent row explosion
open_lib = open_lib.drop_duplicates(subset=["primary_isbn13"])

# Merge datasets
combined = nyt_google.merge(
    open_lib,
    on="primary_isbn13",
    how="left",
    suffixes=("_google", "_ol"),
    validate="m:1",
    indicator=True
)

print(f"\nCombined dataset shape: {combined.shape}")
print("\nMerge results:")
print(combined["_merge"].value_counts())

# Missing values BEFORE filling
missing = (combined.isnull().sum() / len(combined) * 100).round(1)
missing = missing[missing > 0].sort_values(ascending=False)

print("\nMissing data by column BEFORE cleaning (%):")
print(missing)


# Helper function to safely coalesce columns
def coalesce_columns(df, columns):
    existing_cols = [col for col in columns if col in df.columns]

    if not existing_cols:
        return pd.Series([pd.NA] * len(df), index=df.index)

    result = df[existing_cols[0]]

    for col in existing_cols[1:]:
        result = result.fillna(df[col])

    return result

# Create final author field
combined["final_author"] = coalesce_columns(
    combined,
    ["nyt_author", "google_authors", "ol_author_name"]
)

# Create final publisher field
combined["final_publisher"] = coalesce_columns(
    combined,
    ["nyt_publisher", "publisher_google", "ol_publisher"]
)

print("\nMissing final author/publisher:")
print(combined[["final_author", "final_publisher"]].isnull().sum())

# Data completeness score BEFORE imputation
key_fields = [
    "page_count",
    "categories",
    "average_rating",
    "is_series",
    "author_total_works",
    "description"
]

existing_key_fields = [col for col in key_fields if col in combined.columns]

combined["data_completeness_score"] = combined[existing_key_fields].notna().sum(axis=1)

print("\nData completeness score:")
print(combined["data_completeness_score"].value_counts().sort_index())


# ============================================================
# DATA AUDIT SECTION
# Put this BEFORE filling missing values
# ============================================================

print("\n==============================")
print("DATA AUDIT BEFORE IMPUTATION")
print("==============================")

# Open Library match coverage
print("\nOpen Library match coverage:")
print(combined["_merge"].value_counts(normalize=True).round(3) * 100)

# Page count audit
if "page_count" in combined.columns:
    combined["page_count"] = pd.to_numeric(combined["page_count"], errors="coerce")

    print("\nPage count summary:")
    print(combined["page_count"].describe())

    print("\nPage count zero count:")
    print((combined["page_count"] == 0).sum())

    print("\nPage count zero percentage:")
    print(((combined["page_count"] == 0).mean() * 100).round(1), "%")

    # Treat 0 page count as missing because 0 pages is not meaningful
    combined["page_count_clean"] = combined["page_count"].replace(0, np.nan)

    print("\nPage count clean missing percentage:")
    print((combined["page_count_clean"].isna().mean() * 100).round(1), "%")

# Ratings audit
for col in ["average_rating", "ratings_count"]:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

        print(f"\n{col}:")
        print(combined[col].describe())

        print("Missing count:")
        print(combined[col].isna().sum())

        print("Missing percentage:")
        print((combined[col].isna().mean() * 100).round(1), "%")

# Open Library feature coverage
ol_cols = [
    "ol_subjects",
    "ol_edition_count",
    "is_series",
    "author_total_works",
    "ol_ebook_access",
    "ol_has_fulltext",
    "ol_first_publish_year"
]

print("\nOpen Library feature coverage:")
for col in ol_cols:
    if col in combined.columns:
        coverage = combined[col].notna().mean() * 100
        print(f"{col}: {coverage.round(1)}%")


numeric_cols = [
    "page_count_clean",
    "average_rating",
    "ratings_count",
    "author_total_works",
    "ol_edition_count"
]

for col in numeric_cols:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

# Create missingness indicators before filling
for col in numeric_cols:
    if col in combined.columns:
        combined[f"{col}_missing"] = combined[col].isna()


# Fill numeric columns with median
# For now, I would NOT fill average_rating or ratings_count because they are too sparse.
numeric_cols_to_impute = [
    "page_count_clean",
    "author_total_works",
    "ol_edition_count"
]


for col in numeric_cols_to_impute:
    if col in combined.columns:
        median_val = combined[col].median()

        if pd.notna(median_val):
            combined[col] = combined[col].fillna(median_val)
            print(f"Filled {col} missing values with median: {median_val:.1f}")


# Fill categorical columns
categorical_cols = [
    "categories",
    "final_publisher",
    "language",
    "maturity_rating",
    "ol_ebook_access"
]

for col in categorical_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna("Unknown")


# Fill boolean columns
bool_cols = [
    "is_series",
    "ol_has_fulltext",
    "ol_public_scan"
]

for col in bool_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna(False)


# Fill Google description
if "google_description" in combined.columns:
    combined["google_description"] = combined["google_description"].fillna("")


# Drop columns you probably do not need for modeling
cols_to_drop = [
    "amazon_url",
    "ol_internet_archive_ids",
    "_merge"
]

combined = combined.drop(columns=[c for c in cols_to_drop if c in combined.columns])



# # Optional: create missingness indicators before filling
# for col in numeric_cols:
#     if col in combined.columns:
#         combined[f"{col}_missing"] = combined[col].isna()


# # Fill numeric columns with median
# for col in numeric_cols:
#     if col in combined.columns:
#         median_val = combined[col].median()

#         if pd.notna(median_val):
#             combined[col] = combined[col].fillna(median_val)
#             print(f"Filled {col} missing values with median: {median_val:.1f}")


# # Fill categorical columns
# categorical_cols = [
#     "categories",
#     "final_publisher",
#     "language",
#     "pub_season",
#     "ol_ebook_access"
# ]

# for col in categorical_cols:
#     if col in combined.columns:
#         combined[col] = combined[col].fillna("Unknown")



# # Fill boolean columns
# bool_cols = [
#     "is_series",
#     "ol_has_fulltext",
#     "ol_public_scan"
# ]

# for col in bool_cols:
#     if col in combined.columns:
#         combined[col] = combined[col].fillna(False)


# # Fill description
# if "description" in combined.columns:
#     combined["description"] = combined["description"].fillna("")


# # Drop columns you probably do not need for modeling
# cols_to_drop = [
#     "amazon_url",
#     "ol_internet_archive_ids",
#     "_merge"
# ]


# combined = combined.drop(columns=[c for c in cols_to_drop if c in combined.columns])


# Save processed file
combined.to_csv("data/processed/combined_all_sources_cleaned.csv", index=False)


print("\nFinal combined dataset:")
print(f"Shape: {combined.shape}")
print(f"Columns: {combined.columns.tolist()}")

remaining_missing = combined.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

print("\nRemaining missing values:")
print(remaining_missing)




# # left join keeps all your NYT+Google books
# # and adds Open Library columns where available
# combined = nyt_google.merge(
#     open_lib,
#     on="primary_isbn13",
#     how="left",
#     suffixes=("_google", "_ol")
# )

# print(f"Combined dataset shape: {combined.shape}")
# combined.to_csv("data/raw/combined_all_sources.csv", index=False)

# # count missing values per column as a percentage
# missing = (combined.isnull().sum() / len(combined) * 100).round(1)
# missing = missing[missing > 0].sort_values(ascending=False)

# print("Missing data by column (%):")
# print(missing)

# # for author name — prefer NYT, fall back to Google, fall back to Open Library
# if 'ol_author_name' in combined.columns:
#     combined['final_author'] = (
#         combined['nyt_author']
#         .fillna(combined.get('google_authors'))
#         .fillna(combined.get('ol_author_name'))
#     )

# # for publisher — prefer NYT, fall back to Google, fall back to Open Library
# combined['final_publisher'] = (
#     combined['nyt_publisher']
#     .fillna(combined.get('publisher_google'))
#     .fillna(combined.get('ol_publisher'))
# )

# print(combined[['final_author', 'final_publisher']].isnull().sum())

# numeric_cols = ['page_count', 'average_rating', 'ratings_count', 
#                 'author_total_works', 'ol_edition_count']

# for col in numeric_cols:
#     if col in combined.columns:
#         median_val = combined[col].median()
#         combined[col] = combined[col].fillna(median_val)
#         print(f"Filled {col} missing values with median: {median_val:.1f}")

# categorical_cols = ['categories', 'final_publisher', 'language', 
#                     'pub_season', 'ol_ebook_access']

# for col in categorical_cols:
#     if col in combined.columns:
#         combined[col] = combined[col].fillna('Unknown')


# bool_cols = ['is_series', 'ol_has_fulltext', 'ol_public_scan']

# for col in bool_cols:
#     if col in combined.columns:
#         combined[col] = combined[col].fillna(False)

# combined['description'] = combined['description'].fillna('')

# # count how many key fields are filled in for each book
# key_fields = ['page_count', 'categories', 'average_rating', 
#               'is_series', 'author_total_works', 'description']

# combined['data_completeness_score'] = combined[key_fields].notna().sum(axis=1)
# print(combined['data_completeness_score'].value_counts())


# # drop columns you don't need for modeling
# cols_to_drop = [
#     'amazon_url',           # not a feature
#     'ol_internet_archive_ids',  # not useful for prediction
#     'bestsellers_date',     # redundant with published_date
# ]

# combined = combined.drop(columns=[c for c in cols_to_drop if c in combined.columns])

# # final save
# combined.to_csv("data/raw/combined_all_sources.csv", index=False)

# print(f"\nFinal combined dataset:")
# print(f"Shape: {combined.shape}")
# print(f"Columns: {combined.columns.tolist()}")
# print(f"\nRemaining missing values:")
# print(combined.isnull().sum()[combined.isnull().sum() > 0])