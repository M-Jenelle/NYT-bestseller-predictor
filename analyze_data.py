"""
Deeper EDA for NYT + Google Books + Open Library feature design.

Inputs:
    data/raw/nyt_google_enriched.csv
    data/raw/open_library_enriched.csv
    data/raw/negative_samples.csv

Outputs:
    data/processed/merged_feature_design.csv
    data/processed/class_comparison_feature_design.csv
    reports/eda_feature_design_summary.md
    reports/eda_outputs/*.csv
    figures/*.png

Run:
    python analyze_data.py

Optional:
    python analyze_data.py \
        --nyt-google data/raw/nyt_google_enriched.csv \
        --open-library data/raw/open_library_enriched.csv \
        --negative-samples data/raw/negative_samples.csv
"""

import argparse
import ast
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/processed/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Configuration
# -----------------------------

FIG_DIR = Path("figures")
REPORT_DIR = Path("reports")
EDA_OUT_DIR = REPORT_DIR / "eda_outputs"
PROCESSED_DIR = Path("data/processed")

for folder in [FIG_DIR, REPORT_DIR, EDA_OUT_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# -----------------------------
# Helper functions
# -----------------------------

def clean_isbn13(value):
    """Standardize ISBN values so Google and Open Library can merge cleanly."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Handles cases where pandas reads large numeric IDs as 978...0.0
    if value.endswith(".0"):
        value = value[:-2]

    # Keep only digits
    value = "".join(ch for ch in value if ch.isdigit())

    if value == "":
        return np.nan

    return value


def safe_parse_list(value):
    """
    Convert stringified list columns into Python lists.

    Examples:
        "['Fiction', 'Fantasy']" -> ['Fiction', 'Fantasy']
        "[]" -> []
        NaN -> []
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if value in ["", "[]", "nan", "None"]:
        return []

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except (ValueError, SyntaxError):
        return [value]


def first_or_unknown(items, unknown="Unknown"):
    """Return first list item, or Unknown if list is empty."""
    if not items:
        return unknown
    first = items[0]

    if isinstance(first, str):
        if first.strip() == "":
            return unknown
    elif pd.isna(first):
        return unknown

    return str(first)


def extract_year(value):
    """
    Extract a year from date-like strings.

    Handles:
        2024-08-27
        2020
        14 april 1981
    """
    if pd.isna(value):
        return np.nan

    value = str(value)
    match = pd.Series([value]).str.extract(r"(\d{4})")[0].iloc[0]

    if pd.isna(match):
        return np.nan

    return int(match)


def description_word_count(value):
    """Simple text-length feature for book descriptions."""
    if pd.isna(value):
        return 0
    return len(str(value).split())


def has_value(value):
    """Boolean feature for whether a field is populated."""
    if pd.isna(value):
        return False
    if str(value).strip() in ["", "[]", "nan", "None"]:
        return False
    return True


def save_barplot(series, title, xlabel, ylabel, filename, figsize=(12, 6)):
    """Save a horizontal bar chart from a value_counts-style series."""
    plot_data = series.dropna()

    if plot_data.empty:
        print(f"Skipping {filename}; no data available.")
        return

    plt.figure(figsize=figsize)
    sns.barplot(x=plot_data.values, y=plot_data.index, color="#639922")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=150)
    plt.close()


def save_markdown_table(df, max_rows=20):
    """Small helper for markdown reports."""
    if df.empty:
        return "_No data available._"
    return df.head(max_rows).to_markdown(index=False)


# -----------------------------
# Load and prepare data
# -----------------------------

def load_data(nyt_google_path, open_library_path):
    """Read both raw CSV files."""
    nyt_google = pd.read_csv(nyt_google_path)
    open_library = pd.read_csv(open_library_path)

    nyt_google["isbn13_clean"] = nyt_google["primary_isbn13"].apply(clean_isbn13)
    open_library["isbn13_clean"] = open_library["primary_isbn13"].apply(clean_isbn13)

    return nyt_google, open_library


def load_negative_samples(negative_samples_path):
    """Read non-bestseller samples collected for classifier EDA."""
    negative_samples = pd.read_csv(negative_samples_path)
    negative_samples["isbn13_clean"] = negative_samples["primary_isbn13"].apply(clean_isbn13)
    return negative_samples


def add_google_features(df):
    """Create analysis/modeling helper columns from the NYT + Google Books file."""
    df = df.copy()

    # List-like Google columns
    df["categories_list"] = df["categories"].apply(safe_parse_list)
    df["primary_genre"] = df["categories_list"].apply(first_or_unknown)
    df["num_google_categories"] = df["categories_list"].apply(len)

    # Dates
    df["google_pub_year"] = df["google_published_date"].apply(extract_year)
    df["google_pub_date_parsed"] = pd.to_datetime(df["google_published_date"], errors="coerce")
    df["google_pub_month"] = df["google_pub_date_parsed"].dt.month
    df["google_pub_decade"] = (df["google_pub_year"] // 10) * 10

    # Text features
    df["description_word_count"] = df["google_description"].apply(description_word_count)
    df["has_google_description"] = df["google_description"].apply(has_value)
    df["has_google_subtitle"] = df["google_subtitle"].apply(has_value)
    df["has_average_rating"] = df["average_rating"].notna()
    df["has_ratings_count"] = df["ratings_count"].notna()

    # Target helper columns
    df["long_run_52_weeks"] = df["nyt_weeks_on_list"] >= 52
    df["long_run_26_weeks"] = df["nyt_weeks_on_list"] >= 26
    df["above_median_weeks"] = df["nyt_weeks_on_list"] >= df["nyt_weeks_on_list"].median()
    df["top_5_rank"] = df["nyt_rank"] <= 5

    return df


def add_open_library_features(df):
    """Create analysis/modeling helper columns from the Open Library enrichment file."""
    df = df.copy()

    df["ol_subjects_list"] = df["ol_subjects"].apply(safe_parse_list)
    df["primary_ol_subject"] = df["ol_subjects_list"].apply(first_or_unknown)
    df["num_ol_subjects"] = df["ol_subjects_list"].apply(len)

    df["ol_languages_list"] = df["ol_languages"].apply(safe_parse_list)
    df["num_ol_languages"] = df["ol_languages_list"].apply(len)
    df["primary_ol_language"] = df["ol_languages_list"].apply(first_or_unknown)

    df["ol_archive_ids_list"] = df["ol_internet_archive_ids"].apply(safe_parse_list)
    df["num_archive_ids"] = df["ol_archive_ids_list"].apply(len)

    df["ol_first_publish_decade"] = (df["ol_first_publish_year"] // 10) * 10
    df["has_author_birth_date"] = df["author_birth_date"].apply(has_value)
    df["has_series_name"] = df["series_name"].apply(has_value)

    # Some boolean columns may arrive as strings depending on export path
    for col in ["ol_has_fulltext", "ol_public_scan", "is_series"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).fillna(False)

    return df


def merge_feature_data(nyt_google, open_library):
    """Merge the two enrichment sources into one feature-design dataset."""
    merged = nyt_google.merge(
        open_library,
        on="isbn13_clean",
        how="left",
        suffixes=("", "_ol"),
        indicator=True,
    )

    merged["has_open_library_match"] = merged["_merge"].eq("both")
    merged = merged.drop(columns=["_merge"])

    # Agreement / comparison features
    merged["page_count_gap_google_minus_ol"] = merged["page_count"] - merged["ol_number_of_pages"]
    merged["has_page_count_google"] = merged["page_count"].notna()
    merged["has_page_count_ol"] = merged["ol_number_of_pages"].notna()
    merged["has_both_page_counts"] = merged["has_page_count_google"] & merged["has_page_count_ol"]

    # Age-style features based on Google publication year and OL first publish year
    merged["years_since_google_pub"] = 2026 - merged["google_pub_year"]
    merged["years_since_first_publish"] = 2026 - merged["ol_first_publish_year"]
    merged["google_vs_ol_year_gap"] = merged["google_pub_year"] - merged["ol_first_publish_year"]

    return merged


def prepare_class_comparison_data(merged, negative_samples):
    """
    Align NYT positives and non-bestseller negatives to shared feature names.

    Positives come from the merged NYT/Google/Open Library feature dataset.
    Negatives come from the Open Library sampler, so this keeps the comparison
    to fields that can be interpreted consistently across both classes.
    """
    positive = pd.DataFrame(
        {
            "isbn13_clean": merged["isbn13_clean"],
            "title": merged["nyt_title"],
            "author": merged["nyt_author"],
            "publisher": merged["google_publisher"].combine_first(merged["nyt_publisher"]),
            "publish_year": merged["google_pub_year"],
            "page_count": merged["page_count"].combine_first(merged["ol_number_of_pages"]),
            "ol_edition_count": merged["ol_edition_count"],
            "ol_subjects": merged["ol_subjects"],
            "ol_ebook_access": merged["ol_ebook_access"],
            "ol_languages": merged["ol_languages"],
            "ol_first_publish_year": merged["ol_first_publish_year"],
            "is_bestseller": 1,
        }
    )

    negative = negative_samples.copy()
    negative["is_bestseller"] = 0

    shared_cols = [
        "isbn13_clean",
        "title",
        "author",
        "publisher",
        "publish_year",
        "page_count",
        "ol_edition_count",
        "ol_subjects",
        "ol_ebook_access",
        "ol_languages",
        "ol_first_publish_year",
        "is_bestseller",
    ]

    for col in shared_cols:
        if col not in negative.columns:
            negative[col] = np.nan

    class_data = pd.concat(
        [positive[shared_cols], negative[shared_cols]],
        ignore_index=True,
    )

    numeric_cols = [
        "publish_year",
        "page_count",
        "ol_edition_count",
        "ol_first_publish_year",
    ]
    for col in numeric_cols:
        class_data[col] = pd.to_numeric(class_data[col], errors="coerce")

    return class_data


# -----------------------------
# EDA tables
# -----------------------------

def make_summary_tables(nyt_google, open_library, merged, class_data):
    """Save CSV tables that are useful for README/report writing."""
    overview = pd.DataFrame(
        [
            {
                "dataset": "nyt_google_enriched",
                "rows": len(nyt_google),
                "columns": nyt_google.shape[1],
                "unique_isbn13": nyt_google["isbn13_clean"].nunique(),
                "duplicate_isbn13_rows": nyt_google["isbn13_clean"].duplicated().sum(),
            },
            {
                "dataset": "open_library_enriched",
                "rows": len(open_library),
                "columns": open_library.shape[1],
                "unique_isbn13": open_library["isbn13_clean"].nunique(),
                "duplicate_isbn13_rows": open_library["isbn13_clean"].duplicated().sum(),
            },
            {
                "dataset": "merged_feature_design",
                "rows": len(merged),
                "columns": merged.shape[1],
                "unique_isbn13": merged["isbn13_clean"].nunique(),
                "duplicate_isbn13_rows": merged["isbn13_clean"].duplicated().sum(),
            },
        ]
    )

    missingness = (
        merged.isna()
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_percent"})
        .sort_values("missing_percent", ascending=False)
    )

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = merged[numeric_cols].describe().T.round(2).reset_index()
    numeric_summary = numeric_summary.rename(columns={"index": "column"})

    categorical_cols = [
        "nyt_list_name",
        "match_method",
        "primary_genre",
        "language",
        "maturity_rating",
        "ol_ebook_access",
        "primary_ol_language",
        "is_series",
        "has_open_library_match",
    ]

    categorical_summaries = []
    for col in categorical_cols:
        if col in merged.columns:
            counts = (
                merged[col]
                .value_counts(dropna=False)
                .head(15)
                .reset_index()
            )
            counts.columns = ["value", "count"]
            counts.insert(0, "column", col)
            categorical_summaries.append(counts)

    categorical_summary = pd.concat(categorical_summaries, ignore_index=True)

    # Correlation to target. This is not causal; it just points to candidate features.
    candidate_numeric_features = [
        "nyt_rank",
        "num_lists_appeared",
        "page_count",
        "average_rating",
        "ratings_count",
        "google_pub_year",
        "google_pub_month",
        "num_google_categories",
        "description_word_count",
        "ol_edition_count",
        "ol_number_of_pages",
        "num_ol_subjects",
        "num_ol_languages",
        "num_archive_ids",
        "author_total_works",
        "ol_first_publish_year",
        "years_since_google_pub",
        "years_since_first_publish",
        "google_vs_ol_year_gap",
        "page_count_gap_google_minus_ol",
    ]

    available_candidate_features = [
        col for col in candidate_numeric_features
        if col in merged.columns and pd.api.types.is_numeric_dtype(merged[col])
    ]

    correlations = (
        merged[available_candidate_features + ["nyt_weeks_on_list"]]
        .corr(numeric_only=True)["nyt_weeks_on_list"]
        .drop("nyt_weeks_on_list")
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .reset_index()
    )
    correlations.columns = ["feature", "correlation_with_nyt_weeks_on_list"]

    feature_candidates = pd.DataFrame(
        [
            {
                "feature": "primary_genre",
                "source": "Google Books",
                "feature_type": "categorical",
                "why_it_may_help": "Genre/category may strongly affect bestseller shelf life and target audience.",
            },
            {
                "feature": "nyt_list_name",
                "source": "NYT",
                "feature_type": "categorical",
                "why_it_may_help": "Different NYT lists behave differently; children's books, fiction, and nonfiction may have different staying power.",
            },
            {
                "feature": "page_count / ol_number_of_pages",
                "source": "Google Books + Open Library",
                "feature_type": "numeric",
                "why_it_may_help": "Length may separate children's books, novels, nonfiction, and reference-style books.",
            },
            {
                "feature": "google_pub_month",
                "source": "Google Books",
                "feature_type": "date/seasonality",
                "why_it_may_help": "Publication timing may matter because some genres are seasonal.",
            },
            {
                "feature": "description_word_count",
                "source": "Google Books",
                "feature_type": "text-derived numeric",
                "why_it_may_help": "Richer metadata could correlate with better match quality or more commercially supported releases.",
            },
            {
                "feature": "average_rating / ratings_count",
                "source": "Google Books",
                "feature_type": "numeric",
                "why_it_may_help": "Reader engagement could help, but missingness is very high so these should be used carefully.",
            },
            {
                "feature": "is_series / series_name",
                "source": "Open Library",
                "feature_type": "categorical/boolean",
                "why_it_may_help": "Series books may have built-in audiences and repeat demand.",
            },
            {
                "feature": "author_total_works",
                "source": "Open Library",
                "feature_type": "numeric",
                "why_it_may_help": "Author productivity/popularity proxy; prolific authors may have more existing audience awareness.",
            },
            {
                "feature": "ol_edition_count",
                "source": "Open Library",
                "feature_type": "numeric",
                "why_it_may_help": "More editions can proxy for popularity, longevity, translations, or reprints.",
            },
            {
                "feature": "ol_ebook_access / ol_has_fulltext",
                "source": "Open Library",
                "feature_type": "categorical/boolean",
                "why_it_may_help": "Availability/access fields may signal older/public/archived books versus newer commercial releases.",
            },
        ]
    )

    class_balance = (
        class_data["is_bestseller"]
        .value_counts()
        .rename(index={1: "positive_bestseller", 0: "negative_non_bestseller"})
        .reset_index()
    )
    class_balance.columns = ["class", "count"]
    class_balance["percent"] = (
        class_balance["count"] / class_balance["count"].sum() * 100
    ).round(2)

    coverage_features = [
        "isbn13_clean",
        "title",
        "author",
        "publisher",
        "publish_year",
        "page_count",
        "ol_edition_count",
        "ol_subjects",
        "ol_ebook_access",
        "ol_languages",
        "ol_first_publish_year",
    ]
    coverage_rows = []
    for feature in coverage_features:
        feature_presence = class_data[feature].apply(has_value)
        for class_value, label in [(1, "positive_bestseller"), (0, "negative_non_bestseller")]:
            class_mask = class_data["is_bestseller"].eq(class_value)
            class_count = int(class_mask.sum())
            populated_count = int(feature_presence[class_mask].sum())
            coverage_rows.append(
                {
                    "feature": feature,
                    "class": label,
                    "populated_count": populated_count,
                    "total_count": class_count,
                    "coverage_percent": (
                        round(populated_count / class_count * 100, 2)
                        if class_count
                        else np.nan
                    ),
                }
            )

    feature_coverage = pd.DataFrame(coverage_rows)

    shared_distribution_features = [
        "page_count",
        "ol_edition_count",
        "publish_year",
    ]
    distribution_summary = (
        class_data
        .assign(
            class_label=lambda df: df["is_bestseller"].map(
                {1: "positive_bestseller", 0: "negative_non_bestseller"}
            )
        )
        .groupby("class_label")[shared_distribution_features]
        .describe()
        .stack(level=0)
        .reset_index()
        .rename(columns={"level_1": "feature"})
    )

    overview.to_csv(EDA_OUT_DIR / "dataset_overview.csv", index=False)
    missingness.to_csv(EDA_OUT_DIR / "missingness_report.csv", index=False)
    numeric_summary.to_csv(EDA_OUT_DIR / "numeric_feature_summary.csv", index=False)
    categorical_summary.to_csv(EDA_OUT_DIR / "categorical_feature_summary.csv", index=False)
    correlations.to_csv(EDA_OUT_DIR / "feature_target_correlations.csv", index=False)
    feature_candidates.to_csv(EDA_OUT_DIR / "modeling_feature_candidates.csv", index=False)
    class_balance.to_csv(EDA_OUT_DIR / "class_balance.csv", index=False)
    feature_coverage.to_csv(EDA_OUT_DIR / "feature_coverage_by_class.csv", index=False)
    distribution_summary.to_csv(
        EDA_OUT_DIR / "shared_feature_distribution_by_class.csv",
        index=False,
    )

    return {
        "overview": overview,
        "missingness": missingness,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "feature_candidates": feature_candidates,
        "class_balance": class_balance,
        "feature_coverage": feature_coverage,
        "distribution_summary": distribution_summary,
    }


# -----------------------------
# Figures
# -----------------------------

def make_figures(nyt_google, open_library, merged, class_data):
    """Create EDA plots focused on feature design."""
    # 1. Missingness
    missing_top = (
        merged.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(20)
    )
    plt.figure(figsize=(11, 7))
    sns.barplot(x=missing_top.values, y=missing_top.index, color="#639922")
    plt.title("Top 20 columns by missingness in merged feature dataset")
    plt.xlabel("Missing values (%)")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_01_missingness_top20.png", dpi=150)
    plt.close()

    # 2. Source overlap
    overlap_counts = pd.Series(
        {
            "NYT + Google rows": len(nyt_google),
            "Open Library rows": len(open_library),
            "Merged rows": len(merged),
            "Rows with Open Library match": int(merged["has_open_library_match"].sum()),
        }
    )
    save_barplot(
        overlap_counts,
        "Dataset size and source overlap",
        "Rows",
        "Dataset / match status",
        "eda_02_source_overlap.png",
        figsize=(10, 5),
    )

    # 3. NYT target distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(merged["nyt_weeks_on_list"].dropna(), bins=50, color="#639922")
    median_weeks = merged["nyt_weeks_on_list"].median()
    plt.axvline(
        median_weeks,
        color="red",
        linestyle="--",
        label=f"Median: {median_weeks:.0f} weeks",
    )
    plt.axvline(
        52,
        color="black",
        linestyle=":",
        label="52-week long-run threshold",
    )
    plt.title("Target distribution: NYT weeks on list")
    plt.xlabel("Weeks on NYT bestseller list")
    plt.ylabel("Number of books")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_03_target_weeks_distribution.png", dpi=150)
    plt.close()

    # 4. Top NYT lists
    top_lists = merged["nyt_list_name"].value_counts().head(15)
    save_barplot(
        top_lists,
        "Top NYT bestseller lists in dataset",
        "Number of records",
        "NYT list name",
        "eda_04_top_nyt_lists.png",
    )

    # 5. Top Google genres
    top_genres = merged["primary_genre"].value_counts().head(15)
    save_barplot(
        top_genres,
        "Top Google Books genres/categories",
        "Number of books",
        "Primary genre",
        "eda_05_top_google_genres.png",
    )

    # 6. Publication month by genre heatmap
    top_6_genres = merged["primary_genre"].value_counts().head(6).index.tolist()
    month_genre = (
        merged[merged["primary_genre"].isin(top_6_genres)]
        .dropna(subset=["google_pub_month"])
        .groupby(["google_pub_month", "primary_genre"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(1, 13), fill_value=0)
    )

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    if not month_genre.empty:
        plt.figure(figsize=(12, 6))
        sns.heatmap(month_genre, annot=True, fmt="d", cmap="Greens")
        plt.title("Publication month by top Google genre")
        plt.xlabel("Primary genre")
        plt.ylabel("Publication month")
        plt.yticks(
            ticks=np.arange(12) + 0.5,
            labels=month_names,
            rotation=0,
        )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_06_pub_month_genre_heatmap.png", dpi=150)
        plt.close()

    # 7. Numeric correlation heatmap
    numeric_features = [
        "nyt_weeks_on_list",
        "nyt_rank",
        "num_lists_appeared",
        "page_count",
        "average_rating",
        "ratings_count",
        "google_pub_year",
        "description_word_count",
        "ol_edition_count",
        "ol_number_of_pages",
        "author_total_works",
        "ol_first_publish_year",
        "num_ol_subjects",
        "num_ol_languages",
    ]
    numeric_features = [col for col in numeric_features if col in merged.columns]

    corr = merged[numeric_features].corr(numeric_only=True)

    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Numeric feature correlation heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_07_numeric_feature_correlations.png", dpi=150)
    plt.close()

    # 8. Google page count vs Open Library page count
    page_compare = merged.dropna(subset=["page_count", "ol_number_of_pages"])
    if not page_compare.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=page_compare,
            x="page_count",
            y="ol_number_of_pages",
            alpha=0.5,
            color="#639922",
        )
        max_pages = np.nanmax(
            [page_compare["page_count"].max(), page_compare["ol_number_of_pages"].max()]
        )
        plt.plot([0, max_pages], [0, max_pages], linestyle="--", color="red")
        plt.title("Page count comparison: Google Books vs Open Library")
        plt.xlabel("Google Books page_count")
        plt.ylabel("Open Library ol_number_of_pages")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_08_page_count_source_comparison.png", dpi=150)
        plt.close()

    # 9. Series vs target
    if "is_series" in merged.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=merged,
            x="is_series",
            y="nyt_weeks_on_list",
            showfliers=False,
        )
        plt.title("NYT weeks on list by Open Library series flag")
        plt.xlabel("Is part of a series?")
        plt.ylabel("NYT weeks on list")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_09_series_vs_weeks_boxplot.png", dpi=150)
        plt.close()

    # 10. Author total works vs target
    if "author_total_works" in merged.columns:
        works_plot = merged.dropna(subset=["author_total_works", "nyt_weeks_on_list"])
        if not works_plot.empty:
            plt.figure(figsize=(9, 6))
            sns.scatterplot(
                data=works_plot,
                x="author_total_works",
                y="nyt_weeks_on_list",
                alpha=0.45,
                color="#639922",
            )
            plt.xscale("log")
            plt.yscale("symlog")
            plt.title("Author total works vs NYT weeks on list")
            plt.xlabel("Author total works (log scale)")
            plt.ylabel("NYT weeks on list (symlog scale)")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "eda_10_author_works_vs_weeks.png", dpi=150)
            plt.close()

    # 11. Long-run rate by genre
    genre_long_run = (
        merged.groupby("primary_genre")["long_run_52_weeks"]
        .agg(["count", "mean"])
        .query("count >= 10")
        .sort_values("mean", ascending=False)
        .head(15)
    )
    if not genre_long_run.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=genre_long_run.reset_index(),
            x="mean",
            y="primary_genre",
            color="#639922",
        )
        plt.title("Share of books with 52+ NYT weeks by genre (genres with at least 10 books)")
        plt.xlabel("Share with 52+ weeks")
        plt.ylabel("Primary genre")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_11_long_run_rate_by_genre.png", dpi=150)
        plt.close()

    # 12. Open Library edition count vs target
    edition_plot = merged.dropna(subset=["ol_edition_count", "nyt_weeks_on_list"])
    if not edition_plot.empty:
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=edition_plot,
            x="ol_edition_count",
            y="nyt_weeks_on_list",
            alpha=0.45,
            color="#639922",
        )
        plt.xscale("log")
        plt.yscale("symlog")
        plt.title("Open Library edition count vs NYT weeks on list")
        plt.xlabel("Open Library edition count (log scale)")
        plt.ylabel("NYT weeks on list (symlog scale)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "eda_12_edition_count_vs_weeks.png", dpi=150)
        plt.close()

    # 13. Class balance for positive vs negative samples
    class_plot = (
        class_data["is_bestseller"]
        .map({1: "Positive: NYT bestseller", 0: "Negative: not NYT bestseller"})
        .value_counts()
        .rename_axis("class_label")
        .reset_index(name="count")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=class_plot,
        x="class_label",
        y="count",
        palette=["#639922", "#4c78a8"],
    )
    plt.title("Class balance: NYT bestsellers vs negative samples")
    plt.xlabel("")
    plt.ylabel("Books")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_13_class_balance.png", dpi=150)
    plt.close()

    # 14. Feature coverage by class
    coverage_features = [
        "title",
        "author",
        "publisher",
        "publish_year",
        "page_count",
        "ol_edition_count",
        "ol_subjects",
        "ol_ebook_access",
        "ol_languages",
        "ol_first_publish_year",
    ]
    coverage_plot_rows = []
    for feature in coverage_features:
        feature_presence = class_data[feature].apply(has_value)
        for class_value, class_label in [
            (1, "Positive: NYT bestseller"),
            (0, "Negative: not NYT bestseller"),
        ]:
            class_mask = class_data["is_bestseller"].eq(class_value)
            coverage_plot_rows.append(
                {
                    "feature": feature,
                    "class_label": class_label,
                    "coverage_percent": feature_presence[class_mask].mean() * 100,
                }
            )

    coverage_plot = pd.DataFrame(coverage_plot_rows)
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=coverage_plot,
        x="coverage_percent",
        y="feature",
        hue="class_label",
        palette=["#639922", "#4c78a8"],
    )
    plt.title("Feature coverage by class")
    plt.xlabel("Rows with populated value (%)")
    plt.ylabel("Feature")
    plt.xlim(0, 100)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_14_feature_coverage_by_class.png", dpi=150)
    plt.close()

    # 15. Shared numeric feature distributions by class
    distribution_data = class_data.assign(
        class_label=class_data["is_bestseller"].map(
            {1: "NYT bestseller", 0: "Not NYT bestseller"}
        )
    )
    shared_features = [
        ("page_count", "Page count"),
        ("ol_edition_count", "Open Library edition count"),
        ("publish_year", "Publication year"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (feature, label) in zip(axes, shared_features):
        plot_subset = distribution_data.dropna(subset=[feature])
        if plot_subset.empty:
            ax.set_title(f"{label}\n(no data)")
            ax.axis("off")
            continue

        sns.boxplot(
            data=plot_subset,
            x="class_label",
            y=feature,
            ax=ax,
            showfliers=False,
            palette=["#4c78a8", "#639922"],
        )
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Shared feature distributions by class", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_15_shared_feature_distributions_by_class.png", dpi=150)
    plt.close(fig)


# -----------------------------
# Markdown report
# -----------------------------

def write_markdown_report(tables, merged, class_data):
    """Create a short EDA report that can be used in the project README/writeup."""
    overview = tables["overview"]
    missingness = tables["missingness"]
    correlations = tables["correlations"]
    feature_candidates = tables["feature_candidates"]
    class_balance = tables["class_balance"]
    feature_coverage = tables["feature_coverage"]
    distribution_summary = tables["distribution_summary"]

    total_rows = len(merged)
    ol_matches = int(merged["has_open_library_match"].sum())
    ol_match_rate = ol_matches / total_rows if total_rows else 0
    median_weeks = merged["nyt_weeks_on_list"].median()
    long_52_rate = merged["long_run_52_weeks"].mean()
    long_26_rate = merged["long_run_26_weeks"].mean()

    top_genres = (
        merged["primary_genre"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_genres.columns = ["primary_genre", "count"]

    top_lists = (
        merged["nyt_list_name"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_lists.columns = ["nyt_list_name", "count"]

    coverage_pivot = (
        feature_coverage
        .pivot(
            index="feature",
            columns="class",
            values="coverage_percent",
        )
        .reset_index()
    )

    positive_count = int(class_data["is_bestseller"].eq(1).sum())
    negative_count = int(class_data["is_bestseller"].eq(0).sum())
    negative_ratio = negative_count / positive_count if positive_count else np.nan

    report = f"""# EDA Feature Design Summary

## Goal

This analysis combines the NYT + Google Books dataset with the Open Library enrichment dataset to understand which fields are useful for feature design before modeling bestseller performance.

The main modeling target explored here is `nyt_weeks_on_list`. The script also creates helper target flags:

- `long_run_26_weeks`: book stayed on the NYT list for at least 26 weeks
- `long_run_52_weeks`: book stayed on the NYT list for at least 52 weeks
- `above_median_weeks`: book stayed on the list at least as long as the median book

## Dataset Overview

{save_markdown_table(overview)}

## Key Data Quality Notes

- Merged dataset rows: **{total_rows:,}**
- Rows with an Open Library match: **{ol_matches:,}** ({ol_match_rate:.1%})
- Median NYT weeks on list: **{median_weeks:.0f}**
- Share with 26+ weeks: **{long_26_rate:.1%}**
- Share with 52+ weeks: **{long_52_rate:.1%}**

## Top Missingness Issues

{save_markdown_table(missingness.head(15))}

Use fields with very high missingness carefully. For example, `average_rating` and `ratings_count` may look useful, but if most rows are missing them, they can create bias or reduce model coverage.

## Top Genres

{save_markdown_table(top_genres)}

## Top NYT Lists

{save_markdown_table(top_lists)}

## Numeric Features Most Correlated with NYT Weeks on List

This is an early screening step, not proof of causation.

{save_markdown_table(correlations.head(15))}

## Candidate Modeling Features

{save_markdown_table(feature_candidates, max_rows=30)}

## Bestseller vs Non-Bestseller Class EDA

The negative samples from `data/raw/negative_samples.csv` are aligned with the NYT positives on shared fields before comparing coverage and distributions.

Class balance:

{save_markdown_table(class_balance)}

- Positive NYT bestseller rows: **{positive_count:,}**
- Negative non-bestseller rows: **{negative_count:,}**
- Negative-to-positive ratio: **{negative_ratio:.2f}:1**

Feature coverage comparison:

{save_markdown_table(coverage_pivot, max_rows=30)}

Shared numeric feature distribution summary:

{save_markdown_table(distribution_summary, max_rows=30)}

## Generated Figures

- `figures/eda_01_missingness_top20.png`
- `figures/eda_02_source_overlap.png`
- `figures/eda_03_target_weeks_distribution.png`
- `figures/eda_04_top_nyt_lists.png`
- `figures/eda_05_top_google_genres.png`
- `figures/eda_06_pub_month_genre_heatmap.png`
- `figures/eda_07_numeric_feature_correlations.png`
- `figures/eda_08_page_count_source_comparison.png`
- `figures/eda_09_series_vs_weeks_boxplot.png`
- `figures/eda_10_author_works_vs_weeks.png`
- `figures/eda_11_long_run_rate_by_genre.png`
- `figures/eda_12_edition_count_vs_weeks.png`
- `figures/eda_13_class_balance.png`
- `figures/eda_14_feature_coverage_by_class.png`
- `figures/eda_15_shared_feature_distributions_by_class.png`

## Next Modeling Direction

A good first modeling target is either:

1. Regression: predict `nyt_weeks_on_list`
2. Classification: predict `long_run_52_weeks` or `long_run_26_weeks`

For a first pass, start with interpretable features like genre, NYT list name, page count, publication month, description length, series flag, author total works, and Open Library edition count. Then compare a simple baseline model against tree-based models.
"""

    report_path = REPORT_DIR / "eda_feature_design_summary.md"
    report_path.write_text(report)
    return report_path


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Run deeper EDA for NYT bestseller feature design.")
    parser.add_argument(
        "--nyt-google",
        default="data/raw/nyt_google_enriched.csv",
        help="Path to NYT + Google Books enriched CSV.",
    )
    parser.add_argument(
        "--open-library",
        default="data/raw/open_library_enriched.csv",
        help="Path to Open Library enriched CSV.",
    )
    parser.add_argument(
        "--negative-samples",
        default="data/raw/negative_samples.csv",
        help="Path to non-bestseller negative sample CSV.",
    )
    args = parser.parse_args()

    nyt_google, open_library = load_data(args.nyt_google, args.open_library)
    negative_samples = load_negative_samples(args.negative_samples)

    nyt_google = add_google_features(nyt_google)
    open_library = add_open_library_features(open_library)

    merged = merge_feature_data(nyt_google, open_library)
    class_data = prepare_class_comparison_data(merged, negative_samples)

    merged_path = PROCESSED_DIR / "merged_feature_design.csv"
    merged.to_csv(merged_path, index=False)
    class_data_path = PROCESSED_DIR / "class_comparison_feature_design.csv"
    class_data.to_csv(class_data_path, index=False)

    tables = make_summary_tables(nyt_google, open_library, merged, class_data)
    make_figures(nyt_google, open_library, merged, class_data)
    report_path = write_markdown_report(tables, merged, class_data)

    print("EDA complete.")
    print(f"Saved merged dataset: {merged_path}")
    print(f"Saved class comparison dataset: {class_data_path}")
    print(f"Saved markdown report: {report_path}")
    print(f"Saved summary tables in: {EDA_OUT_DIR}")
    print(f"Saved figures in: {FIG_DIR}")
    print()
    print("Most useful next files to review:")
    print(f"- {report_path}")
    print(f"- {EDA_OUT_DIR / 'modeling_feature_candidates.csv'}")
    print(f"- {EDA_OUT_DIR / 'feature_target_correlations.csv'}")
    print(f"- {EDA_OUT_DIR / 'class_balance.csv'}")
    print(f"- {EDA_OUT_DIR / 'feature_coverage_by_class.csv'}")
    print(f"- {EDA_OUT_DIR / 'shared_feature_distribution_by_class.csv'}")


if __name__ == "__main__":
    main()
