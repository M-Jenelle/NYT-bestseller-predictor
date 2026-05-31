# negative_sampler.py

import re
import unicodedata
import requests
import pandas as pd
import time
import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/negative_sampler.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

CHECKPOINT_PATH = Path("data/raw/negative_samples_checkpoint.csv")


def clean_isbn(value):
    """
    Normalize ISBN values so comparisons are more reliable.
    Keeps only digits. Returns None unless it is ISBN-13 length.
    """
    if pd.isna(value):
        return None

    cleaned = re.sub(r"[^0-9]", "", str(value))

    if len(cleaned) == 13:
        return cleaned

    return None


def normalize_text(value):
    """
    Normalize text for title/author matching.
    Example: 'Fourth Wing!' -> 'fourth wing'
    """
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

BAD_TITLE_KEYWORDS = [
    "fourth wing",
    "iron flame",
    "onyx storm",
    "empyrean",
    "the goldfinch",
    "catching fire",
    "mockingjay",
    "hunger games",
    "diary of a wimpy kid",
    "people we meet on vacation",
    "the girl on the train",
    "gone girl",
    "the da vinci code",
    "angels and demons",
    "angels demons",
    "inferno",
    "the lost symbol",
    "kingdom of ash",
    "harry potter",
    "the bfg",
    "beloved",
    "watchmen",
    "the drawing of the three",
    "the 48 laws of power",
    "who moved my cheese",
    "a clash of kings",
    "the perks of being a wallflower",
    "fifty shades of grey",
    "eragon",
    "the secret",
    "life of pi",
    "breaking dawn",
    "coraline",
    "the art of seduction",
    "american gods",
    "the god delusion",
    "the kite runner",
    "wolves of the calla",
    "rich dad poor dad",
    "blink",
    "brisingr",
    "my sister s keeper",
    "new moon",
    "eclipse",
    "the road",
    "mindset",
    "the 4 hour workweek",
    "the last olympian",
    "the help",
    "the fault in our stars",
    "divergent",
    "insurgent",
    "allegiant",
    "unbroken",
    "the martian",
    "the son of neptune",
    "the mark of athena",
    "queen of shadows",
    "girl wash your face",
    "atomic habits",
    "it ends with us",
    "ugly love",
    "november 9",
    "verity",
    "26 beauties",
    "story of my life",
    "twilight",
    "message in a bottle",
    "summer island",
    "the teacher",
    "the coworker",
    "the boyfriend",
    "freida mcfadden",
    "stephenie meyer",
    "nicholas sparks",
    "kristin hannah",
    "the scorpio races",
    "below zero",
    "small great things",
    "simon vs the homo sapiens agenda",
    "remain",
    "house of flame and shadow",
    "rowley jefferson",
    "tiger tiger",
    "president s shadow",
    "apt pupil",
    "double trouble puzzle",
]

BAD_AUTHOR_KEYWORDS = [
    "freida mcfadden",
    "stephenie meyer",
    "nicholas sparks",
    "kristin hannah",
    "c j box",
    "jodi picoult",
    "becky albertalli",
    "sarah j maas",
    "jeff kinney",
    "james patterson",
    "stephen king",
]

def is_bad_title(title):
    """
    Manually remove obvious NYT bestseller leaks or bestseller series.
    This catches boxed sets, reprints, and variant titles.
    """
    title_clean = normalize_text(title)
    return any(bad in title_clean for bad in BAD_TITLE_KEYWORDS)

def is_bad_author(author):
    author_clean = normalize_text(author)
    return any(bad in author_clean for bad in BAD_AUTHOR_KEYWORDS)

class NegativeSampler:
    def __init__(self, blocklist_csv="data/raw/nyt_blocklist.csv"):
        self.base_url = "https://openlibrary.org/search.json"

        nyt = pd.read_csv(blocklist_csv)

        # 1. ISBN blocklist
        self.known_isbns = set()

        if "primary_isbn13" not in nyt.columns:
            raise ValueError("Blocklist must include a primary_isbn13 column")

        for isbn in nyt["primary_isbn13"].dropna():
            cleaned = clean_isbn(isbn)
            if cleaned:
                self.known_isbns.add(cleaned)

        # Optional: if your NYT/blocklist file has more ISBN columns,
        # include them too.
        possible_isbn_cols = [
            "isbn13",
            "isbn_13",
            "primary_isbn13",
            "primary_isbn10"
        ]

        for col in possible_isbn_cols:
            if col in nyt.columns:
                for isbn in nyt[col].dropna():
                    cleaned = clean_isbn(isbn)
                    if cleaned:
                        self.known_isbns.add(cleaned)

        # 2. Title-author blocklist
        # This catches different editions with different ISBNs.
        self.known_title_authors = set()
        self.known_nyt_authors = set()

        title_col = None
        author_col = None


        for candidate in ["title", "book_title", "name"]:
            if candidate in nyt.columns:
                title_col = candidate
                break

        for candidate in ["author", "book_author", "contributor"]:
            if candidate in nyt.columns:
                author_col = candidate
                break
        # 3. Auto NYT author blocklist
        # Any author appearing in the NYT blocklist is risky as a negative example.
        if author_col:
            for author in nyt[author_col].dropna():
                author_clean = normalize_text(author)
                if author_clean:
                    self.known_nyt_authors.add(author_clean)

        print(f"NYT author blocklist loaded: {len(self.known_nyt_authors)} authors")

        if title_col and author_col:
            for _, row in nyt[[title_col, author_col]].dropna().iterrows():
                title_clean = normalize_text(row[title_col])
                author_clean = normalize_text(row[author_col])

                if title_clean and author_clean:
                    self.known_title_authors.add((title_clean, author_clean))

            print(
                f"Title-author blocklist loaded: "
                f"{len(self.known_title_authors)} known NYT title-author pairs"
            )
        else:
            print(
                "Warning: Could not build title-author blocklist because "
                "title and/or author columns were not found in nyt_blocklist.csv"
            )

        print(f"ISBN blocklist loaded: {len(self.known_isbns)} known NYT ISBNs")

    def search_page(self, year, page=1, limit=100):
        """Pull one page of English books from a given year"""
        
        params = {
            "q": f"first_publish_year:{year} language:eng",
            "limit": limit,
            "page": page,
            "sort": "random",
            "fields": "isbn,title,author_name,first_publish_year,publish_year,number_of_pages_median,publisher,subject,edition_count,ebook_access,language"
        }

        try:
            response = requests.get(
                self.base_url,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            return response.json().get("docs", [])

        except Exception as e:
            print(f"Error on page {page} year {year}: {e}")
            logging.error(f"Error on page {page} year {year}: {e}")
            return []

    def load_checkpoint(self):
        """Resume from checkpoint if it exists."""
        if CHECKPOINT_PATH.exists():
            try:
                df = pd.read_csv(CHECKPOINT_PATH, dtype={"primary_isbn13": str}) 

                if df.empty:
                    return [], set()

                records = df.to_dict("records")
                seen = set(
                    df["primary_isbn13"]
                    .dropna()
                    .astype(str)
                    .map(clean_isbn)
                    .dropna()
                    .tolist()
                )

                print(
                    f"Resuming from checkpoint — "
                    f"{len(records)} negatives already collected"
                )
                return records, seen

            except pd.errors.EmptyDataError:
                return [], set()

        return [], set()

    def is_known_nyt_book(self, doc):
        """
        Return True if this Open Library doc appears to match a known
        NYT bestseller from the blocklist.This is stricter, but helps prevent famous-author leakage.
        """

        # Check every ISBN-13 in the Open Library result, not just the first one.
        open_library_isbns = set()

        for isbn in doc.get("isbn", []):
            cleaned = clean_isbn(isbn)
            if cleaned:
                open_library_isbns.add(cleaned)

        if open_library_isbns & self.known_isbns:
            return True

        # Check title-author match.
        title_clean = normalize_text(doc.get("title"))

        author_list = doc.get("author_name", [])
        author_clean = normalize_text(author_list[0]) if author_list else ""

        if title_clean and author_clean:
            if (title_clean, author_clean) in self.known_title_authors:
                return True

        return False
    
    def is_known_nyt_author(self, doc):
        """
        Return True if the Open Library book's author appears in the NYT blocklist.
        This helps prevent famous-author leakage.
        """
        author_list = doc.get("author_name", [])

        if not author_list:
            return False

        author_clean = normalize_text(author_list[0])

        return author_clean in self.known_nyt_authors

    def get_primary_isbn13(self, doc):
        """
        Pick the first usable ISBN-13 from Open Library.
        """
        for isbn in doc.get("isbn", []):
            cleaned = clean_isbn(isbn)
            if cleaned:
                return cleaned

        return None

    def collect(self, years: dict, total_target=5500):
        """
        Collect negative examples with per-year targets.
        years = dict of {year: target_count}
        """
        negatives, seen_isbns = self.load_checkpoint()

        already_by_year = {}

        for r in negatives:
            y = r.get("publish_year")
            already_by_year[y] = already_by_year.get(y, 0) + 1

        for year, year_target in years.items():
            already_done = already_by_year.get(year, 0)
            remaining = year_target - already_done

            if remaining <= 0:
                print(f"\n{year} already complete ({already_done}/{year_target})")
                continue

            print(
                f"\nCollecting {remaining} more books from {year} "
                f"(have {already_done}/{year_target})..."
            )

            page = 1
            year_count = already_done

            while year_count < year_target:
                docs = self.search_page(year, page=page)

                if not docs:
                    print(f"  No more results for {year} at page {page}")
                    logging.warning(f"No more results for {year} at page {page}")
                    break

                for doc in docs:
                    first_publish_year = doc.get("first_publish_year")

                    if first_publish_year != year:
                        continue

                    isbn13 = self.get_primary_isbn13(doc)
                    
                    if not isbn13:
                        continue

                    if isbn13 in seen_isbns:
                        continue

                    if self.is_known_nyt_book(doc):
                        logging.info(
                            f"Skipped known NYT book match: "
                            f"{doc.get('title')} | {doc.get('author_name')}"
                        )
                        continue

                    if self.is_known_nyt_author(doc):
                        logging.info(
                            f"Skipped known NYT author: "
                            f"{doc.get('title')} | {doc.get('author_name')}"
                        )
                        continue

                    if is_bad_title(doc.get("title")):
                        logging.info(
                            f"Skipped manual bestseller leak: "
                            f"{doc.get('title')} | {doc.get('author_name')}"
                        )
                        continue

                    author_list = doc.get("author_name", [])
                    publisher_list = doc.get("publisher", [])

                    author_name = author_list[0] if author_list else ""

                    if is_bad_author(author_name):
                        logging.info(
                            f"Skipped manual high-risk author: "
                            f"{doc.get('title')} | {author_name}"
                        )
                        continue

                    seen_isbns.add(isbn13)

                    negatives.append({
                        "primary_isbn13": isbn13,
                        "title": doc.get("title"),
                        "author": author_list[0] if author_list else None,
                        "publish_year": first_publish_year,
                        "page_count": doc.get("number_of_pages_median"),
                        "publisher": publisher_list[0] if publisher_list else None,
                        "ol_edition_count": doc.get("edition_count"),
                        "ol_subjects": str(doc.get("subject", [])[:10]),
                        "ol_ebook_access": doc.get("ebook_access"),
                        "ol_languages": str(doc.get("language", [])),
                        "ol_first_publish_year": doc.get("first_publish_year"),
                        "is_bestseller": 0
                    })

                    year_count += 1

                    if year_count >= year_target:
                        break

                print(f"  Page {page} — {year_count}/{year_target} for {year}")
                logging.info(
                    f"Year {year} page {page}: {year_count}/{year_target}"
                )

                
                df = pd.DataFrame(negatives)
                df["primary_isbn13"] = df["primary_isbn13"].astype(str)
                df.to_csv(CHECKPOINT_PATH, index=False)

                page += 1
                time.sleep(2)

        df = pd.DataFrame(negatives)
        df["primary_isbn13"] = df["primary_isbn13"].astype(str)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/raw/negative_samples.csv", index=False)

        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()

        print(f"\nDone — {len(df)} negative examples saved")
        print(df["publish_year"].value_counts().sort_index())
        logging.info(f"Negative sampling complete — {len(df)} examples")
        return df


if __name__ == "__main__":
    sampler = NegativeSampler()

    df = sampler.collect(
        years={
            1980: 3,
            1987: 6,
            1998: 3,
            1999: 3,
            2000: 6,
            2001: 9,
            2003: 6,
            2004: 9,
            2006: 6,
            2007: 9,
            2008: 12,
            2009: 3,
            2010: 15,
            2011: 25,
            2012: 9,
            2013: 15,
            2014: 25,
            2015: 52,
            2016: 52,
            2017: 49,
            2018: 67,
            2019: 70,
            2020: 85,
            2021: 86,
            2022: 95,
            2023: 122,
            2024: 323,
            2025: 3015,
            2026: 1320
        },
        total_target=5500
    )

    print(df.head())