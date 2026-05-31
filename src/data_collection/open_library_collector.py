import requests
import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv()

class OpenLibraryCollector:
    def __init__(self):
        self.base_url = "https://openlibrary.org"
        self.min_request_interval = 2  # Open Library is generous — 2 sec is safe
        self.last_request_time = 0

        self.headers = {
        "User-Agent": "NYTBookEnrichmentProject (jenellemoore@ucla.edu)"
         }

        Path("logs").mkdir(exist_ok=True)
        Path("data/raw/open_library").mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename="logs/open_library_collector.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def wait_if_needed(self):
        """Wait between requests to respect rate limits"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def make_request(self, url, params=None, retries=3):
        """Make request with error handling"""
        self.wait_if_needed()

        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 429:
                if retries > 0:
                    wait_time = 60
                    logging.warning(f"Rate limit hit — waiting {wait_time} seconds")
                    print(f"Rate limit hit — waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    return self.make_request(url, params, retries - 1)
                else:
                    logging.error(f"Rate limit failed after retries: {url}")
                    print(f"Rate limit failed after retries: {url}")
                    return None
                
            if response.status_code == 404:
                logging.info(f"Not found: {url}")
                return None


            response.raise_for_status()
            logging.info(f"Successfully fetched {url}")
            return response.json()

        except requests.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")
            if retries > 0:
                logging.info(f"Retrying. Retries left: {retries}")
                time.sleep(3)
                return self.make_request(url, params, retries - 1)
            return None

    def search_book_by_isbn(self, isbn13):
        """Look up a book by ISBN13"""
        url = f"{self.base_url}/isbn/{isbn13}.json"
        return self.make_request(url)

    def get_page_count(self, book_data):
        """
        Return the page count from an ISBN edition record.
        Open Library exposes this as number_of_pages on edition JSON
        returned from /isbn/{isbn}.json.
        """
        if not book_data:
            return None

        page_count = book_data.get("number_of_pages")

        try:
            page_count = int(page_count)
        except (TypeError, ValueError):
            return None

        if page_count <= 0:
            return None

        return page_count
    
    def search_metadata_by_isbn(self, isbn13):
        """
        Search Open Library by ISBN.
        This gives extra fields like edition_count, ebook_access,
        has_fulltext, public_scan_b, and language.
        """
        url = f"{self.base_url}/search.json"

        params = {
            "isbn": isbn13,
            "limit": 1
        }

        data = self.make_request(url, params=params)

        if data and data.get("docs"):
            return data["docs"][0]

        return None

    def get_work(self, work_key):
        """
        Get the 'work' record for a book.
        A work is the parent record that holds series info
        and subject tags across all editions.
        """
        url = f"{self.base_url}{work_key}.json"
        return self.make_request(url)

    def get_author(self, author_key):
        """Get full author record including bibliography info"""
        url = f"{self.base_url}{author_key}.json"
        return self.make_request(url)

    def get_author_works(self, author_key):
        """
        Get all works by an author.
        This is how we count their prior bibliography.
        """
        url = f"{self.base_url}/authors/{author_key}/works.json"
        data = self.make_request(url, params={"limit": 100})
        if data:
            return data.get("entries", [])
        return []

    def extract_series_info(self, work_data):
        """
        Check if a book is part of a series.
        Open Library stores this in the work's subject fields.
        """
        if not work_data:
            return False, None

        # check series field directly
        series = work_data.get("series", [])
        if series:
            return True, series[0] if series else None

        # check subjects for series indicators
        subjects = work_data.get("subjects", [])
        series_keywords = ["series", "book 1", "book 2", "volume", "trilogy"]
        for subject in subjects:
            subject_str = str(subject).lower()
            if subject_str.startswith("nyt:"):   # <-- add this guard
                continue
            if any(keyword in subject_str for keyword in series_keywords):
                return True, subject

        return False, None

    def extract_clean_data(self, isbn13, book_data, search_data, work_data, author_data, author_works):
        """Pull only the fields we need"""
        is_series, series_name = self.extract_series_info(work_data)

        # count prior works as bibliography size
        total_works = len(author_works) if author_works else 0

        # author name and birth date
        author_name = None
        author_birth_date = None

        if author_data:
            author_name = author_data.get("name") or author_data.get("personal_name")
            author_birth_date = author_data.get("birth_date")

        # publisher from ISBN edition record
        publishers = []
        if book_data:
            publishers = book_data.get("publishers", [])

        publisher = publishers[0] if publishers else None

        # languages from ISBN edition record
        languages = []
        if book_data:
            raw_languages = book_data.get("languages", [])

            for lang in raw_languages:
                lang_key = lang.get("key", "")
                language_code = lang_key.replace("/languages/", "")
                if language_code:
                    languages.append(language_code)

        # if languages are missing from book_data, try search_data
        if not languages and search_data:
            languages = search_data.get("language", [])

        # ebook availability from search result
        ebook_access = None
        has_fulltext = None
        public_scan = None
        internet_archive_ids = []

        if search_data:
            ebook_access = search_data.get("ebook_access")
            has_fulltext = search_data.get("has_fulltext")
            public_scan = search_data.get("public_scan_b")
            internet_archive_ids = search_data.get("ia", [])

        # edition count from search result
        edition_count = None
        if search_data:
            edition_count = search_data.get("edition_count")

        # subject tags from work
        subjects = []
        if work_data:
            subjects = work_data.get("subjects", [])[:10]

        return {
            "primary_isbn13": isbn13,
            "ol_author_name": author_name,
            "ol_publisher": publisher,
            "ol_languages": languages,
            "ol_ebook_access": ebook_access,
            "ol_has_fulltext": has_fulltext,
            "ol_public_scan": public_scan,
            "ol_internet_archive_ids": internet_archive_ids,
            "ol_edition_count": edition_count,
            "ol_number_of_pages": self.get_page_count(book_data),
            "is_series": is_series,
            "series_name": series_name,
            "author_total_works": total_works,
            "author_birth_date": author_birth_date,
            "ol_subjects": subjects,
            "ol_first_publish_year": (search_data.get("first_publish_year") if search_data else None) or (work_data.get("first_publish_year") or work_data.get("first_publish_date") if work_data else None)
        }

    def save_json(self, data, filename):
        """Save data to JSON"""
        file_path = Path("data/raw/open_library") / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logging.info(f"Saved: {file_path}")

    def collect_all(self, input_csv="data/raw/nyt_google_enriched.csv"):
        """
        Loop through enriched books and add Open Library features
        """
        df = pd.read_csv(input_csv)

        print(f"Loaded {len(df)} books from enriched dataset")
        print(f"Estimated time: {round((len(df) * self.min_request_interval * 5) / 60, 1)} minutes")
        print("Each book can need up to 5 API calls — book, search metadata, work, author, and author works")
        print("-" * 50)

        not_found = []

        # check for already collected books so we can resume if interrupted
        checkpoint_path = Path("data/raw/open_library_enriched.csv")

        if checkpoint_path.exists():
            existing_df = pd.read_csv(checkpoint_path)
            ol_data = existing_df.to_dict("records")
            already_done = existing_df["primary_isbn13"].astype(str).tolist()
            print(f"Skipping {len(already_done)} books already collected")
        else:
            ol_data = []
            already_done = []

        for i, row in df.iterrows():
            isbn13 = str(row.get("primary_isbn13", "")).strip()

            # skip missing ISBNs
            if not isbn13 or isbn13.lower() == "nan":
                print(f"[{i+1}/{len(df)}] Missing ISBN, skipping")
                continue

            if isbn13 in already_done:
                continue

            title = str(row.get("nyt_title", "")).strip()

            # step 1 — get the book record by ISBN
            book_data = self.search_book_by_isbn(isbn13)

            # step 1b — get search metadata by ISBN
            search_data = self.search_metadata_by_isbn(isbn13)

            if not book_data:
                not_found.append({"isbn13": isbn13, "title": title})
                print(f"[{i+1}/{len(df)}] Not found: {title}")
                continue

            # step 2 — get the work record for series info
            work_key = book_data.get("works", [{}])[0].get("key") if book_data.get("works") else None
            work_data = self.get_work(work_key) if work_key else None

            # step 3 — get the author record and their works
            author_key = None
            author_data = None
            author_works = []

            authors = book_data.get("authors", [])
            if not authors and work_data:
                authors = work_data.get("authors", [])
                if authors:
                    author_key = authors[0].get("author", {}).get("key", "").replace("/authors/", "")
            if authors and not author_key:
                author_key = authors[0].get("key", "").replace("/authors/", "")

            if author_key:
                author_data = self.get_author(f"/authors/{author_key}")
                author_works = self.get_author_works(author_key)

            # extract and save clean data
            clean = self.extract_clean_data(
                isbn13, book_data, search_data, work_data, author_data, author_works
                )

            ol_data.append(clean)
            self.save_json(clean, f"ol_{isbn13}.json")
            google_pages = row.get("page_count", "N/A")

            print(
                f"[{i+1}/{len(df)}] Done: {title} | "
                f"Series: {clean['is_series']} | "
                f"Works: {clean['author_total_works']} | "
                f"OL Pages: {clean['ol_number_of_pages']} | "
                f"Google Pages: {google_pages}"
            )

            logging.info(f"Collected: {title}")

            # save checkpoint every 50 total collected books
            if len(ol_data) % 50 == 0:
                pd.DataFrame(ol_data).to_csv(checkpoint_path, index=False)
                print(f"Checkpoint saved — {len(ol_data)} books collected so far")

        # final save
        ol_df = pd.DataFrame(ol_data)
        ol_df.to_csv(checkpoint_path, index=False)

        print("-" * 50)
        print(f"Done — {len(ol_data)} books enriched with Open Library data")
        print(f"Not found: {len(not_found)} books")
        print(f"Saved to {checkpoint_path}")

        return ol_df


if __name__ == "__main__":
    collector = OpenLibraryCollector()
    df = collector.collect_all()
    print(df.head())
    print(df.columns.tolist())
