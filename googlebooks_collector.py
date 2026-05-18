import requests
import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv()


class GoogleBooksCollector:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
        self.base_url = "https://www.googleapis.com/books/v1"

        # Safe delay between Google Books API requests
        # change from 1 to 10 for more time in between to get last 30 books 
        self.min_request_interval = 10
        self.last_request_time = 0

        if not self.api_key:
            raise ValueError("Missing GOOGLE_BOOKS_API_KEY. Check your .env file.")

        Path("logs").mkdir(exist_ok=True)
        Path("data/raw/google_books").mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename="logs/google_books_collector.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def wait_if_needed(self):
        """Wait between requests to avoid hitting rate limits."""
        elapsed = time.time() - self.last_request_time

        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        self.last_request_time = time.time()

    def make_request(self, endpoint, params=None):
        """Make API request with simple error handling."""
        if params is None:
            params = {}

        params["key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        max_retries = 5

        for attempt in range(max_retries):
            self.wait_if_needed()

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                print(f"Rate limit hit. Waiting {wait_time} seconds before retrying...")
                logging.warning(f"Rate limit hit. Waiting {wait_time} seconds.")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()
        raise Exception("Too many rate limit errors. Try again later.")

    def search_by_isbn(self, isbn13):
        """Look up a book by ISBN13. This is the most reliable method."""
        data = self.make_request(
            "volumes",
            {
                "q": f"isbn:{isbn13}",
                "maxResults": 1
            }
        )

        items = data.get("items", [])

        if items:
            return items[0]

        return None

    def search_by_title_author(self, title, author):
        """Fallback search when ISBN returns nothing."""
        query = f'intitle:{title} inauthor:{author}'

        data = self.make_request(
            "volumes",
            {
                "q": query,
                "maxResults": 1
            }
        )

        items = data.get("items", [])

        if items:
            return items[0]

        return None

    def extract_clean_data(self, volume):
        """Pull only the fields we need from the Google Books response."""
        if not volume:
            return {}

        info = volume.get("volumeInfo", {})

        return {
            "google_books_id": volume.get("id"),
            "google_title": info.get("title"),
            "google_subtitle": info.get("subtitle"),
            "google_authors": info.get("authors", []),
            "google_publisher": info.get("publisher"),
            "google_published_date": info.get("publishedDate"),
            "google_description": info.get("description"),
            "page_count": info.get("pageCount"),
            "categories": info.get("categories", []),
            "average_rating": info.get("averageRating"),
            "ratings_count": info.get("ratingsCount"),
            "language": info.get("language"),
            "maturity_rating": info.get("maturityRating"),
            "preview_link": info.get("previewLink"),
            "info_link": info.get("infoLink")
        }

    def make_safe_filename(self, isbn13, title):
        """Create a safe JSON filename."""
        if isbn13 and isbn13 != "nan":
            return f"book_{isbn13}.json"

        safe_title = (
            title.lower()
            .replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "")
            .replace("*", "")
            .replace("?", "")
            .replace('"', "")
            .replace("<", "")
            .replace(">", "")
            .replace("|", "")
        )

        return f"book_{safe_title[:50]}.json"

    def save_json(self, data, filename):
        """Save one enriched book record to JSON."""
        file_path = Path("data/raw/google_books") / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logging.info(f"Saved: {file_path}")

    def collect_all(self, input_csv="data/raw/nyt_unique_books.csv"):
        """
        Loop through every unique NYT book and enrich it
        with Google Books metadata.
        """
        df = pd.read_csv(input_csv)

        print(f"Loaded {len(df)} unique books from NYT data")
        print(f"Estimated time: {round((len(df) * self.min_request_interval) / 60, 1)} minutes")
        print("-" * 50)

        enriched_books = []
        not_found = []

        for i, row in df.iterrows():
            isbn13 = str(row.get("primary_isbn13", "")).strip()
            title = str(row.get("title", "")).strip()
            author = str(row.get("author", "")).strip()

            # Clean up possible missing values
            if isbn13 == "nan":
                isbn13 = ""

            if title == "nan":
                title = ""

            if author == "nan":
                author = ""

            filename = self.make_safe_filename(isbn13, title)
            file_path = Path("data/raw/google_books") / filename

            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

                enriched_books.append(existing_data)
                print(f"[{i + 1}/{len(df)}] Already saved, skipping: {title}")
                continue

            volume = None
            match_method = None

            # Try ISBN first because it is more accurate
            if isbn13:
                volume = self.search_by_isbn(isbn13)

                if volume:
                    match_method = "isbn"

            # If ISBN fails, fall back to title + author
            if not volume and title and author:
                volume = self.search_by_title_author(title, author)

                if volume:
                    match_method = "title_author"

            google_data = self.extract_clean_data(volume)

            if google_data:
                combined = {
                    "primary_isbn13": isbn13,
                    "nyt_title": title,
                    "nyt_author": author,
                    "nyt_publisher": row.get("publisher"),
                    "nyt_weeks_on_list": row.get("weeks_on_list"),
                    "nyt_rank": row.get("rank"),
                    "nyt_list_name": row.get("list_name"),
                    "num_lists_appeared": row.get("num_lists_appeared", None),
                    "match_method": match_method,
                    **google_data
                }

                enriched_books.append(combined)

                filename = self.make_safe_filename(isbn13, title)
                self.save_json(combined, filename)

                print(f"[{i + 1}/{len(df)}] Found using {match_method}: {title}")
                logging.info(f"Found using {match_method}: {title}")

            else:
                missing_book = {
                    "title": title,
                    "author": author,
                    "isbn13": isbn13
                }

                not_found.append(missing_book)

                print(f"[{i + 1}/{len(df)}] Not found: {title}")
                logging.warning(f"Not found: {title}")

        enriched_df = pd.DataFrame(enriched_books)
        enriched_df.to_csv("data/raw/nyt_google_enriched.csv", index=False)

        not_found_df = pd.DataFrame(not_found)
        not_found_df.to_csv("data/raw/not_found_books.csv", index=False)

        print("-" * 50)
        print(f"Done — {len(enriched_books)} books enriched")
        print(f"Not found: {len(not_found)} books — saved to data/raw/not_found_books.csv")
        print("Full enriched dataset saved to data/raw/nyt_google_enriched.csv")

        return enriched_df


if __name__ == "__main__":
    collector = GoogleBooksCollector()
    df = collector.collect_all()

    print(df.head())
    print(df.columns.tolist())