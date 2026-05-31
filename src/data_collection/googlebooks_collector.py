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
    
    def get_page_count(self, volume):
                """Return a clean page count. Treat 0 or missing as None."""
                if not volume:
                    return None

                info = volume.get("volumeInfo", {})
                page_count = info.get("pageCount")

                try:
                    page_count = int(page_count)
                except (TypeError, ValueError):
                    return None

                if page_count <= 0:
                    return None

                return page_count
    
    def volume_has_isbn(self, volume, isbn13):
        """Check whether a Google Books volume contains the same ISBN13."""
        if not volume or not isbn13:
            return False

        info = volume.get("volumeInfo", {})
        identifiers = info.get("industryIdentifiers", [])

        clean_input_isbn = str(isbn13).replace("-", "").strip()

        for item in identifiers:
            found_isbn = str(item.get("identifier", "")).replace("-", "").strip()

            if found_isbn == clean_input_isbn:
                return True

        return False

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
            "page_count": self.get_page_count(volume),
            # info.get("pageCount")
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

    # testing - def collect_all(self, input_csv="data/raw/nyt_unique_books.csv"):
    def collect_all(self, input_csv="data/raw/nyt_unique_books.csv", test_limit=None):
        """
        Loop through every unique NYT book and enrich it
        with Google Books metadata.
        """
        # testing - df = pd.read_csv(input_csv)

        # print(f"Loaded {len(df)} unique books from NYT data")
        df = pd.read_csv(input_csv)

        # TEST MODE: only run the first few books
        if test_limit is not None:
            df = df.head(test_limit).reset_index(drop=True)
            print(f"TEST MODE: only running first {test_limit} books")

        print(f"Loaded {len(df)} unique books from NYT data")

        #end testing 

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

            # if file_path.exists():
            #     with open(file_path, "r", encoding="utf-8") as f:
            #         existing_data = json.load(f)

            #     enriched_books.append(existing_data)
            #     print(f"[{i + 1}/{len(df)}] Already saved, skipping: {title}")
            #     continue

            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)

                existing_page_count = existing_data.get("page_count")
                existing_match_method = existing_data.get("match_method")

                unreliable_methods = [
                    "title_author_page_count_fallback",
                    "title_author"
                ]

                try:
                    existing_page_count = int(existing_page_count)
                except (TypeError, ValueError):
                    existing_page_count = None

                # If existing saved data has a real page count, use it
                # AND was not from the risky fallback, use it
                if (
                    existing_page_count
                    and existing_page_count > 0
                    and existing_match_method not in unreliable_methods
                ):
                    enriched_books.append(existing_data)
                    print(f"[{i + 1}/{len(df)}] Already saved with page count, skipping: {title}")
                    continue

                # If page count is missing or 0, refetch it
                print(f"[{i + 1}/{len(df)}] Refetching missing/zero/unreliable page count: {title}")

            volume = None
            match_method = None

            # Try ISBN first because it is more accurate
            if isbn13:
                volume = self.search_by_isbn(isbn13)

                if volume:
                    match_method = "isbn"

            # If ISBN fails, fall back to title + author
            # try title + author as a fallback.
            # if not volume and title and author:
            #     volume = self.search_by_title_author(title, author)

            #     if volume:
            #         match_method = "title_author"

            if title and author:
                isbn_page_count = self.get_page_count(volume)

                 # Case 1: ISBN search found nothing, so title + author is okay as fallback
                if not volume:
                    fallback_volume = self.search_by_title_author(title, author)

                    if fallback_volume:
                        volume = fallback_volume
                        match_method = "title_author"

                # Case 2: ISBN search found a book, but page_count is missing
                # Only use title + author fallback if it has the SAME ISBN
                elif isbn_page_count is None:
                    fallback_volume = self.search_by_title_author(title, author)

                    if fallback_volume and self.volume_has_isbn(fallback_volume, isbn13):
                        fallback_page_count = self.get_page_count(fallback_volume)

                        if fallback_page_count is not None:
                            volume = fallback_volume
                            match_method = "title_author_same_isbn_page_count_fallback"
                    else:
                        print(f"Fallback did not match ISBN, keeping ISBN result: {title}")

            
            

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

                print(
                    f"[{i + 1}/{len(df)}] Found using {match_method}: {title} "
                    f"| page_count={combined.get('page_count')}"
                )
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
        #enriched_df.to_csv("data/raw/nyt_google_enriched.csv", index=False)
        enriched_df.to_csv("data/raw/nyt_google_enriched_v2.csv", index=False)

        # -----------------------------
        # Validation checks
        # -----------------------------
        expected_rows = len(df)
        found_rows = len(enriched_books)
        not_found_rows = len(not_found)
        processed_rows = found_rows + not_found_rows

        print("-" * 50)
        print("VALIDATION SUMMARY")
        print(f"Input rows expected: {expected_rows}")
        print(f"Books enriched/found: {found_rows}")
        print(f"Books not found: {not_found_rows}")
        print(f"Total processed: {processed_rows}")

        if processed_rows == expected_rows:
            print("✅ All input rows were processed.")
        else:
            print("❌ WARNING: Some input rows were not processed.")

        if not enriched_df.empty and "page_count" in enriched_df.columns:
            enriched_df["page_count_numeric"] = pd.to_numeric(enriched_df["page_count"], errors="coerce")

            missing_page_count_df = enriched_df[
                enriched_df["page_count_numeric"].isna() |
                (enriched_df["page_count_numeric"] <= 0)
            ]

            page_count_found = len(enriched_df) - len(missing_page_count_df)

            print(f"Books with page count: {page_count_found}/{len(enriched_df)}")
            print(f"Books missing page count: {len(missing_page_count_df)}")

            missing_page_count_df[
                ["primary_isbn13", "nyt_title", "nyt_author", "google_title", "match_method", "page_count"]
            ].to_csv("data/raw/missing_page_counts.csv", index=False)

            print("Missing page counts saved to data/raw/missing_page_counts.csv")

            print("\nMatch method counts:")
            print(enriched_df["match_method"].value_counts(dropna=False))

        not_found_df = pd.DataFrame(not_found)
        not_found_df.to_csv("data/raw/not_found_books.csv", index=False)

        print("-" * 50)
        print(f"Done — {len(enriched_books)} books enriched")
        print(f"Not found: {len(not_found)} books — saved to data/raw/not_found_books.csv")
        # print("Full enriched dataset saved to data/raw/nyt_google_enriched.csv")
        print("Full enriched dataset saved to data/raw/nyt_google_enriched_v2.csv")

        return enriched_df


# if __name__ == "__main__":
#     collector = GoogleBooksCollector()
#     df = collector.collect_all()

#     print(df.head())
#     print(df.columns.tolist())

if __name__ == "__main__":
    collector = GoogleBooksCollector()

    # Set this to True when testing
    TEST_MODE = True

    if TEST_MODE:
        collector.min_request_interval = 1
        df = collector.collect_all(test_limit=20)
    else:
        df = collector.collect_all()


    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 100)

    print(df[["nyt_title", "nyt_author", "google_title", "match_method", "page_count"]])
    print(df.columns.tolist())