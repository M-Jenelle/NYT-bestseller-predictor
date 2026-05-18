import requests
import os
import json
import csv
from pathlib import Path
from dotenv import load_dotenv
import logging
import time

load_dotenv()

class NYTBooksCollector:
    def __init__(self):
        self.api_key = os.getenv("NYT_API_KEY")
        self.base_url = "https://api.nytimes.com/svc/books/v3"

        # Rate limit safety settings
        self.min_request_interval = 12
        self.last_request_time = 0

        if not self.api_key:
            raise ValueError("Missing NYT_API_KEY. Check your .env file.")
        
        Path("logs").mkdir(exist_ok=True)
        Path("data/raw").mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename="logs/nyt_collector.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def wait_if_needed(self):
        """
        Wait between API calls so we do not hit the rate limit.
        """
        elapsed = time.time() - self.last_request_time

        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        self.last_request_time = time.time()

    def make_request(self, endpoint, params=None):
        if params is None:
            params = {}
        
        params["api-key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        # Wait before making the API request
        self.wait_if_needed()

        response = requests.get(url, params=params)
        if response.status_code == 429:
            print("Rate limit hit. Waiting 60 seconds...")
            time.sleep(60)
            response = requests.get(url, params=params)

        response.raise_for_status()

        return response.json()
    
    def get_overview(self, published_date=None):
        """
        Get all NYT bestseller lists for the current week
        or for a specific published date.
        """
        params = {}

        if published_date:
            params["published_date"] = published_date

        data = self.make_request("lists/overview.json", params)
        return data.get("results", {})
    
    def save_json(self, data, filename):
        file_path = Path("data/raw") / filename

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Saved: {file_path}")

    def save_csv(self, data, filename):
        file_path = Path("data/raw") / filename

        if not data:
            print("No data to save.")
            return

        fieldnames = data[0].keys()

        with open(file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        print(f"Saved CSV: {file_path}")

    def extract_books_from_overview(self, results):
        """
        Turn one week's API response into clean book rows.
        """
        all_books = []

        published_date = results.get("published_date")
        bestsellers_date = results.get("bestsellers_date")

        for book_list in results.get("lists", []):
            list_name = book_list.get("list_name")
            display_name = book_list.get("display_name")
            list_name_encoded = book_list.get("list_name_encoded")

            for book in book_list.get("books", []):
                clean_book = {
                    "published_date": published_date,
                    "bestsellers_date": bestsellers_date,
                    "list_name": list_name,
                    "display_name": display_name,
                    "list_name_encoded": list_name_encoded,
                    "rank": book.get("rank"),
                    "rank_last_week": book.get("rank_last_week"),
                    "weeks_on_list": book.get("weeks_on_list"),
                    "title": book.get("title"),
                    "author": book.get("author"),
                    "publisher": book.get("publisher"),
                    "description": book.get("description"),
                    "primary_isbn10": book.get("primary_isbn10"),
                    "primary_isbn13": book.get("primary_isbn13"),
                    "amazon_url": book.get("amazon_product_url")
                }

                all_books.append(clean_book)

        return all_books
    
    def collect_multiple_weeks(self, number_of_weeks=12):
        """
        Collect NYT bestseller data going backward week by week.
        """
        print(f"Collecting NYT bestseller data for {number_of_weeks} weeks...")
        print("This may take a few minutes because we are waiting between requests.")
        print("-" * 50)

        all_books = []

        # Start with the current week
        results = self.get_overview()

        for week_number in range(number_of_weeks):
            published_date = results.get("published_date")
            print(f"Collecting week {week_number + 1}: {published_date}")

            books_for_week = self.extract_books_from_overview(results)
            all_books.extend(books_for_week)

            logging.info(
                f"Collected {len(books_for_week)} book rows for published date {published_date}"
            )

            if week_number == number_of_weeks - 1:
                break

            # Move backward to the previous week
            previous_date = results.get("previous_published_date")

            if not previous_date:
                print("No previous published date found. Stopping.")
                break

            results = self.get_overview(published_date=previous_date)

        self.save_json(all_books, "nyt_bestsellers_multiple_weeks.json")
        self.save_csv(all_books, "nyt_bestsellers_multiple_weeks.csv")

        print("-" * 50)
        print(f"Done. Collected {len(all_books)} total book rows.")
        print("Saved files in data/raw/")

        return all_books
    
if __name__ == "__main__":
    collector = NYTBooksCollector()
    # Change this number depending on how much data you want
    books = collector.collect_multiple_weeks(number_of_weeks=52)