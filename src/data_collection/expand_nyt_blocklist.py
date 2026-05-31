# expand_nyt_blocklist.py

import requests
import os
import time
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/expand_nyt_blocklist.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

API_KEY = os.getenv("NYT_API_KEY")
BASE_URL = "https://api.nytimes.com/svc/books/v3"
BLOCKLIST_PATH = Path("data/raw/nyt_blocklist.csv")
CHECKPOINT_PATH = Path("data/raw/nyt_blocklist_checkpoint.csv")


def make_request(endpoint, params=None, retries=5):
    """Make NYT API request with aggressive retry/backoff."""
    if params is None:
        params = {}
    params["api-key"] = API_KEY
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 429:
                wait_time = 60 * (attempt + 1)  # 60, 120, 180, 240, 300
                print(f"Rate limited — waiting {wait_time}s (attempt {attempt + 1}/{retries})")
                logging.warning(f"Rate limited on {endpoint} — waiting {wait_time}s")
                time.sleep(wait_time)
                continue

            if response.status_code == 200:
                return response.json()

            print(f"Status {response.status_code} on {endpoint} — retrying")
            time.sleep(30)

        except requests.RequestException as e:
            wait_time = 30 * (attempt + 1)
            print(f"Request error: {e} — waiting {wait_time}s")
            logging.error(f"Request error on {endpoint}: {e}")
            time.sleep(wait_time)

    logging.error(f"Failed after {retries} retries: {endpoint}")
    return None


def load_checkpoint():
    """Resume from checkpoint if it exists."""
    if CHECKPOINT_PATH.exists():
        try:
            df = pd.read_csv(CHECKPOINT_PATH)
            if df.empty:
                print("Checkpoint file is empty — starting fresh")
                return pd.DataFrame(columns=["primary_isbn13", "nyt_title"]), None
            print(f"Resuming from checkpoint — {len(df)} ISBNs already collected")
            logging.info(f"Resuming from checkpoint — {len(df)} ISBNs")
            return df, df["last_date"].iloc[-1] if "last_date" in df.columns else None
        except pd.errors.EmptyDataError:
            print("Checkpoint file is empty — starting fresh")
            return pd.DataFrame(columns=["primary_isbn13", "nyt_title"]), None
    return pd.DataFrame(columns=["primary_isbn13", "nyt_title"]), None


def build_blocklist(number_of_weeks=260):
    """
    Collect NYT history for blocklist ONLY.
    Never touches nyt_google_enriched.csv or any feature CSV.
    """
    print(f"Collecting {number_of_weeks} weeks of NYT history...")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output: {BLOCKLIST_PATH}")
    print("This will NOT touch any existing feature CSVs.")
    print("-" * 50)

    existing_df, resume_date = load_checkpoint()
    all_records = existing_df[["primary_isbn13", "nyt_title"]].to_dict("records")
    already_collected = len(all_records)

    # start from current week
    data = make_request("lists/overview.json")
    if not data:
        print("Could not reach NYT API. Check your key and connection.")
        return

    results = data.get("results", {})

    # if resuming, fast forward to where we left off
    if resume_date:
        print(f"Fast forwarding to resume date: {resume_date}")
        data = make_request("lists/overview.json", {"published_date": resume_date})
        if data:
            results = data.get("results", {})

    consecutive_failures = 0

    for week_num in range(number_of_weeks):
        published_date = results.get("published_date")
        print(f"[{week_num + 1}/{number_of_weeks}] Week: {published_date}")

        # extract ISBNs from this week
        week_count = 0
        for book_list in results.get("lists", []):
            for book in book_list.get("books", []):
                isbn13 = book.get("primary_isbn13", "")
                title = book.get("title", "")

                if isbn13 and isbn13.lower() != "nan":
                    all_records.append({
                        "primary_isbn13": str(isbn13).strip(),
                        "nyt_title": title,
                        "nyt_author": book.get("author", ""),
                        "nyt_published_date": published_date,
                        "last_date": published_date
                    })
                    week_count += 1

        logging.info(f"Week {week_num + 1} ({published_date}): {week_count} ISBNs")
        consecutive_failures = 0

        # checkpoint every 10 weeks
        if (week_num + 1) % 10 == 0:
            df_so_far = pd.DataFrame(all_records).drop_duplicates(subset=["primary_isbn13"])
            df_so_far.to_csv(CHECKPOINT_PATH, index=False)
            total = len(df_so_far)
            print(f"  Checkpoint saved — {total} unique ISBNs so far")
            logging.info(f"Checkpoint at week {week_num + 1} — {total} ISBNs")

        if week_num == number_of_weeks - 1:
            break

        # move to previous week
        previous_date = results.get("previous_published_date")
        if not previous_date:
            print("No previous date found — stopping early")
            break

        data = make_request(
            "lists/overview.json",
            {"published_date": previous_date}
        )

        if not data:
            consecutive_failures += 1
            print(f"Failed to fetch week — consecutive failures: {consecutive_failures}")
            logging.error(f"Failed to fetch {previous_date}")

            if consecutive_failures >= 5:
                print("5 consecutive failures — saving and stopping")
                print("Re-run to resume from checkpoint")
                break

            # wait longer and try to continue
            time.sleep(120)
            continue

        results = data.get("results", {})

        # NYT allows 5 requests per minute — 12s between requests is safe
        time.sleep(12)

    # final save — blocklist only, no feature data
    blocklist = (
        pd.DataFrame(all_records)
        [["primary_isbn13", "nyt_title", "nyt_author", "nyt_published_date"]]
        .drop_duplicates(subset=["primary_isbn13"])
        .dropna(subset=["primary_isbn13"])
    )

    BLOCKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    blocklist.to_csv(BLOCKLIST_PATH, index=False)

    # clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint cleaned up")

    print("-" * 50)
    print(f"Done — {len(blocklist)} unique ISBNs saved to {BLOCKLIST_PATH}")
    logging.info(f"Blocklist complete — {len(blocklist)} ISBNs")

    return blocklist


if __name__ == "__main__":
    blocklist = build_blocklist(number_of_weeks=260)
    print(blocklist.head())