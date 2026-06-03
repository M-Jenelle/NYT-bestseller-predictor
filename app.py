
import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st


PREDICTION_LOG_PATH = Path("reports/app_predictions_log.csv")

API_URL = st.secrets.get(
    "API_URL",
    os.getenv("API_URL", "http://127.0.0.1:8000"),
).rstrip("/")


st.set_page_config(
    page_title="NYT Bestseller Predictor",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 8px;
        padding: 1.25rem 1.4rem 1.4rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_comma_list(value):
    """Convert comma-separated input text into a clean list of strings."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_optional_int(value):
    """Parse optional whole-number text input for API payloads."""
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return int(value)


def parse_form_year(value, field_name):
    """Parse optional year text and show a friendly Streamlit error."""
    try:
        year = parse_optional_int(value)
    except ValueError:
        st.error(f"{field_name} must be a whole year, such as 2023.")
        st.stop()

    if year is not None and not 1400 <= year <= 2100:
        st.error(f"{field_name} must be between 1400 and 2100.")
        st.stop()
    return year


def parse_form_nonnegative_int(value, field_name):
    """Parse optional nonnegative whole-number text and show a friendly error."""
    try:
        number = parse_optional_int(value)
    except ValueError:
        st.error(f"{field_name} must be a whole number.")
        st.stop()

    if number is not None and number < 0:
        st.error(f"{field_name} cannot be negative.")
        st.stop()
    return number


def call_predict_api(payload):
    """Send one book payload to the FastAPI prediction endpoint."""
    response = requests.post(
        f"{API_URL}/v1/predict",
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def call_model_info_api():
    """Fetch model metadata for display."""
    response = requests.get(f"{API_URL}/v1/model-info", timeout=10)
    response.raise_for_status()
    return response.json()


def save_prediction_log(payload, result):
    """Append a successful prediction submission to a local CSV file."""
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": payload.get("title") or "",
        "author": payload.get("author") or "",
        "publisher": payload.get("publisher") or "",
        "original_publication_year": payload.get("ol_first_publish_year") or "",
        "edition_publication_year": payload.get("publish_year") or "",
        "page_count": payload.get("page_count") or "",
        "number_of_editions": payload.get("ol_edition_count") or "",
        "genres_or_subjects": "; ".join(payload.get("ol_subjects") or []),
        "language": "; ".join(payload.get("ol_languages") or []),
        "digital_availability": payload.get("ol_ebook_access") or "",
        "bestseller_probability": result.get("bestseller_probability"),
        "prediction": result.get("prediction"),
        "label": result.get("label"),
        "model_name": result.get("model_name"),
        "api_url": API_URL,
    }

    file_exists = PREDICTION_LOG_PATH.exists()
    with PREDICTION_LOG_PATH.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_recent_prediction_logs(limit=10):
    """Load recent prediction submissions for display in the sidebar."""
    if not PREDICTION_LOG_PATH.exists():
        return []

    with PREDICTION_LOG_PATH.open("r", newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    return rows[-limit:]


EBOOK_ACCESS_OPTIONS = {
    "No digital copy available": "no_ebook",
    "Borrowable digital copy": "borrowable",
    "Accessible digital copy": "printdisabled",
    "Public digital copy": "public",
    "Unknown": "unknown",
}


st.title("NYT Bestseller Predictor")
st.caption("Estimate the likelihood that a book could become a New York Times bestseller.")

with st.expander("How to use the predictor", expanded=True):
    st.write(
        "Enter as much book information as you know then submit the form to estimate "
        "bestseller likelihood. Blank fields are allowed, but fuller information usually "
        "gives the model a stronger signal. The result is a probability estimate, not a "
        "guarantee of future bestseller status."
    )

with st.sidebar:
    st.header("API")
    st.write(f"`{API_URL}`")
    if API_URL.startswith("http://127.0.0.1") or API_URL.startswith("http://localhost"):
        st.warning(
            "Using a local API URL. For Streamlit Cloud, set API_URL to the deployed "
            "Cloud Run API URL in app secrets."
        )
    if st.button("Check API health"):
        try:
            health = requests.get(f"{API_URL}/health", timeout=10).json()
            st.success(f"API status: {health.get('status', 'unknown')}")
        except requests.RequestException as error:
            st.error(f"Could not reach API: {error}")

    try:
        model_info = call_model_info_api()
        st.divider()
        st.metric("Model", model_info.get("model_name", "unknown"))
        st.metric("Threshold", model_info.get("threshold", 0.5))
    except requests.RequestException:
        st.divider()
        st.warning("Start the FastAPI server to load model details.")

    st.divider()
    st.subheader("Submission Log")
    recent_logs = load_recent_prediction_logs()
    if recent_logs:
        st.caption(f"Saved locally to `{PREDICTION_LOG_PATH}`")
        st.dataframe(
            recent_logs,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download CSV",
            data=PREDICTION_LOG_PATH.read_text(encoding="utf-8"),
            file_name="app_predictions_log.csv",
            mime="text/csv",
        )
    else:
        st.caption("No successful predictions logged yet.")


left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Book Information")

    with st.form("prediction_form"):
        title = st.text_input("Title", placeholder="e.g., Fourth Wing")
        author = st.text_input("Author", placeholder="e.g., Rebecca Yarros")
        publisher = st.text_input("Publisher", placeholder="e.g., Red Tower Books")

        first_year_col, edition_year_col = st.columns(2)
        with first_year_col:
            ol_first_publish_year_text = st.text_input(
                "Original publication year",
                placeholder="e.g., 2023",
                help=(
                    "The year the book was first published. For reprints or new editions, "
                    "this may be earlier than the edition publication year."
                ),
            )
        with edition_year_col:
            publish_year_text = st.text_input(
                "Edition publication year",
                placeholder="e.g., 2023",
                help=(
                    "The publication year for the version being evaluated. "
                    "For a brand-new book, this is usually the same as the original publication year."
                ),
            )

        page_col, edition_col = st.columns(2)
        with page_col:
            page_count_text = st.text_input(
                "Page count",
                placeholder="e.g., 528",
                help="Approximate number of pages. Leave blank if you do not know.",
            )
        with edition_col:
            ol_edition_count = st.number_input(
                "Number of editions",
                min_value=0,
                max_value=10000,
                value=1,
                step=1,
                help=(
                    "Approximate number of editions, formats, or versions. "
                    "Use 1 if you do not know."
                ),
            )

        subjects_text = st.text_area(
            "Genres or subjects",
            placeholder="e.g., Fantasy, Romance, Dragons",
            help="Comma-separated genres or subject tags.",
        )
        languages_text = st.text_input(
            "Language",
            value="eng",
            help="Use a language code if available, such as eng for English.",
        )
        digital_availability_label = st.selectbox(
            "Digital availability",
            options=list(EBOOK_ACCESS_OPTIONS.keys()),
            index=0,
            help="Whether a digital version appears to be available.",
        )

        submitted = st.form_submit_button("Predict Bestseller Likelihood", type="primary")

with right:
    st.subheader("Prediction")

    if submitted:
        publish_year = parse_form_year(publish_year_text, "Edition publication year")
        page_count = parse_form_nonnegative_int(page_count_text, "Page count")
        ol_first_publish_year = parse_form_year(
            ol_first_publish_year_text,
            "Original publication year",
        )

        payload = {
            "title": title or None,
            "author": author or None,
            "publisher": publisher or None,
            "publish_year": publish_year,
            "page_count": page_count,
            "ol_edition_count": int(ol_edition_count),
            "ol_subjects": parse_comma_list(subjects_text),
            "ol_ebook_access": EBOOK_ACCESS_OPTIONS[digital_availability_label],
            "ol_languages": parse_comma_list(languages_text),
            "ol_first_publish_year": ol_first_publish_year,
        }

        try:
            result = call_predict_api(payload)
        except requests.ConnectionError:
            st.error(f"Could not connect to the prediction API at `{API_URL}`.")
            if API_URL.startswith("http://127.0.0.1") or API_URL.startswith("http://localhost"):
                st.code(
                    'API_URL = "https://nyt-bestseller-api-701318602800.us-west1.run.app"',
                    language="toml",
                )
                st.write(
                    "If this app is running on Streamlit Cloud, add the value above "
                    "to the app secrets and reboot the app."
                )
            else:
                st.write("Check that the deployed API is live and that the API_URL secret is correct.")
        except requests.HTTPError as error:
            detail = error.response.text if error.response is not None else str(error)
            st.error("The API returned an error.")
            st.code(detail)
        except requests.RequestException as error:
            st.error(f"Prediction request failed: {error}")
        else:
            save_prediction_log(payload, result)

            probability = result["bestseller_probability"]
            label = result["label"].replace("_", " ").title()

            st.metric("Bestseller Probability", f"{probability:.1%}")
            st.progress(probability)

            if result["prediction"] == 1:
                st.success(label)
            else:
                st.info(label)

            st.write("Model details")
            st.json(
                {
                    "model_name": result["model_name"],
                    "threshold": result["threshold"],
                    "prediction": result["prediction"],
                }
            )
    else:
        st.info("Enter book information and submit the form to get a prediction.")


st.divider()
st.subheader("Technical Notes")
st.write(
    "When you submit the form, this Streamlit app sends the book information to a "
    "FastAPI model service deployed on Google Cloud Run. The API loads the trained "
    "gradient boosting model, applies the same feature engineering used during training, "
    "and returns a bestseller probability."
)
