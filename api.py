"""
Run locally:
    .venv/bin/python -m uvicorn api:app --reload

Docs:
    http://127.0.0.1:8000/docs
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

import modeling


MODEL_PATH = Path("models/bestseller_model.joblib")
API_VERSION = "v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("bestseller-api")


class BookMetadata(BaseModel):
    """Input fields used by the trained bestseller model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "Example Book",
                "author": "Example Author",
                "publisher": "Penguin Group",
                "publish_year": 2025,
                "page_count": 320,
                "ol_edition_count": 4,
                "ol_subjects": ["Fiction", "Romance"],
                "ol_ebook_access": "no_ebook",
                "ol_languages": ["eng"],
                "ol_first_publish_year": 2025,
            }
        }
    )

    title: str | None = Field(None, description="Book title.")
    author: str | None = Field(None, description="Primary author name.")
    publisher: str | None = Field(None, description="Publisher name.")
    publish_year: int | None = Field(None, ge=1400, le=2100, description="Publication year.")
    page_count: int | None = Field(None, ge=0, le=10000, description="Number of pages.")
    ol_edition_count: int | None = Field(
        None,
        ge=0,
        le=10000,
        description="Open Library edition count.",
    )
    ol_subjects: list[str] | str | None = Field(
        default_factory=list,
        description="Open Library subjects as a list or stringified list.",
    )
    ol_ebook_access: str | None = Field(
        None,
        description="Open Library ebook access value, e.g. no_ebook or borrowable.",
    )
    ol_languages: list[str] | str | None = Field(
        default_factory=list,
        description="Open Library languages as a list or stringified list.",
    )
    ol_first_publish_year: int | None = Field(
        None,
        ge=1400,
        le=2100,
        description="Open Library first publish year.",
    )

    @field_validator("ol_subjects", "ol_languages", mode="before")
    @classmethod
    def normalize_list_fields(cls, value):
        if value is None:
            return []
        return value


class PredictionResponse(BaseModel):
    bestseller_probability: float = Field(..., ge=0, le=1)
    prediction: int = Field(..., description="1 means likely bestseller at the selected threshold.")
    label: str
    threshold: float
    model_name: str


class BatchPredictionRequest(BaseModel):
    books: list[BookMetadata] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


def load_model_artifact() -> dict[str, Any]:
    """Load the trained model artifact from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python modeling.py` first."
        )
    return joblib.load(MODEL_PATH)


model_artifact = load_model_artifact()
pipeline = model_artifact["pipeline"]
model_name = model_artifact.get("model_name", "unknown")
threshold = float(model_artifact.get("threshold", 0.5))


app = FastAPI(
    title="NYT Bestseller Predictor API",
    description=(
        "Serves a trained model that estimates the likelihood that a book "
        "could become a New York Times bestseller from book metadata."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def book_to_row(book: BookMetadata) -> dict[str, Any]:
    """Convert a validated request object into the training-data column shape."""
    data = book.model_dump()
    return {
        "title": data.get("title"),
        "author": data.get("author"),
        "publisher": data.get("publisher"),
        "publish_year": data.get("publish_year"),
        "page_count": data.get("page_count"),
        "ol_edition_count": data.get("ol_edition_count"),
        "ol_subjects": data.get("ol_subjects") or [],
        "ol_ebook_access": data.get("ol_ebook_access"),
        "ol_languages": data.get("ol_languages") or [],
        "ol_first_publish_year": data.get("ol_first_publish_year"),
        # engineer_features expects this column to exist in training-shaped data.
        "is_bestseller": 0,
    }


def predict_books(books: list[BookMetadata]) -> list[PredictionResponse]:
    """Generate predictions for one or more books."""
    try:
        raw = pd.DataFrame([book_to_row(book) for book in books])
        features = modeling.engineer_features(raw)
        probabilities = pipeline.predict_proba(features)[:, 1]
    except Exception as error:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

    responses = []
    for probability in probabilities:
        probability = float(probability)
        prediction = int(probability >= threshold)
        responses.append(
            PredictionResponse(
                bestseller_probability=round(probability, 4),
                prediction=prediction,
                label="likely_bestseller" if prediction else "unlikely_bestseller",
                threshold=threshold,
                model_name=model_name,
            )
        )
    return responses


@app.get("/")
def root():
    return {
        "message": "NYT Bestseller Predictor API",
        "docs": "/docs",
        "health": "/health",
        "predict": f"/{API_VERSION}/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "model_path": str(MODEL_PATH),
    }


@app.get(f"/{API_VERSION}/model-info")
@app.get("/model-info")
def model_info():
    return {
        "model_name": model_name,
        "positive_class": model_artifact.get("positive_class", "NYT bestseller"),
        "threshold": threshold,
        "feature_columns": model_artifact.get("feature_columns", []),
    }


@app.post(f"/{API_VERSION}/predict", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
def predict(book: BookMetadata):
    logger.info("Received single prediction request.")
    return predict_books([book])[0]


@app.post(f"/{API_VERSION}/predict-batch", response_model=BatchPredictionResponse)
@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    logger.info("Received batch prediction request with %s books.", len(request.books))
    return BatchPredictionResponse(predictions=predict_books(request.books))
