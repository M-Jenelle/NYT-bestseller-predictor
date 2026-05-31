"""
Train and compare models for predicting NYT bestseller likelihood.

Inputs:
    data/processed/class_comparison_feature_design.csv

Outputs:
    models/bestseller_model.joblib
    reports/modeling/model_comparison.csv
    reports/modeling/test_predictions.csv
    reports/modeling/feature_importance.csv

Run:
    python modeling.py
"""

import argparse
import ast
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as error:
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = error
else:
    XGBOOST_IMPORT_ERROR = None


CURRENT_YEAR = 2026
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports/modeling")


class TopListItemsEncoder(BaseEstimator, TransformerMixin):
    """Multi-hot encode the most common items from a stringified list column."""

    def __init__(self, top_n=50, prefix="item"):
        self.top_n = top_n
        self.prefix = prefix

    def fit(self, X, y=None):
        values = _as_series(X)
        counts = Counter()
        for value in values:
            counts.update(parse_list(value))

        self.top_items_ = [
            item for item, _ in counts.most_common(self.top_n)
        ]
        self.feature_names_ = [
            f"{self.prefix}_{clean_feature_name(item)}"
            for item in self.top_items_
        ]
        return self

    def transform(self, X):
        values = _as_series(X)
        rows = []
        for value in values:
            item_set = set(parse_list(value))
            rows.append([int(item in item_set) for item in self.top_items_])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


# Keep joblib artifacts importable when this file is executed as a script.
sys.modules.setdefault("modeling", sys.modules[__name__])
TopListItemsEncoder.__module__ = "modeling"


def _as_series(X):
    """Return a 1D pandas Series from sklearn's possible transformer inputs."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0]
    if isinstance(X, pd.Series):
        return X
    return pd.Series(np.asarray(X).ravel())


def parse_list(value):
    """Parse list-like CSV strings into normalized string lists."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []

    text = str(value).strip()
    if text in ["", "[]", "nan", "None"]:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]

    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()]


def remove_target_leakage_subjects(value):
    """Remove subject tags that directly reveal NYT bestseller status."""
    blocked_terms = [
        "nyt:",
        "new york times bestseller",
        "new york times best seller",
        "bestseller",
        "best seller",
    ]
    safe_items = []
    for item in parse_list(value):
        lowered = item.lower()
        if any(term in lowered for term in blocked_terms):
            continue
        safe_items.append(item)
    return safe_items


def clean_feature_name(value):
    """Create compact feature names from arbitrary subject/language strings."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:60] or "unknown"


def first_item(value, default="unknown"):
    items = parse_list(value)
    return items[0] if items else default


def cap_rare_categories(series, top_n=40, other_label="Other"):
    """Keep common categories and group the long tail."""
    filled = series.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    top_values = filled.value_counts().head(top_n).index
    return filled.where(filled.isin(top_values), other_label)


def load_training_data(path):
    """Load the EDA class-comparison dataset."""
    df = pd.read_csv(path)
    df["is_bestseller"] = pd.to_numeric(df["is_bestseller"], errors="coerce").astype(int)
    return df


def engineer_features(df):
    """Create model-ready features from shared positive/negative fields."""
    features = df.copy()

    numeric_cols = [
        "publish_year",
        "page_count",
        "ol_edition_count",
        "ol_first_publish_year",
    ]
    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    # In the EDA, zero page counts showed up as placeholders for missingness.
    features.loc[features["page_count"] <= 0, "page_count"] = np.nan

    features["has_page_count"] = features["page_count"].notna().astype(int)
    features["has_ol_edition_count"] = features["ol_edition_count"].notna().astype(int)
    features["ol_subjects_model"] = features["ol_subjects"].apply(remove_target_leakage_subjects)
    features["has_ol_subjects"] = features["ol_subjects_model"].apply(lambda value: int(len(value) > 0))
    features["has_ebook_access"] = features["ol_ebook_access"].notna().astype(int)
    features["has_ol_languages"] = features["ol_languages"].apply(lambda value: int(len(parse_list(value)) > 0))
    features["has_first_publish_year"] = features["ol_first_publish_year"].notna().astype(int)

    features["book_age"] = CURRENT_YEAR - features["publish_year"]
    features["first_publish_age"] = CURRENT_YEAR - features["ol_first_publish_year"]
    features["publication_gap"] = features["publish_year"] - features["ol_first_publish_year"]

    features["log_page_count"] = np.log1p(features["page_count"])
    features["log_edition_count"] = np.log1p(features["ol_edition_count"])

    features["num_ol_subjects"] = features["ol_subjects_model"].apply(len)
    features["num_ol_languages"] = features["ol_languages"].apply(lambda value: len(parse_list(value)))
    features["primary_language"] = features["ol_languages"].apply(first_item)

    features["publisher_grouped"] = cap_rare_categories(features["publisher"], top_n=50)
    features["author_grouped"] = cap_rare_categories(features["author"], top_n=50)
    features["ol_ebook_access"] = (
        features["ol_ebook_access"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )

    keep_cols = [
        "publish_year",
        "page_count",
        "ol_edition_count",
        "ol_first_publish_year",
        "book_age",
        "first_publish_age",
        "publication_gap",
        "log_page_count",
        "log_edition_count",
        "num_ol_subjects",
        "num_ol_languages",
        "has_page_count",
        "has_ol_edition_count",
        "has_ol_subjects",
        "has_ebook_access",
        "has_ol_languages",
        "has_first_publish_year",
        "publisher_grouped",
        "author_grouped",
        "ol_ebook_access",
        "primary_language",
        "ol_subjects_model",
    ]
    return features[keep_cols]


def build_preprocessor():
    """Create preprocessing pipeline for numeric, categorical, and list features."""
    numeric_features = [
        "publish_year",
        "page_count",
        "ol_edition_count",
        "ol_first_publish_year",
        "book_age",
        "first_publish_age",
        "publication_gap",
        "log_page_count",
        "log_edition_count",
        "num_ol_subjects",
        "num_ol_languages",
        "has_page_count",
        "has_ol_edition_count",
        "has_ol_subjects",
        "has_ebook_access",
        "has_ol_languages",
        "has_first_publish_year",
    ]
    categorical_features = [
        "publisher_grouped",
        "author_grouped",
        "ol_ebook_access",
        "primary_language",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
            ("subjects", TopListItemsEncoder(top_n=75, prefix="subject"), ["ol_subjects_model"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(class_weight, scale_pos_weight):
    """Return model comparison candidates."""
    models = {
        "baseline_logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight=class_weight,
            random_state=42,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,
        )
    else:
        print(f"Skipping xgboost because it could not be imported: {XGBOOST_IMPORT_ERROR}")

    return models


def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test):
    """Fit one model and return test-set metrics plus predictions."""
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
    }

    prediction_frame = pd.DataFrame(
        {
            "model": name,
            "actual": y_test.to_numpy(),
            "predicted": predictions,
            "bestseller_probability": probabilities,
        },
        index=y_test.index,
    )
    return metrics, prediction_frame


def cross_validation_summary(name, pipeline, X_train, y_train):
    """Estimate performance stability on the training set."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )
    return {
        "model": name,
        **{
            f"cv_{metric}_mean": cv_scores[f"test_{metric}"].mean()
            for metric in scoring
        },
        **{
            f"cv_{metric}_std": cv_scores[f"test_{metric}"].std()
            for metric in scoring
        },
    }


def extract_feature_importance(best_name, best_pipeline):
    """Return feature importances or coefficients from the selected model."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
        score_name = "importance"
    elif hasattr(model, "coef_"):
        scores = model.coef_[0]
        score_name = "coefficient"
    else:
        return pd.DataFrame()

    importance = pd.DataFrame(
        {
            "model": best_name,
            "feature": feature_names,
            score_name: scores,
            "absolute_value": np.abs(scores),
        }
    )
    return importance.sort_values("absolute_value", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Train NYT bestseller prediction models.")
    parser.add_argument(
        "--input",
        default="data/processed/class_comparison_feature_design.csv",
        help="Path to class comparison CSV produced by analyze_data.py.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test size.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_training_data(args.input)
    X = engineer_features(raw)
    y = raw["is_bestseller"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count
    class_weight = {
        0: 1.0,
        1: scale_pos_weight,
    }

    metrics_rows = []
    cv_rows = []
    prediction_frames = []
    fitted_pipelines = {}

    for name, model in build_models(class_weight, scale_pos_weight).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )
        cv_rows.append(cross_validation_summary(name, pipeline, X_train, y_train))
        metrics, predictions = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        metrics_rows.append(metrics)
        prediction_frames.append(predictions)
        fitted_pipelines[name] = pipeline

    metrics_df = pd.DataFrame(metrics_rows)
    cv_df = pd.DataFrame(cv_rows)
    comparison = metrics_df.merge(cv_df, on="model")
    comparison = comparison.sort_values(
        ["average_precision", "roc_auc", "f1"],
        ascending=False,
    )

    best_name = comparison.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_name]
    best_predictions = next(
        frame for frame in prediction_frames
        if frame["model"].iloc[0] == best_name
    )

    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    pd.concat(prediction_frames).to_csv(REPORT_DIR / "test_predictions.csv", index_label="row_id")
    extract_feature_importance(best_name, best_pipeline).to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False,
    )
    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_pipeline,
            "feature_columns": X.columns.tolist(),
            "positive_class": "NYT bestseller",
            "threshold": 0.5,
        },
        MODEL_DIR / "bestseller_model.joblib",
    )

    print("Modeling complete.")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Train class ratio, negative:positive = {negative_count}:{positive_count}")
    print()
    print("Model comparison:")
    print(comparison[["model", "average_precision", "roc_auc", "f1", "precision", "recall", "accuracy"]].to_string(index=False))
    print()
    print(f"Selected model: {best_name}")
    print("Confusion matrix for selected model:")
    print(confusion_matrix(y_test, best_predictions["predicted"]))
    print()
    print("Classification report for selected model:")
    print(classification_report(y_test, best_predictions["predicted"], target_names=["not_bestseller", "bestseller"]))
    print()
    print(f"Saved model: {MODEL_DIR / 'bestseller_model.joblib'}")
    print(f"Saved reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
