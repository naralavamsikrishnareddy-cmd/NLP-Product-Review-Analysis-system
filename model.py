import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "sentiment_model.pkl"
TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"
ENCODER_FILE = MODEL_DIR / "onehot_encoder.pkl"

# These are the categorical columns used with One-Hot Encoding.
CAT_COLUMNS = ["category", "product_type", "gender"]


def make_sentiment_from_rating(rating):
    rating = float(rating)
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"


def prepare_dataframe(df):
    """Clean column names and create missing fields when possible."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Accept a few common alternative names.
    rename_map = {
        "review": "review_text",
        "text": "review_text",
        "product": "product_name",
        "type": "product_type"
    }
    df.rename(columns=rename_map, inplace=True)

    required = ["product_name", "category", "product_type", "rating", "review_text", "gender"]
    for col in required:
        if col not in df.columns:
            if col == "rating":
                df[col] = 3
            elif col == "review_text":
                raise ValueError("Dataset must contain a review_text column (or review/review_text).")
            else:
                df[col] = "Unknown"

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3).clip(1, 5)

    if "sentiment" not in df.columns:
        df["sentiment"] = df["rating"].apply(make_sentiment_from_rating)
    else:
        df["sentiment"] = df["sentiment"].astype(str).str.strip().str.title()
        valid = {"Positive", "Negative", "Neutral"}
        df.loc[~df["sentiment"].isin(valid), "sentiment"] = df.loc[
            ~df["sentiment"].isin(valid), "rating"
        ].apply(make_sentiment_from_rating)

    for col in ["product_name", "category", "product_type", "gender", "review_text"]:
        df[col] = df[col].fillna("Unknown").astype(str)

    df = df[df["review_text"].str.strip() != ""].reset_index(drop=True)

    if len(df) < 6:
        raise ValueError("Please provide at least 6 non-empty reviews.")

    return df


def train_model(csv_path):
    """Train Logistic Regression using TF-IDF + One-Hot encoded categories."""
    df = prepare_dataframe(pd.read_csv(csv_path))

    df["clean_review"] = df["review_text"].apply(preprocess_text)

    # Text features
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df["clean_review"])

    # Categorical features
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_category = encoder.fit_transform(df[CAT_COLUMNS])

    # Combine TF-IDF and One-Hot features
    X = hstack([X_text, X_category]).tocsr()
    y = df["sentiment"]

    # Stratify only when every class has at least 2 records.
    counts = y.value_counts()
    use_stratify = y if len(counts) >= 2 and counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=use_stratify
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 3),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 3),
        "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 3),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=["Positive", "Neutral", "Negative"]
        ).tolist()
    }

    # Save model components.
    import joblib
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, TFIDF_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    # Top words from the TF-IDF vocabulary by average TF-IDF score.
    mean_scores = np.asarray(X_text.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_indices = mean_scores.argsort()[::-1][:15]
    top_words = [
        {"word": terms[i], "score": round(float(mean_scores[i]), 4)}
        for i in top_indices
    ]

    return df, {"metrics": metrics, "top_words": top_words}


def predict_sentiment(review, category="Unknown", product_type="Unknown", gender="Unknown"):
    """Predict sentiment for one new review."""
    import joblib

    if not MODEL_FILE.exists():
        raise ValueError("Model not trained yet. Upload/train a dataset first.")

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(TFIDF_FILE)
    encoder = joblib.load(ENCODER_FILE)

    clean_review = preprocess_text(review)
    X_text = vectorizer.transform([clean_review])

    category_df = pd.DataFrame([{
        "category": category,
        "product_type": product_type,
        "gender": gender
    }])
    X_category = encoder.transform(category_df[CAT_COLUMNS])

    X = hstack([X_text, X_category]).tocsr()
    return str(model.predict(X)[0])
