from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request

from model import train_model
from recommendation import get_recommendation

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "product_reviews.csv"

df = pd.DataFrame()
model_info = {}


def load_dataset():
    global df, model_info

    DATASET_DIR.mkdir(exist_ok=True)

    if not DATASET_PATH.exists():
        df = pd.DataFrame()
        model_info = {
            "error": "Dataset not found."
        }
        return

    try:
        df, model_info = train_model(str(DATASET_PATH))
        print("Dataset loaded successfully.")
        print("Model information:", model_info)

    except Exception as exc:
        df = pd.DataFrame()
        model_info = {
            "error": str(exc)
        }

        print("Model training error:", exc)


# Load dataset when application starts
load_dataset()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():

    if df.empty:
        return jsonify({
            "found": False,
            "message": "Dataset is empty. Please upload a CSV dataset."
        }), 400

    data = request.get_json(silent=True) or {}

    user_query = str(
        data.get("query", "")
    ).strip()

    if not user_query:
        return jsonify({
            "error": "Please enter what product you want to buy."
        }), 400

    try:
        result = get_recommendation(
            df,
            user_query
        )

        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500


@app.route("/upload", methods=["POST"])
def upload():

    global df, model_info

    file = request.files.get("file")

    if not file:
        return jsonify({
            "error": "Please select a CSV file."
        }), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({
            "error": "Only CSV files are allowed."
        }), 400

    try:

        DATASET_DIR.mkdir(exist_ok=True)

        file.save(DATASET_PATH)

        df, model_info = train_model(
            str(DATASET_PATH)
        )

        print("New dataset uploaded.")
        print("Model information:", model_info)

        return redirect("/")

    except Exception as exc:

        print("Upload error:", exc)

        return jsonify({
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )