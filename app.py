
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO

app = Flask(__name__)

# Maximum upload size: 100 MB
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Load trained models
model = joblib.load("best_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
anomaly_model = joblib.load("anomaly_model.pkl")

latest_results = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global latest_results

    if "file" not in request.files:
        return render_template(
            "index.html",
            error="Please select a CSV file."
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="No file selected."
        )

    if not file.filename.lower().endswith(".csv"):
        return render_template(
            "index.html",
            error="Please upload a CSV file only."
        )

    try:
        df = pd.read_csv(file)

        if df.empty:
            return render_template(
                "index.html",
                error="CSV file is empty."
            )

        # Check required features
        missing = [
            col for col in feature_columns
            if col not in df.columns
        ]

        if missing:
            return render_template(
                "index.html",
                error=f"Missing {len(missing)} required features."
            )

        # Prepare input data
        X = df[feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )

        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        if X.isnull().any().any():
            return render_template(
                "index.html",
                error="CSV contains missing or invalid values."
            )

        # Run predictions in batches
        batch_size = 2000
        predictions_list = []
        anomalies_list = []

        for start in range(0, len(X), batch_size):
            batch = X.iloc[start:start + batch_size]

            predictions_list.extend(
                model.predict(batch)
            )

            anomalies_list.extend(
                anomaly_model.predict(batch)
            )

        predictions = np.array(predictions_list)
        anomalies = np.array(anomalies_list)

        # Create results
        results = df.copy()

        results["Attack_Prediction"] = np.where(
            predictions == 1,
            "Attack",
            "Normal"
        )

        results["Anomaly_Status"] = np.where(
            anomalies == -1,
            "Anomaly",
            "Normal"
        )

        latest_results = results

        total = len(results)
        attacks = int((predictions == 1).sum())
        normal = total - attacks
        anomaly_count = int((anomalies == -1).sum())

        preview = results[
            ["Attack_Prediction", "Anomaly_Status"]
        ].head(10).to_dict(orient="records")

        return render_template(
            "index.html",
            total=total,
            attacks=attacks,
            normal=normal,
            anomaly_count=anomaly_count,
            preview=preview
        )

    except Exception as e:
        app.logger.exception("CSV processing failed")

        return render_template(
            "index.html",
            error=f"Unable to process CSV: {str(e)}"
        )


@app.route("/download")
def download():
    if latest_results is None:
        return "No prediction report available.", 404

    output = BytesIO()

    latest_results.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="NetGuard_Predictions.csv"
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="File exceeds the 100 MB upload limit."
    ), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
