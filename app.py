
import os
import json
import uuid
import hmac
import sqlite3
import smtplib

from pathlib import Path
from io import BytesIO
from contextlib import contextmanager
from datetime import datetime, timezone
from email.message import EmailMessage

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    flash,
    jsonify,
    abort
)

from werkzeug.utils import secure_filename


# ==========================================
# 1. FLASK CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(exist_ok=True)

REPORT_DIR = Path(
    os.environ.get("REPORT_DIR", "/tmp/netguard_reports")
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = Path(
    os.environ.get(
        "DATABASE_PATH",
        str(INSTANCE_DIR / "netguard_history.db")
    )
)

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "change-this-before-production"
)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

CHUNK_SIZE = int(
    os.environ.get("PREDICTION_BATCH_SIZE", "2000")
)


# ==========================================
# 2. LOAD EXISTING ML MODELS
# ==========================================

model = joblib.load(
    BASE_DIR / "best_model.pkl"
)

feature_columns = joblib.load(
    BASE_DIR / "feature_columns.pkl"
)

anomaly_model = joblib.load(
    BASE_DIR / "anomaly_model.pkl"
)


# ==========================================
# 3. LOAD MULTICLASS MODEL
# ==========================================

multiclass_model = None
multiclass_encoder = None

multiclass_path = BASE_DIR / "multiclass_model.pkl"
encoder_path = BASE_DIR / "multiclass_label_encoder.pkl"

if multiclass_path.exists() and encoder_path.exists():

    multiclass_model = joblib.load(multiclass_path)

    multiclass_encoder = joblib.load(encoder_path)

else:

    app.logger.warning(
        "Multiclass model files are not available. "
        "Only binary classification will be used."
    )


# ==========================================
# 4. DATABASE CONNECTION
# ==========================================

@contextmanager
def get_db():

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def init_db():

    with get_db() as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                total INTEGER NOT NULL,
                attacks INTEGER NOT NULL,
                normal INTEGER NOT NULL,
                anomalies INTEGER NOT NULL,
                category_counts TEXT NOT NULL,
                severity_counts TEXT NOT NULL,
                csv_path TEXT NOT NULL
            )
            """
        )


init_db()


# ==========================================
# 5. LOAD MODEL METRICS
# ==========================================

def load_metrics():

    metrics_path = BASE_DIR / "model_metrics.json"

    if not metrics_path.exists():
        return {}

    try:

        with open(metrics_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):

        app.logger.exception("Unable to load model metrics.")

        return {}


# ==========================================
# 6. PREDICTION CONFIDENCE
# ==========================================

def get_confidence(estimator, X):

    if not hasattr(estimator, "predict_proba"):

        return [None] * len(X)

    try:

        probabilities = estimator.predict_proba(X)

        confidence = np.max(probabilities, axis=1) * 100

        return np.round(confidence, 2).tolist()

    except Exception:

        app.logger.exception(
            "Prediction confidence could not be calculated."
        )

        return [None] * len(X)


# ==========================================
# 7. CONVERT MULTICLASS LABELS
# ==========================================

def decode_attack_types(predictions):

    if multiclass_encoder is None:

        return [str(value) for value in predictions]

    try:

        encoded = np.asarray(predictions, dtype=int)

        decoded = multiclass_encoder.inverse_transform(
            encoded
        )

        return [str(value) for value in decoded]

    except (ValueError, TypeError, AttributeError):

        app.logger.exception(
            "Unable to decode multiclass predictions."
        )

        return [str(value) for value in predictions]


# ==========================================
# 8. THREAT SEVERITY
# ==========================================

def get_severity(is_attack, is_anomaly):

    if is_attack and is_anomaly:
        return "High"

    if is_attack:
        return "Medium"

    if is_anomaly:
        return "Low"

    return "Informational"


# ==========================================
# 9. EMAIL ALERT FUNCTION
# ==========================================

def send_email_alert(
    filename,
    total,
    attacks,
    anomalies
):

    smtp_host = os.environ.get("SMTP_HOST")
    smtp_user = os.environ.get("SMTP_USERNAME")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    recipient = os.environ.get("ALERT_TO_EMAIL")

    if not all([
        smtp_host,
        smtp_user,
        smtp_password,
        recipient
    ]):

        return False

    try:

        message = EmailMessage()

        message["Subject"] = "NetGuard AI Security Alert"

        message["From"] = smtp_user
        message["To"] = recipient

        message.set_content(
            f"""
NetGuard AI has completed network traffic analysis.

File: {filename}

Total Records: {total}

Attack Predictions: {attacks}

Detected Anomalies: {anomalies}

Please review the dashboard.
"""
        )

        smtp_port = int(
            os.environ.get("SMTP_PORT", "587")
        )

        with smtplib.SMTP(
            smtp_host,
            smtp_port,
            timeout=15
        ) as server:

            server.starttls()

            server.login(
                smtp_user,
                smtp_password
            )

            server.send_message(message)

        return True

    except Exception:

        app.logger.exception("Email alert failed.")

        return False


# ==========================================
# 10. DATABASE HISTORY HELPERS
# ==========================================

def format_analysis(row):

    item = dict(row)

    item["category_counts"] = json.loads(
        item["category_counts"] or "{}"
    )

    item["severity_counts"] = json.loads(
        item["severity_counts"] or "{}"
    )

    return item


def get_history():

    with get_db() as connection:

        rows = connection.execute(
            """
            SELECT *
            FROM analyses
            ORDER BY created_at DESC
            LIMIT 50
            """
        ).fetchall()

    return [
        format_analysis(row)
        for row in rows
    ]


# ==========================================
# 11. HOME DASHBOARD
# ==========================================

@app.route("/")
def home():

    history = get_history()

    latest = history[0] if history else None

    return render_template(
        "index.html",
        latest=latest,
        history=history,
        metrics=load_metrics(),
        multiclass_ready=(
            multiclass_model is not None
        ),
        preview=[],
        alerts=[]
    )


# ==========================================
# 12. CSV UPLOAD AND ANALYSIS
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    uploaded_file = request.files.get("file")

    if uploaded_file is None or not uploaded_file.filename:

        flash("Please select a CSV file.", "error")

        return redirect(url_for("home"))

    filename = secure_filename(
        uploaded_file.filename
    )

    if Path(filename).suffix.lower() != ".csv":

        flash("Please upload a CSV file only.", "error")

        return redirect(url_for("home"))

    analysis_id = uuid.uuid4().hex

    output_path = (
        REPORT_DIR / f"{analysis_id}_predictions.csv"
    )

    totals = {
        "total": 0,
        "attacks": 0,
        "normal": 0,
        "anomalies": 0
    }

    category_counts = {}

    severity_counts = {
        "High": 0,
        "Medium": 0,
        "Low": 0,
        "Informational": 0
    }

    preview = []

    first_chunk = True

    try:

        chunks = pd.read_csv(
            uploaded_file.stream,
            chunksize=CHUNK_SIZE
        )

        for chunk in chunks:

            if chunk.empty:
                continue

            # Check required features

            missing = [
                column
                for column in feature_columns
                if column not in chunk.columns
            ]

            if missing:

                raise ValueError(
                    f"Missing {len(missing)} required columns. "
                    f"Examples: {', '.join(missing[:5])}"
                )

            # Prepare numerical input

            X = chunk[list(feature_columns)].apply(
                pd.to_numeric,
                errors="coerce"
            )

            X.replace(
                [np.inf, -np.inf],
                np.nan,
                inplace=True
            )

            if X.isnull().any().any():

                raise ValueError(
                    "CSV contains missing or invalid feature values."
                )

            # Binary attack prediction

            binary_predictions = np.asarray(
                model.predict(X)
            )

            # Anomaly detection

            anomaly_predictions = np.asarray(
                anomaly_model.predict(X)
            )

            # Prediction confidence

            confidence = get_confidence(
                model,
                X
            )

            is_attack = binary_predictions == 1

            is_anomaly = anomaly_predictions == -1

            # Multiclass attack prediction

            if multiclass_model is not None:

                raw_types = multiclass_model.predict(X)

                decoded_types = decode_attack_types(
                    raw_types
                )

            else:

                decoded_types = [
                    "Not configured"
                ] * len(chunk)

            # Readable output labels

            attack_labels = [
                "Attack" if value else "Normal"
                for value in is_attack
            ]

            anomaly_labels = [
                "Anomaly" if value else "Normal"
                for value in is_anomaly
            ]

            final_types = []

            for attack, category in zip(
                is_attack,
                decoded_types
            ):

                if not attack:

                    final_types.append("BENIGN / Normal")

                elif multiclass_model is None:

                    final_types.append(
                        "Multiclass model not configured"
                    )

                else:

                    final_types.append(category)

            severities = [
                get_severity(
                    bool(attack),
                    bool(anomaly)
                )
                for attack, anomaly in zip(
                    is_attack,
                    is_anomaly
                )
            ]

            # Add prediction results

            result_chunk = chunk.copy()

            result_chunk["Attack_Prediction"] = attack_labels

            result_chunk["Anomaly_Status"] = anomaly_labels

            result_chunk["Predicted_Attack_Type"] = final_types

            result_chunk["Prediction_Confidence_Percent"] = confidence

            result_chunk["Threat_Severity"] = severities

            # Write output CSV in batches

            result_chunk.to_csv(
                output_path,
                index=False,
                mode="w" if first_chunk else "a",
                header=first_chunk
            )

            first_chunk = False

            # Update dashboard statistics

            count = len(result_chunk)

            attack_count = int(np.sum(is_attack))

            anomaly_count = int(np.sum(is_anomaly))

            totals["total"] += count

            totals["attacks"] += attack_count

            totals["normal"] += count - attack_count

            totals["anomalies"] += anomaly_count

            for category, attack in zip(
                final_types,
                is_attack
            ):

                if attack:

                    category_counts[category] = (
                        category_counts.get(category, 0) + 1
                    )

            for severity in severities:

                severity_counts[severity] += 1

            # First 100 rows for preview

            remaining = 100 - len(preview)

            if remaining > 0:

                preview.extend(
                    result_chunk[
                        [
                            "Attack_Prediction",
                            "Anomaly_Status",
                            "Predicted_Attack_Type",
                            "Prediction_Confidence_Percent",
                            "Threat_Severity"
                        ]
                    ].head(remaining).to_dict(
                        orient="records"
                    )
                )

        if first_chunk or totals["total"] == 0:

            raise ValueError(
                "The uploaded CSV contains no data."
            )

        # Save analysis history

        created_at = (
            datetime.now(timezone.utc)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S %Z")
        )

        with get_db() as connection:

            connection.execute(
                """
                INSERT INTO analyses (
                    id,
                    created_at,
                    filename,
                    total,
                    attacks,
                    normal,
                    anomalies,
                    category_counts,
                    severity_counts,
                    csv_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    created_at,
                    filename,
                    totals["total"],
                    totals["attacks"],
                    totals["normal"],
                    totals["anomalies"],
                    json.dumps(category_counts),
                    json.dumps(severity_counts),
                    str(output_path)
                )
            )

        # Send optional email alert

        email_sent = False

        if totals["attacks"] > 0 or totals["anomalies"] > 0:

            email_sent = send_email_alert(
                filename,
                totals["total"],
                totals["attacks"],
                totals["anomalies"]
            )

        latest = {
            "id": analysis_id,
            "created_at": created_at,
            "filename": filename,
            **totals,
            "category_counts": category_counts,
            "severity_counts": severity_counts,
            "email_alert_sent": email_sent
        }

        alerts = [
            row
            for row in preview
            if (
                row["Attack_Prediction"] == "Attack"
                or row["Anomaly_Status"] == "Anomaly"
            )
        ][:10]

        return render_template(
            "index.html",
            latest=latest,
            history=get_history(),
            metrics=load_metrics(),
            multiclass_ready=(
                multiclass_model is not None
            ),
            preview=preview,
            alerts=alerts
        )

    except Exception as error:

        if output_path.exists():

            output_path.unlink(missing_ok=True)

        app.logger.exception(
            "CSV analysis failed."
        )

        flash(
            f"Analysis failed: {error}",
            "error"
        )

        return redirect(url_for("home"))


# ==========================================
# 13. HISTORY PAGE
# ==========================================

@app.route("/history")
def history():

    return render_template(
        "index.html",
        latest=None,
        history=get_history(),
        metrics=load_metrics(),
        multiclass_ready=(
            multiclass_model is not None
        ),
        preview=[],
        alerts=[],
        show_history=True
    )


# ==========================================
# 14. DOWNLOAD CSV REPORT
# ==========================================

@app.route("/download/<analysis_id>")
def download_csv(analysis_id):

    with get_db() as connection:

        row = connection.execute(
            """
            SELECT filename, csv_path
            FROM analyses
            WHERE id = ?
            """,
            (analysis_id,)
        ).fetchone()

    if row is None:

        abort(404)

    report_path = Path(row["csv_path"])

    if not report_path.exists():

        flash(
            "Report is no longer available on server storage.",
            "error"
        )

        return redirect(url_for("home"))

    download_name = (
        f"NetGuard_{Path(row['filename']).stem}_predictions.csv"
    )

    return send_file(
        report_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=download_name
    )


# ==========================================
# 15. GENERATE PDF REPORT
# ==========================================

@app.route("/report/<analysis_id>.pdf")
def download_pdf(analysis_id):

    try:

        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle
        )

    except ImportError:

        flash(
            "PDF library is not installed.",
            "error"
        )

        return redirect(url_for("home"))

    with get_db() as connection:

        row = connection.execute(
            """
            SELECT *
            FROM analyses
            WHERE id = ?
            """,
            (analysis_id,)
        ).fetchone()

    if row is None:

        abort(404)

    item = format_analysis(row)

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    elements = [

        Paragraph(
            "NetGuard AI - Security Analysis Report",
            styles["Title"]
        ),

        Spacer(1, 5 * mm),

        Paragraph(
            f"File: {item['filename']}",
            styles["Normal"]
        ),

        Paragraph(
            f"Analysis Date: {item['created_at']}",
            styles["Normal"]
        ),

        Spacer(1, 5 * mm)

    ]

    summary = [

        ["Metric", "Count"],

        ["Total Records", str(item["total"])],

        ["Attack Predictions", str(item["attacks"])],

        ["Normal Predictions", str(item["normal"])],

        ["Anomalies", str(item["anomalies"])]

    ]

    table = Table(
        summary,
        colWidths=[90 * mm, 65 * mm]
    )

    table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172554")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )

    elements.extend([

        Paragraph(
            "Analysis Summary",
            styles["Heading2"]
        ),

        table,

        Spacer(1, 6 * mm)

    ])

    categories = [
        ["Attack Category", "Count"]
    ]

    categories.extend([

        [name, str(count)]

        for name, count in item["category_counts"].items()

    ])

    if len(categories) == 1:

        categories.append([
            "No categories available",
            "0"
        ])

    category_table = Table(
        categories,
        colWidths=[90 * mm, 65 * mm]
    )

    category_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172554")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )

    elements.extend([

        Paragraph(
            "Attack Category Distribution",
            styles["Heading2"]
        ),

        category_table,

        Spacer(1, 6 * mm)

    ])

    severity_rows = [
        ["Severity", "Count"]
    ]

    severity_rows.extend([

        [name, str(count)]

        for name, count in item["severity_counts"].items()

    ])

    severity_table = Table(
        severity_rows,
        colWidths=[90 * mm, 65 * mm]
    )

    severity_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#172554")
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )

    elements.extend([

        Paragraph(
            "Threat Severity Summary",
            styles["Heading2"]
        ),

        severity_table,

        Spacer(1, 6 * mm),

        Paragraph(
            "Note: Machine learning predictions are indicators "
            "for investigation, not proof of an attack. "
            "Severity levels are triage labels and should "
            "be reviewed by an administrator.",
            styles["Italic"]
        )

    ])

    document.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=(
            f"NetGuard_Report_{analysis_id[:8]}.pdf"
        )
    )


# ==========================================
# 16. AUTHORISED DEVICE DATA API
# ==========================================

@app.route("/api/analyse-flow", methods=["POST"])
def analyse_flow_api():

    expected_key = os.environ.get("DEVICE_API_KEY")

    if not expected_key:

        return jsonify({
            "error": "Device API is not enabled."
        }), 503

    auth_header = request.headers.get(
        "Authorization", ""
    )

    supplied_key = auth_header.removeprefix(
        "Bearer "
    ).strip()

    if not hmac.compare_digest(
        supplied_key,
        expected_key
    ):

        return jsonify({
            "error": "Unauthorised request."
        }), 401

    payload = request.get_json(silent=True)

    if isinstance(payload, dict):

        rows = payload.get("flows")

    elif isinstance(payload, list):

        rows = payload

    else:

        rows = None

    if not isinstance(rows, list) or not rows or len(rows) > 2000:

        return jsonify({
            "error": "Send between 1 and 2000 flow records."
        }), 400

    try:

        frame = pd.DataFrame(rows)

        missing = [
            column
            for column in feature_columns
            if column not in frame.columns
        ]

        if missing:

            return jsonify({
                "error": "Required features are missing.",
                "missing_count": len(missing),
                "examples": missing[:5]
            }), 400

        X = frame[list(feature_columns)].apply(
            pd.to_numeric,
            errors="coerce"
        )

        X.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        if X.isnull().any().any():

            return jsonify({
                "error": "Invalid or missing feature values."
            }), 400

        binary_predictions = model.predict(X)

        anomaly_predictions = anomaly_model.predict(X)

        confidence = get_confidence(model, X)

        if multiclass_model is not None:

            categories = decode_attack_types(
                multiclass_model.predict(X)
            )

        else:

            categories = [
                "Not configured"
            ] * len(X)

        results = []

        for i in range(len(X)):

            attack = bool(binary_predictions[i] == 1)

            anomaly = bool(anomaly_predictions[i] == -1)

            results.append({

                "prediction": (
                    "Attack" if attack else "Normal"
                ),

                "anomaly": (
                    "Anomaly" if anomaly else "Normal"
                ),

                "predicted_attack_type": (
                    categories[i] if attack else "BENIGN / Normal"
                ),

                "confidence_percent": confidence[i],

                "severity": get_severity(
                    attack,
                    anomaly
                )

            })

        return jsonify({
            "records_processed": len(results),
            "results": results
        })

    except Exception:

        app.logger.exception(
            "Device flow analysis failed."
        )

        return jsonify({
            "error": "Unable to analyse flow data."
        }), 500


# ==========================================
# 17. UPLOAD SIZE ERROR
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "File exceeds the 100 MB application upload limit.",
        413
    )


# ==========================================
# 18. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
