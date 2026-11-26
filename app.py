from __future__ import annotations

import json
import os
import sqlite3
import uuid
import hmac
import smtplib
from contextlib import contextmanager
from email.message import EmailMessage
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(exist_ok=True)

# Render's filesystem is temporary unless a persistent disk is attached.
# For a demo, reports are written to /tmp; configure REPORT_DIR if needed.
REPORT_DIR = Path(os.environ.get("REPORT_DIR", "/tmp/netguard_reports"))
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.environ.get("DATABASE_PATH", str(INSTANCE_DIR / "netguard_history.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "development-key-change-this")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB Flask request limit
app.config["UPLOAD_EXTENSIONS"] = {".csv"}

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "change-this-password")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH")
CHUNK_SIZE = int(os.environ.get("PREDICTION_BATCH_SIZE", "2000"))

# Load the existing models and feature list from the repository root.
try:
    model = joblib.load(BASE_DIR / "best_model.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    anomaly_model = joblib.load(BASE_DIR / "anomaly_model.pkl")
except Exception as exc:
    raise RuntimeError(
        "Could not load best_model.pkl, feature_columns.pkl, or anomaly_model.pkl. "
        "Check that all three compatible files are in the repository root."
    ) from exc

# Optional multiclass model. The current binary model alone cannot name attack types.
multiclass_model = None
multiclass_encoder = None
try:
    multiclass_path = BASE_DIR / "multiclass_model.pkl"
    encoder_path = BASE_DIR / "multiclass_label_encoder.pkl"
    if multiclass_path.exists() and encoder_path.exists():
        multiclass_model = joblib.load(multiclass_path)
        multiclass_encoder = joblib.load(encoder_path)
except Exception:
    app.logger.exception("Optional multiclass model could not be loaded")


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


def init_db() -> None:
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


def login_required(view):
    from functools import wraps

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def load_metrics() -> dict[str, Any]:
    path = BASE_DIR / "model_metrics.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        app.logger.exception("Could not read model_metrics.json")
        return {}


def get_probability_details(estimator, data: pd.DataFrame, predictions: np.ndarray):
    """Return predicted-class confidence and, when possible, probability of class 1."""
    confidence = np.full(len(data), np.nan, dtype=float)
    attack_probability = np.full(len(data), np.nan, dtype=float)
    if not hasattr(estimator, "predict_proba"):
        return confidence, attack_probability
    try:
        probabilities = estimator.predict_proba(data)
        classes = list(getattr(estimator, "classes_", []))
        if probabilities.ndim == 2 and probabilities.shape[0] == len(data):
            confidence = np.max(probabilities, axis=1)
            if 1 in classes:
                attack_probability = probabilities[:, classes.index(1)]
    except Exception:
        app.logger.exception("Could not calculate model probabilities")
    return confidence, attack_probability


def to_attack_type(predicted_value: Any) -> str:
    if multiclass_encoder is not None:
        try:
            decoded = multiclass_encoder.inverse_transform(
                np.asarray([int(predicted_value)], dtype=int)
            )[0]
            return str(decoded)
        except Exception:
            pass
    return str(predicted_value)


def build_severity(is_attack: bool, is_anomaly: bool) -> str:
    # Triage labels only; these are not a verified measure of real-world impact.
    if is_attack and is_anomaly:
        return "High"
    if is_attack:
        return "Medium"
    if is_anomaly:
        return "Low"
    return "Informational"



def send_email_alert(filename: str, total: int, attacks: int, anomalies: int) -> bool:
    """Send an optional notification when SMTP settings are configured."""
    host = os.environ.get("SMTP_HOST")
    recipient = os.environ.get("ALERT_TO_EMAIL")
    username = os.environ.get("SMTP_USERNAME")
    password = os.environ.get("SMTP_PASSWORD")
    if not all([host, recipient, username, password]):
        return False
    try:
        message = EmailMessage()
        message["Subject"] = f"NetGuard AI alert: {attacks} attack predictions"
        message["From"] = username
        message["To"] = recipient
        message.set_content(
            "NetGuard AI completed a traffic analysis.\n\n"
            f"File: {filename}\nTotal records: {total}\n"
            f"Attack predictions: {attacks}\nAnomalies: {anomalies}\n\n"
            "Please review the dashboard and verify the detections."
        )
        port = int(os.environ.get("SMTP_PORT", "587"))
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(message)
        return True
    except Exception:
        app.logger.exception("Optional email alert could not be sent")
        return False


def analysis_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["category_counts"] = json.loads(item["category_counts"] or "{}")
    item["severity_counts"] = json.loads(item["severity_counts"] or "{}")
    return item


def dashboard_context(latest: dict[str, Any] | None = None):
    with get_db() as connection:
        rows = connection.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC LIMIT 15"
        ).fetchall()
    history = [analysis_to_dict(row) for row in rows]
    metrics = load_metrics()
    return {
        "latest": latest,
        "history": history,
        "metrics": metrics,
        "multiclass_ready": multiclass_model is not None,
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        password_ok = (
            check_password_hash(ADMIN_PASSWORD_HASH, password)
            if ADMIN_PASSWORD_HASH
            else password == ADMIN_PASSWORD
        )
        if username == ADMIN_USERNAME and password_ok:
            session.clear()
            session["logged_in"] = True
            session["username"] = username
            return redirect(request.args.get("next") or url_for("home"))
        flash("Incorrect username or password.", "error")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    latest = None
    with get_db() as connection:
        row = connection.execute(
            "SELECT * FROM analyses ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row:
        latest = analysis_to_dict(row)
    return render_template("index.html", **dashboard_context(latest))


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        flash("Please choose a CSV file first.", "error")
        return redirect(url_for("home"))

    original_name = secure_filename(uploaded_file.filename)
    if Path(original_name).suffix.lower() not in app.config["UPLOAD_EXTENSIONS"]:
        flash("Only CSV files are supported.", "error")
        return redirect(url_for("home"))

    analysis_id = uuid.uuid4().hex
    output_path = REPORT_DIR / f"{analysis_id}_predictions.csv"
    totals = {"total": 0, "attacks": 0, "normal": 0, "anomalies": 0}
    category_counts: dict[str, int] = {}
    severity_counts = {
        "High": 0,
        "Medium": 0,
        "Low": 0,
        "Informational": 0,
    }
    preview: list[dict[str, Any]] = []
    first_chunk = True

    try:
        uploaded_file.stream.seek(0)
        chunks = pd.read_csv(uploaded_file.stream, chunksize=CHUNK_SIZE)
        for chunk in chunks:
            if chunk.empty:
                continue

            missing = [column for column in feature_columns if column not in chunk.columns]
            if missing:
                raise ValueError(
                    f"CSV is missing {len(missing)} required model features. "
                    f"Example missing columns: {', '.join(missing[:5])}"
                )

            X = chunk[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            if X.isnull().any().any():
                raise ValueError(
                    "CSV contains missing or non-numeric values in required feature columns. "
                    "Clean those values before uploading."
                )

            binary_predictions = np.asarray(model.predict(X))
            anomaly_predictions = np.asarray(anomaly_model.predict(X))
            confidence, _ = get_probability_details(model, X, binary_predictions)

            is_attack_values = binary_predictions == 1
            is_anomaly_values = anomaly_predictions == -1

            if multiclass_model is not None:
                raw_types = np.asarray(multiclass_model.predict(X))
                predicted_types = [to_attack_type(value) for value in raw_types]
            else:
                predicted_types = ["Not configured"] * len(chunk)

            attack_labels = ["Attack" if value else "Normal" for value in is_attack_values]
            anomaly_labels = ["Anomaly" if value else "Normal" for value in is_anomaly_values]
            severities = [
                build_severity(bool(attack), bool(anomaly))
                for attack, anomaly in zip(is_attack_values, is_anomaly_values)
            ]

            # Only show a named attack category when the binary classifier also flags an attack.
            final_types = []
            for is_attack, predicted_type in zip(is_attack_values, predicted_types):
                if not is_attack:
                    final_types.append("BENIGN / Normal")
                elif multiclass_model is None:
                    final_types.append("Train multiclass model to identify type")
                else:
                    final_types.append(predicted_type)

            result_chunk = chunk.copy()
            result_chunk["Attack_Prediction"] = attack_labels
            result_chunk["Anomaly_Status"] = anomaly_labels
            result_chunk["Predicted_Attack_Type"] = final_types
            result_chunk["Prediction_Confidence_Percent"] = np.where(
                np.isnan(confidence), "Not available", np.round(confidence * 100, 2)
            )
            result_chunk["Threat_Severity"] = severities

            result_chunk.to_csv(
                output_path,
                index=False,
                mode="w" if first_chunk else "a",
                header=first_chunk,
            )
            first_chunk = False

            count = len(result_chunk)
            attack_count = int(np.sum(is_attack_values))
            anomaly_count = int(np.sum(is_anomaly_values))
            totals["total"] += count
            totals["attacks"] += attack_count
            totals["normal"] += count - attack_count
            totals["anomalies"] += anomaly_count

            for category, is_attack in zip(final_types, is_attack_values):
                if is_attack:
                    category_counts[category] = category_counts.get(category, 0) + 1
            for severity in severities:
                severity_counts[severity] += 1

            if len(preview) < 100:
                remaining = 100 - len(preview)
                preview.extend(
                    result_chunk[
                        [
                            "Attack_Prediction",
                            "Anomaly_Status",
                            "Predicted_Attack_Type",
                            "Prediction_Confidence_Percent",
                            "Threat_Severity",
                        ]
                    ].head(remaining).to_dict(orient="records")
                )

        if first_chunk or totals["total"] == 0:
            raise ValueError("The CSV file contains no data rows.")

        created_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        with get_db() as connection:
            connection.execute(
                """
                INSERT INTO analyses
                (id, created_at, filename, total, attacks, normal, anomalies,
                 category_counts, severity_counts, csv_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    created_at,
                    original_name,
                    totals["total"],
                    totals["attacks"],
                    totals["normal"],
                    totals["anomalies"],
                    json.dumps(category_counts),
                    json.dumps(severity_counts),
                    str(output_path),
                ),
            )

        email_alert_sent = False
        if totals["attacks"] > 0 or totals["anomalies"] > 0:
            email_alert_sent = send_email_alert(
                original_name, totals["total"], totals["attacks"], totals["anomalies"]
            )

        latest = {
            "id": analysis_id,
            "email_alert_sent": email_alert_sent,
            "created_at": created_at,
            "filename": original_name,
            **totals,
            "category_counts": category_counts,
            "severity_counts": severity_counts,
        }
        alerts = [
            row for row in preview
            if row["Attack_Prediction"] == "Attack" or row["Anomaly_Status"] == "Anomaly"
        ][:8]
        return render_template(
            "index.html",
            **dashboard_context(latest),
            preview=preview,
            alerts=alerts,
        )

    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        app.logger.exception("Network traffic CSV processing failed")
        flash(f"Could not analyse this CSV: {exc}", "error")
        return redirect(url_for("home"))



@app.route("/api/analyse-flow", methods=["POST"])
def analyse_flow_api():
    """Accept pre-extracted, authorised network-flow features from a trusted collector."""
    expected_key = os.environ.get("DEVICE_API_KEY")
    if not expected_key:
        return jsonify({"error": "Device API is not enabled on this deployment."}), 503
    supplied_header = request.headers.get("Authorization", "")
    supplied_key = supplied_header.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied_key, expected_key):
        return jsonify({"error": "Unauthorised."}), 401

    payload = request.get_json(silent=True)
    if isinstance(payload, dict) and isinstance(payload.get("flows"), list):
        rows = payload["flows"]
    elif isinstance(payload, list):
        rows = payload
    else:
        return jsonify({"error": "Send JSON as a list of flow objects or {flows: [...]} ."}), 400
    if not rows or len(rows) > 2000:
        return jsonify({"error": "Provide between 1 and 2,000 flow records per request."}), 400

    try:
        frame = pd.DataFrame(rows)
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            return jsonify({"error": "Missing required model features.", "missing_count": len(missing), "examples": missing[:5]}), 400
        X = frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X.isnull().any().any():
            return jsonify({"error": "Flow data has missing or non-numeric required features."}), 400

        binary = np.asarray(model.predict(X))
        anomaly = np.asarray(anomaly_model.predict(X))
        confidence, _ = get_probability_details(model, X, binary)
        if multiclass_model is not None:
            category_values = [to_attack_type(value) for value in multiclass_model.predict(X)]
        else:
            category_values = ["Not configured"] * len(frame)

        results = []
        for index in range(len(frame)):
            is_attack = bool(binary[index] == 1)
            is_anomaly = bool(anomaly[index] == -1)
            category = category_values[index] if is_attack else "BENIGN / Normal"
            results.append({
                "prediction": "Attack" if is_attack else "Normal",
                "anomaly": "Anomaly" if is_anomaly else "Normal",
                "predicted_attack_type": category,
                "confidence_percent": None if np.isnan(confidence[index]) else round(float(confidence[index]) * 100, 2),
                "severity": build_severity(is_attack, is_anomaly),
            })
        return jsonify({"results": results, "records_processed": len(results)})
    except Exception:
        app.logger.exception("Device flow API processing failed")
        return jsonify({"error": "Could not process the supplied flow data."}), 500


@app.route("/history/<analysis_id>")
@login_required
def history_detail(analysis_id: str):
    with get_db() as connection:
        row = connection.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        abort(404)
    latest = analysis_to_dict(row)
    return render_template("index.html", **dashboard_context(latest))


@app.route("/download/<analysis_id>")
@login_required
def download_csv(analysis_id: str):
    with get_db() as connection:
        row = connection.execute(
            "SELECT filename, csv_path FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        abort(404)
    path = Path(row["csv_path"])
    if not path.exists():
        flash("This report file is no longer available. Render storage may have restarted.", "error")
        return redirect(url_for("home"))
    return send_file(
        path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"NetGuard_{Path(row['filename']).stem}_predictions.csv",
    )


@app.route("/report/<analysis_id>.pdf")
@login_required
def download_pdf(analysis_id: str):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        flash("PDF support is missing. Add reportlab to requirements.txt and redeploy.", "error")
        return redirect(url_for("home"))

    with get_db() as connection:
        row = connection.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        abort(404)
    item = analysis_to_dict(row)

    buffer = __import__("io").BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("NetGuard AI – Network Security Analysis", styles["Title"]),
        Spacer(1, 5 * mm),
        Paragraph(f"File: {item['filename']}", styles["Normal"]),
        Paragraph(f"Analysis time: {item['created_at']}", styles["Normal"]),
        Spacer(1, 5 * mm),
    ]
    summary_data = [
        ["Metric", "Result"],
        ["Total records", str(item["total"])],
        ["Attack predictions", str(item["attacks"])],
        ["Normal predictions", str(item["normal"])],
        ["Anomalies", str(item["anomalies"])],
    ]
    summary_table = Table(summary_data, colWidths=[85 * mm, 75 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    elements.extend([Paragraph("Analysis summary", styles["Heading2"]), summary_table, Spacer(1, 5 * mm)])

    category_data = [["Predicted attack category", "Count"]]
    category_data.extend([[key, str(value)] for key, value in sorted(item["category_counts"].items())])
    if len(category_data) == 1:
        category_data.append(["No attack categories recorded", "0"])
    category_table = Table(category_data, colWidths=[110 * mm, 50 * mm])
    category_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    elements.extend([Paragraph("Attack category summary", styles["Heading2"]), category_table, Spacer(1, 5 * mm)])

    severity_data = [["Triage level", "Records"]]
    severity_data.extend([[key, str(value)] for key, value in item["severity_counts"].items()])
    severity_table = Table(severity_data, colWidths=[110 * mm, 50 * mm])
    severity_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172554")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    elements.extend([
        Paragraph("Threat triage summary", styles["Heading2"]),
        severity_table,
        Spacer(1, 6 * mm),
        Paragraph(
            "Note: Attack labels, confidence values and severity levels are model outputs for triage. "
            "They are not proof of malicious activity. Review alerts before taking action.",
            styles["Italic"],
        ),
    ])
    document.build(elements)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"NetGuard_{analysis_id[:8]}_report.pdf",
    )


@app.errorhandler(413)
def file_too_large(_error):
    flash("File is larger than this app's 100 MB upload limit.", "error")
    return redirect(url_for("home")), 413


@app.errorhandler(500)
def internal_error(_error):
    app.logger.exception("Unexpected server error")
    return "An unexpected server error occurred. Check the Render runtime logs.", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
