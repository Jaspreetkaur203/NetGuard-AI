"""Train an optional multiclass attack-type model from NetGuard_Cleaned.csv.

Run in Colab or locally after placing NetGuard_Cleaned.csv and feature_columns.pkl
in the working directory. This script does NOT overwrite best_model.pkl or
anomaly_model.pkl. It creates the optional multiclass model and evaluation metrics.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("NetGuard_Cleaned.csv")
FEATURES_PATH = Path("feature_columns.pkl")

if not DATA_PATH.exists():
    raise FileNotFoundError("Place NetGuard_Cleaned.csv in the current folder.")
if not FEATURES_PATH.exists():
    raise FileNotFoundError("Place the existing feature_columns.pkl in the current folder.")

df = pd.read_csv(DATA_PATH)
if "Label" not in df.columns:
    raise ValueError("The dataset must contain the original multiclass 'Label' column.")

feature_columns = joblib.load(FEATURES_PATH)
missing = [column for column in feature_columns if column not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing required feature columns: {missing[:8]}")

X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
y_text = df["Label"].astype(str).str.strip()
valid = X.notna().all(axis=1) & y_text.notna() & (y_text != "")
X = X.loc[valid].reset_index(drop=True)
y_text = y_text.loc[valid].reset_index(drop=True)

encoder = LabelEncoder()
y = encoder.fit_transform(y_text)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=180,
    max_depth=24,
    min_samples_split=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
print("Training multiclass Random Forest...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, predictions, average="weighted", zero_division=0
)
metrics = {
    "multiclass": {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "classes": [str(label) for label in encoder.classes_],
        "test_records": int(len(y_test)),
    }
}

joblib.dump(model, "multiclass_model.pkl")
joblib.dump(encoder, "multiclass_label_encoder.pkl")
with open("model_metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("Training complete.")
print("Created: multiclass_model.pkl")
print("Created: multiclass_label_encoder.pkl")
print("Created: model_metrics.json")
print(json.dumps(metrics, indent=2))
