"""
model.py
ML training pipeline for CardioAI.
Trains 4 classifiers, selects the best, saves it with joblib.

Run standalone:
    cd backend
    python model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
FEATURES = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
    "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
    "High_BP", "High_Cholesterol", "Diabetes", "Smoking", "Obesity",
    "Sedentary_Lifestyle", "Family_History", "Chronic_Stress",
    "Gender", "Age",
]
TARGET       = "Heart_Risk"
MODEL_PATH   = "saved_model.pkl"
SCALER_PATH  = "saved_scaler.pkl"
RESULTS_PATH = "model_results.json"

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=15, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# ---------------------------------------------------------------------------

def train_and_evaluate(csv_path: str) -> dict:
    """
    Load dataset, train all classifiers, save the best model.

    Parameters
    ----------
    csv_path : str
        Path to the heart disease CSV file.

    Returns
    -------
    dict
        Summary of all model results, feature importance, and dataset stats.
    """
    print(f"\nLoading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Fill missing values with column median
    for col in FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled missing values in '{col}' with median={median_val}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    results    = {}
    best_acc   = 0
    best_model = None
    best_name  = ""

    print("\nTraining models...\n")
    for name, clf in CLASSIFIERS.items():
        print(f"  [{name}]", end="  ")
        clf.fit(X_train_sc, y_train)
        preds = clf.predict(X_test_sc)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        f1   = f1_score(y_test, preds, zero_division=0)
        cm   = confusion_matrix(y_test, preds).tolist()

        print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

        results[name] = {
            "accuracy":         round(acc,  4),
            "precision":        round(prec, 4),
            "recall":           round(rec,  4),
            "f1_score":         round(f1,   4),
            "confusion_matrix": cm,
        }

        if acc > best_acc:
            best_acc   = acc
            best_model = clf
            best_name  = name

    # Feature importance from Random Forest
    rf = CLASSIFIERS["Random Forest"]
    feature_importance = {
        feat: round(float(imp), 4)
        for feat, imp in sorted(
            zip(FEATURES, rf.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
    }

    # Save best model and scaler
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)

    output = {
        "best_model":         best_name,
        "best_accuracy":      round(best_acc, 4),
        "model_results":      results,
        "feature_importance": feature_importance,
        "dataset_stats": {
            "total":     len(df),
            "high_risk": int(df[TARGET].sum()),
            "low_risk":  int((df[TARGET] == 0).sum()),
            "avg_age":   round(float(df["Age"].mean()), 1),
            "features":  FEATURES,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBest model : {best_name}  (Accuracy: {best_acc:.4f})")
    print(f"Saved to   : {MODEL_PATH}  +  {SCALER_PATH}")
    return output


def load_model():
    """
    Load the saved model and scaler from disk.

    Returns
    -------
    tuple : (model, scaler)

    Raises
    ------
    FileNotFoundError if model hasn't been trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model found at '{MODEL_PATH}'.\n"
            "Run training first:  python model.py"
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_risk(patient_data: dict, model=None, scaler=None) -> dict:
    """
    Predict heart disease risk for one patient.

    Parameters
    ----------
    patient_data : dict
        Keys must match the FEATURES list.
    model  : trained sklearn model (optional — loads from disk if None)
    scaler : fitted StandardScaler  (optional — loads from disk if None)

    Returns
    -------
    dict with keys: probability, risk_percent, prediction, risk_level
    """
    if model is None or scaler is None:
        model, scaler = load_model()

    values     = [float(patient_data.get(f, 0)) for f in FEATURES]
    arr        = np.array(values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    probability = float(model.predict_proba(arr_scaled)[0][1])
    prediction  = int(model.predict(arr_scaled)[0])

    if probability >= 0.6:
        risk_level = "High"
    elif probability >= 0.35:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "probability":  round(probability, 4),
        "risk_percent": round(probability * 100, 1),
        "prediction":   prediction,
        "risk_level":   risk_level,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    csv_file = os.path.join(
        os.path.dirname(__file__),
        "..", "dataset",
        "heart_disease_risk_dataset_earlymed.csv"
    )
    train_and_evaluate(csv_file)
