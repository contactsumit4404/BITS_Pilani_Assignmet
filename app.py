import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "Decision Tree": os.path.join(MODEL_DIR, "decision_tree.pkl"),
    "kNN": os.path.join(MODEL_DIR, "knn.pkl"),
    "Naive Bayes": os.path.join(MODEL_DIR, "naive_bayes.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
}
SCALER_PATH = os.path.join(MODEL_DIR, "minmax_scaler.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


def normalize_diagnosis(value):
    """
    Accept all diagnosis formats used by the assignment/deployment data:
        B / Benign / 0 -> 0
        M / Malignant / 1 -> 1
    Returns NaN for missing/invalid values.
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()

    mapping = {
        "b": 0,
        "benign": 0,
        "0": 0,
        "m": 1,
        "malignant": 1,
        "1": 1,
    }

    return mapping.get(s, np.nan)


def validate_and_normalize_diagnosis(df):
    """Validate diagnosis only when the uploaded CSV contains the target column."""
    if "diagnosis" not in df.columns:
        return df, None

    raw = df["diagnosis"].copy()
    normalized = raw.apply(normalize_diagnosis)

    invalid_mask = normalized.isna()

    if invalid_mask.any():
        invalid_values = raw[invalid_mask].dropna().astype(str).unique().tolist()
        missing_count = int(raw.isna().sum())

        msg = "Invalid or missing values were found in the diagnosis column.\n\n"
        if invalid_values:
            msg += f"Invalid values: {invalid_values}\n"
        if missing_count:
            msg += f"Missing values: {missing_count}\n"

        msg += (
            "\nExpected values are:\n"
            "B = Benign, M = Malignant\n"
            "or\n"
            "0 = Benign, 1 = Malignant"
        )
        raise ValueError(msg)

    df = df.copy()
    df["diagnosis"] = normalized.astype(int)
    return df, df["diagnosis"]


def load_artifacts():
    missing = []

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing.append(path)

    if not os.path.exists(SCALER_PATH):
        missing.append(SCALER_PATH)

    if not os.path.exists(FEATURE_PATH):
        missing.append(FEATURE_PATH)

    if missing:
        st.error("Required deployment files are missing:")
        for path in missing:
            st.write(f"- `{path}`")
        st.info(
            "Run the updated assignment notebook once to create the model "
            "files, scaler, and feature-column metadata."
        )
        st.stop()

    models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    return models, scaler, feature_columns


models, scaler, feature_columns = load_artifacts()

st.title("🩺 Breast Cancer Diagnosis Prediction")
st.write(
    "Upload a CSV containing the 30 model features. The `diagnosis` column is "
    "optional for prediction. If it is present, the app accepts B/M or 0/1."
)

selected_model = st.selectbox("Select ML Model", list(models.keys()))
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(data.head())

        # Validate diagnosis only if it exists.
        data, actual_diagnosis = validate_and_normalize_diagnosis(data)

        missing_features = [c for c in feature_columns if c not in data.columns]
        extra_columns = [c for c in data.columns if c not in feature_columns and c != "diagnosis"]

        if missing_features:
            st.error(
                "The uploaded CSV is missing required model features:\n\n"
                + ", ".join(missing_features)
            )
            st.stop()

        # Use exactly the same feature order as training.
        X = data[feature_columns].copy()

        # Convert features to numeric and reject invalid/missing values.
        for col in feature_columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        if X.isna().any().any():
            bad_columns = X.columns[X.isna().any()].tolist()
            st.error(
                "Missing or non-numeric values were found in model features: "
                + ", ".join(bad_columns)
            )
            st.stop()

        X_scaled = scaler.transform(X)

        model = models[selected_model]
        predictions = model.predict(X_scaled)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = np.full(len(predictions), np.nan)

        result = data.copy()
        result["Predicted_Diagnosis"] = np.where(
            predictions == 1, "Malignant", "Benign"
        )
        result["Predicted_Label"] = predictions.astype(int)

        if not np.isnan(probabilities).all():
            result["Malignant_Probability"] = np.round(probabilities, 4)

        st.subheader("Prediction Results")
        st.dataframe(result)

        if actual_diagnosis is not None:
            actual = actual_diagnosis.to_numpy()
            predicted = predictions.astype(int)
            accuracy = float((actual == predicted).mean())

            st.subheader("Evaluation on Uploaded Data")
            st.metric("Accuracy", f"{accuracy:.4f}")

            comparison = pd.DataFrame({
                "Actual": np.where(actual == 1, "Malignant", "Benign"),
                "Predicted": np.where(predicted == 1, "Malignant", "Benign"),
            })
            st.dataframe(comparison)

        csv_output = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Results",
            data=csv_output,
            file_name="prediction_results.csv",
            mime="text/csv",
        )

    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Deployment error: {e}")
