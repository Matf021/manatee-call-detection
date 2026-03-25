import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# Constants
TIME_STEPS = 14
FEATURES_PER_STEP = 13

# Paths
MODEL_DIR = "models/lstm_detection"
MODEL_H5 = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PKL = os.path.join(MODEL_DIR, "scaler.pkl")

# Load model and scaler once
model = load_model(MODEL_H5)
scaler = pickle.load(open(SCALER_PKL, "rb"))

def run_detection_model(df: pd.DataFrame, file_name: str):
    df = df.copy()
    df.fillna(0, inplace=True)

    # Drop any metadata columns if they exist
    drop_cols = [col for col in ['begin time (s)', 'end time (s)', 'label'] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Extract feature columns for the model
    feature_cols = [col for col in df.columns if "_seq_" in col and df[col].dtype != "object"]
    X = df[feature_cols].values

    # Reshape into LSTM-friendly shape: (samples, time_steps, features)
    try:
        X = X.reshape((-1, TIME_STEPS, FEATURES_PER_STEP))
    except Exception as e:
        raise ValueError(f"Feature shape mismatch. Make sure you have {TIME_STEPS * FEATURES_PER_STEP} values per row. Error: {str(e)}")

    # Apply scaler per timestep
    X_scaled = np.array([scaler.transform(X[:, i, :]) for i in range(TIME_STEPS)]).transpose(1, 0, 2)

    # Model prediction
    preds = model.predict(X_scaled, verbose=0)

    output = []
    for i, row in df.iterrows():
        confidence = float(preds[i][0])
        output.append({
            "start_time": row.get("start_time", round(i * 0.01, 3)),  # fallback if missing
            "end_time": row.get("end_time", round((i + TIME_STEPS - 1) * 0.01 + 0.025, 3)),
            "confidence": f"{confidence:.2f}",
            "is_manatee": "Yes" if confidence > 0.5 else "No",
            "file": file_name
        })

    return output
