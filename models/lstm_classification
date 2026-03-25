import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

# Load model and scaler
BASE_DIR = os.path.dirname(__file__)
model = load_model(os.path.join(BASE_DIR, "cnn_lstm_manatee_classifier_3class.h5"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

def run_classification_model(df: pd.DataFrame, base_results=None):
    # Filter only detected manatee samples
    df = df[df["is_manatee"] == "Yes"]
    if df.empty:
        return base_results if base_results else []

    # Drop metadata columns to isolate only numeric features
    drop_cols = ["start_time", "end_time", "file", "confidence", "is_manatee", "wav_file"]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    # Fill missing values and prepare input
    df = df.fillna(0)
    X = df[feature_cols].values

    # Reshape for LSTM input (time_steps × features_per_step)
    time_steps = 14
    features_per_step = X.shape[1] // time_steps
    X = X[:, :time_steps * features_per_step].reshape((X.shape[0], time_steps, features_per_step))

    # Apply scaling per time step
    X_scaled = np.array([scaler.transform(X[:, i, :]) for i in range(time_steps)]).transpose(1, 0, 2)

    # Predict class probabilities and get class index
    y_pred = np.argmax(model.predict(X_scaled, verbose=0), axis=1)
    label_map = {0: "HS", 1: "K", 2: "Others"}

    # Add prediction to results
    if base_results:
        manatee_idx = 0
        for i in range(len(base_results)):
            if base_results[i]["is_manatee"] == "Yes":
                base_results[i]["vocal_type"] = label_map[y_pred[manatee_idx]]
                manatee_idx += 1
        return base_results
    else:
        df["vocal_type"] = [label_map[i] for i in y_pred]
        return df.to_dict(orient="records")
