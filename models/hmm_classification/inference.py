import pickle
import pandas as pd
import re
import numpy as np

BASE_DIR = __file__.rsplit("/", 1)[0] if "/" in __file__ else __file__.rsplit("\\", 1)[0]
model_k = pickle.load(open(f"{BASE_DIR}/model_K_classification.pkl", "rb"))
model_hs = pickle.load(open(f"{BASE_DIR}/model_HS_classification.pkl", "rb"))
scaler = pickle.load(open(f"{BASE_DIR}/scaler_mfcc_classification.pkl", "rb"))

def run_classification_model(df: pd.DataFrame, base_results=None):
    pattern = re.compile(r"((?:mfcc|delta|delta_delta|energy|spectral_energy|log_power))(?:_[0-9]+)?_seq_(\d+)")
    frame_columns = {}

    for col in df.columns:
        m = pattern.match(col)
        if m:
            feature_name = m.group(1)
            frame_idx = int(m.group(2))
            frame_columns.setdefault(frame_idx, []).append(col)

    sorted_frames = sorted(frame_columns.keys())

    def sort_columns(col_list):
        feature_order = ["mfcc", "delta", "delta_delta", "energy", "spectral_energy", "log_power"]
        def extract_key(col):
            match = pattern.match(col)
            if match:
                return (feature_order.index(match.group(1)), int(match.group(2)))
            return (float("inf"), float("inf"))
        return sorted(col_list, key=extract_key)

    for k in frame_columns:
        frame_columns[k] = sort_columns(frame_columns[k])

    predicted_labels = []
    for idx, row in df.iterrows():
        frames = []
        for frame_idx in sorted_frames:
            feat_values = row[frame_columns[frame_idx]].values.astype(np.float64)
            if np.all(np.isnan(feat_values)):
                continue
            feat_values = np.nan_to_num(feat_values, nan=0.0)
            frames.append(feat_values)

        if not frames:
            predicted_labels.append("Others")
            continue

        seq_array = np.vstack(frames)
        seq_scaled = scaler.transform(seq_array)
        score_k = model_k.score(seq_scaled)
        score_hs = model_hs.score(seq_scaled)
        predicted = "K" if score_k > score_hs else "HS"
        predicted_labels.append(predicted)

    if base_results:
        vocal_index = 0
        for i in range(len(base_results)):
            if base_results[i]["is_manatee"] == "Yes":
                base_results[i]["vocal_type"] = predicted_labels[vocal_index]
                vocal_index += 1
        return base_results

    df["vocal_type"] = predicted_labels
    return df.to_dict(orient="records")
