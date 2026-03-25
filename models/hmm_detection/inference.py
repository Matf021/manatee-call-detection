import os
import re
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
model_manatee = pickle.load(open(os.path.join(BASE_DIR, "detection_manatee_mfcc5000.pkl"), "rb"))
model_others = pickle.load(open(os.path.join(BASE_DIR, "detection_others_mfcc5000.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "detection_mfcc5000.pkl"), "rb"))

def run_detection_model(df: pd.DataFrame, file_name: str):
    pattern = re.compile(r"((?:mfcc|delta|delta_delta|energy|spectral_energy|log_power))(?:_[0-9]+)?_seq_(\d+)")
    frame_columns = {}

    # Extract and group columns by time frame
    for col in df.columns:
        m = pattern.match(col)
        if m:
            feature = m.group(1)
            frame_idx = int(m.group(2))
            frame_columns.setdefault(frame_idx, []).append(col)

    sorted_frames = sorted(frame_columns.keys())

    def sort_columns(col_list):
        order = ["mfcc", "delta", "delta_delta", "energy", "spectral_energy", "log_power"]
        def key(col):
            match = pattern.match(col)
            if match:
                return (order.index(match.group(1)), int(match.group(2)))
            return (float("inf"), float("inf"))
        return sorted(col_list, key=key)

    for k in frame_columns:
        frame_columns[k] = sort_columns(frame_columns[k])

    results = []
    for i, row in df.iterrows():
        frames = []
        for frame_idx in sorted_frames:
            vals = row[frame_columns[frame_idx]].astype(np.float64)
            if np.all(np.isnan(vals)):
                continue
            vals = np.nan_to_num(vals, nan=0.0)
            frames.append(vals)

        if not frames:
            continue

        X = np.vstack(frames)
        X_scaled = scaler.transform(X)

        score_manatee = model_manatee.score(X_scaled)
        score_others = model_others.score(X_scaled)
        label = "Yes" if score_manatee > score_others else "No"
        confidence = round(abs(score_manatee - score_others), 2)

        results.append({
            "start_time": row.get("start_time", 0),
            "end_time": row.get("end_time", 0),
            "confidence": f"{confidence:.2f}",
            "is_manatee": label,
            "file": file_name
        })

    return results
