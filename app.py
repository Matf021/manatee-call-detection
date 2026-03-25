import os
import math
import pandas as pd
import librosa
import time
from flask import Flask, render_template, request
from flask import send_from_directory
from werkzeug.utils import secure_filename
from src.pipeline import process_audio_pipeline

# Detection model imports
from models.lstm_detection.inference import run_detection_model as run_lstm_detection
from models.hmm_detection.inference import run_detection_model as run_hmm_detection

app = Flask(__name__)
RESULTS_FOLDER = "results"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # upload & duration check
        file = request.files.get("audioFile")
        if not file:
            return "Missing audio file."
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        if librosa.get_duration(filename=save_path) > 3.5 * 3600:
            os.remove(save_path)
            return "⚠️ Audio exceeds 3 h 30 m."

        # feature extraction
        detection_df = process_audio_pipeline(
            save_path,
            filter_range=(500, 5000),
            selected_features=["mfcc"]
        )

        # run detectors
        results_hmm  = run_hmm_detection(detection_df, filename)
        results_lstm = run_lstm_detection(detection_df, filename)

        # compute mean of avg log‑likelihood per frame across all segments
        avg_lls = []
        for r in results_hmm:
            total_ll = float(r["confidence"])
            n_frames = 182  # 14 sequences * 13 mfcc features    
            avg_lls.append(total_ll / n_frames if n_frames else 0)
        mu = sum(avg_lls) / len(avg_lls)

        alpha = 1.0

        final_results = []
        for idx in range(len(detection_df)):
            hmm_r  = results_hmm[idx]
            lstm_r = results_lstm[idx]

            # per-frame average log‑likelihood
            raw_ll   = float(hmm_r["confidence"])
            n_frames = 182 # 14 sequences * 13 mfcc features
            avg_ll   = (raw_ll / n_frames) if n_frames else 0

            # sigmoid‑squash into [0,1]
            p_hmm = 1.0 / (1.0 + math.exp(-alpha * (avg_ll - mu)))

            # LSTM already gives p_lstm ∈ [0,1]
            p_lstm = float(lstm_r["confidence"])

            # soft‑voting
            p_yes = p_hmm + p_lstm
            p_no  = (1 - p_hmm) + (1 - p_lstm)
            final_label = "Yes" if p_yes >= p_no else "No"

            # for display, average the two confidences
            avg_conf = round((p_hmm + p_lstm) / 2, 2)

            final_results.append({
                "start_time":      f"{detection_df.iloc[idx]['start_time']:.2f}",
                "end_time":        f"{detection_df.iloc[idx]['end_time']:.2f}",
                "hmm_confidence":  f"{p_hmm:.2f}",
                "lstm_confidence": f"{p_lstm:.2f}",
                "avg_confidence":  f"{avg_conf:.2f}",
                "is_manatee":      final_label,
                "file":            filename
            })

        # save out a CSV
        df = pd.DataFrame(final_results)
        base, _ = os.path.splitext(filename)
        csv_fname = f"results_{base}_{int(time.time())}.csv"
        csv_path  = os.path.join(RESULTS_FOLDER, csv_fname)
        df.to_csv(csv_path, index=False)

        return render_template(
            "results.html",
            results=final_results,
            csv_filename=csv_fname
        )

    return render_template("index.html")

@app.route("/download_results/<filename>")
def download_results(filename):
    # build the absolute path to your folder
    results_dir = os.path.join(app.root_path, RESULTS_FOLDER)

    # full path to the requested file
    full_path = os.path.join(results_dir, filename)

    # 1) sanity check—fail fast with a clear message if missing
    if not os.path.isfile(full_path):
        return (
            f"⚠️ File not found on server: <code>{filename}</code>",
            404,
            {"Content-Type": "text/html"}
        )

    # 2) serve it
    return send_from_directory(
        directory=results_dir,
        filename=filename,
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True)
