import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from librosa.feature import melspectrogram, mfcc, delta
from librosa import yin
from spafe.features.gfcc import gfcc
from pydub import AudioSegment
import tempfile

# Configuration
SR = 48000
WINDOW_SIZE = 0.025  # 25 ms
STEP_SIZE = 0.01     # 10 ms
N_COEFFS = 13
MAX_SEQ = 14
TOP_DB = 60          # splitting threshold (dB below peak)
GAP_SILENCE = 0.01
MIN_DUR = 0.05

def pad_with_nan(values):
    return np.pad(values, ((0, max(0, MAX_SEQ - values.shape[0])), (0, 0)), constant_values=np.nan)[:MAX_SEQ]

def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(audio, sr, lowcut, highcut):
    b, a = butter_bandpass(lowcut, highcut, sr)
    return filtfilt(b, a, audio)

def ephraim_malah_noise_reduction(audio, sr):
    n_fft = 1024
    hop_length = 512
    stft_audio = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft_audio)
    phase = np.angle(stft_audio)
    noise_profile = np.mean(mag, axis=1, keepdims=True)
    alpha = 1.8
    beta = 0.025
    noise_est = alpha * noise_profile
    enhanced_mag = np.maximum(mag - noise_est, beta * mag)
    enhanced_stft = enhanced_mag * np.exp(1j * phase)
    return librosa.istft(enhanced_stft, hop_length=hop_length)

def ensure_wav_format(path):
    if path.lower().endswith(".mp3"):
        wav_path = path.replace(".mp3", ".converted.wav")
        AudioSegment.from_mp3(path).export(wav_path, format="wav")
        return wav_path
    return path

def safe_append(features, vals, expected_len=N_COEFFS):
    if isinstance(vals, (list, np.ndarray)) and len(vals) == expected_len:
        features.extend(vals)
    else:
        features.extend([np.nan] * expected_len)

def extract_frame_features(frame, sr, selected_features, mfcc_cache=None):
    features = []

    if "energy" in selected_features:
        try:
            features.append(librosa.feature.rms(y=frame)[0][0])
        except:
            features.append(np.nan)

    if "log_power" in selected_features:
        try:
            S = np.abs(librosa.stft(frame, n_fft=2048))**2
            power = np.sum(S, axis=0)
            log_power = librosa.power_to_db(power)
            features.append(log_power[0])
        except:
            features.append(np.nan)

    if "zcr" in selected_features:
        try:
            features.append(librosa.feature.zero_crossing_rate(y=frame)[0][0])
        except:
            features.append(np.nan)

    if "pitch" in selected_features:
        try:
            features.append(yin(y=frame, sr=sr, fmin=50, fmax=8000)[0])
        except:
            features.append(np.nan)

    if "entropy" in selected_features:
        try:
            S = np.abs(librosa.stft(frame, n_fft=2048))**2
            p = S / np.sum(S, axis=0, keepdims=True)
            entropy = -np.sum(p * np.log2(p + 1e-10), axis=0)
            features.append(entropy[0])
        except:
            features.append(np.nan)

    if "gfcc" in selected_features:
        try:
            safe_append(features, gfcc(frame, fs=sr, num_ceps=N_COEFFS)[0])
        except:
            safe_append(features, [])

    if "mfcc" in selected_features or "delta" in selected_features or "delta_delta" in selected_features:
        try:
            mf = mfcc(y=frame, sr=sr, n_mfcc=N_COEFFS)
            if mfcc_cache is not None:
                mfcc_cache.append(mf)
            safe_append(features, mf[:, 0])
        except:
            safe_append(features, [])

    if "filterbanks" in selected_features:
        try:
            mel = melspectrogram(y=frame, sr=sr, n_mels=N_COEFFS)
            logmel = librosa.power_to_db(mel)
            safe_append(features, logmel[:, 0])
        except:
            safe_append(features, [])

    if "spectral_energy" in selected_features:
        try:
            S = np.abs(librosa.stft(frame, n_fft=2048, hop_length=int(STEP_SIZE * sr)))**2
            mel_spec = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=N_COEFFS)
            log_power = librosa.power_to_db(mel_spec)
            safe_append(features, log_power[:, 0])
        except:
            safe_append(features, [])

    return features

def generate_column_names(selected_features, max_seq=14, n_coeffs=13):
    col_names = []
    for seq in range(max_seq):
        for feat in selected_features:
            if feat in ["mfcc", "gfcc", "delta", "delta_delta", "filterbanks", "spectral_energy"]:
                for c in range(n_coeffs):
                    col_names.append(f"{feat}_{c}_seq_{seq}")
            else:
                col_names.append(f"{feat}_seq_{seq}")
    return ["wav_file", "start_time", "end_time"] + col_names

# Energy-based split
def process_audio_pipeline(wav_path, filter_range=(500,5000), selected_features=None):

    if selected_features is None:
        selected_features = ['mfcc']

    # Load and preprocess
    wav_path = ensure_wav_format(wav_path)
    audio, sr = librosa.load(wav_path, sr=SR)
    filtered = bandpass_filter(audio, sr, *filter_range)
    clean    = ephraim_malah_noise_reduction(filtered, sr)

    # Energy-based splitting
    raw_intervals = librosa.effects.split(clean, top_db=TOP_DB)
    merged = []
    gap_samps = int(GAP_SILENCE * sr)
    for s, e in raw_intervals:
        if not merged or s - merged[-1][1] > gap_samps:
            merged.append([s, e])
        else:
            merged[-1][1] = e
    final_segs = [(s, e) for s, e in merged if (e - s) / sr >= MIN_DUR]

    # Frame parameters
    frame_len = int(WINDOW_SIZE * sr)
    hop_len   = int(STEP_SIZE * sr)

    rows = []
    # Extract fixed-length sequence per segment
    for seg_idx, (s_start, s_end) in enumerate(final_segs, 1):
        segment = clean[s_start:s_end]
        feats = []
        for offset in range(0, len(segment) - frame_len + 1, hop_len):
            frame = segment[offset:offset + frame_len]
            feats.append(extract_frame_features(frame, sr, selected_features))
        feats = np.array(feats)
        # pad or truncate to MAX_SEQ frames
        if feats.shape[0] < MAX_SEQ:
            pad = np.full((MAX_SEQ - feats.shape[0], feats.shape[1]), np.nan)
            feats = np.vstack([feats, pad])
        else:
            feats = feats[:MAX_SEQ]
        # flatten
        flat = feats.flatten()
        # prepare row with segment metadata
        rows.append([
            os.path.basename(wav_path),
            seg_idx,
            s_start / sr,
            s_end   / sr,
            *flat.tolist()
        ])

    # Construct DataFrame
    feat_cols = []
    for seq in range(MAX_SEQ):
        for feat in selected_features:
            if feat == 'mfcc':
                for c in range(N_COEFFS):
                    feat_cols.append(f"mfcc_{c}_seq_{seq}")
            else:
                feat_cols.append(f"{feat}_seq_{seq}")
    cols = ['wav_file','segment','start_time','end_time'] + feat_cols
    return pd.DataFrame(rows, columns=cols)


