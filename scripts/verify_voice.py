"""
verify_voice.py
===============
Importable voiceprint verification for the CLI app.

Loads the trained voice model from models/voice_model.pkl and exposes
a single function:

    from scripts.verify_voice import verify_voice
    label, confidence = verify_voice("audio/David1.wav")

Returns the predicted team member name if confidence exceeds the
threshold, otherwise returns "unauthorized".
"""
import os
import numpy as np
import pandas as pd
import librosa
import joblib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _ROOT / "models" / "voice_model.pkl"
_SR = 22050
_THRESHOLD = 0.6

_bundle = None


def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(_MODEL_PATH)
    return _bundle


def _extract_features(y, sr):
    y, _ = librosa.effects.trim(y, top_db=20)
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    return mfcc_means, rolloff, rms


def verify_voice(audio_path, threshold=_THRESHOLD):
    """
    Verify a speaker's identity from an audio file.

    Returns (predicted_member, confidence) if confidence > threshold,
    otherwise ("unauthorized", confidence).
    """
    bundle = _get_bundle()
    model = bundle["model"]
    feature_cols = bundle["features"]

    y, sr = librosa.load(audio_path, sr=_SR)
    mfcc_means, rolloff, rms = _extract_features(y, sr)

    row = {f"mfcc_{i}": val for i, val in enumerate(mfcc_means, start=1)}
    row["spectral_rolloff"] = rolloff
    row["rms_energy"] = rms
    x_row = pd.DataFrame([row], columns=feature_cols)

    proba = model.predict_proba(x_row)[0]
    classes = model.classes_
    best_idx = np.argmax(proba)
    confidence = proba[best_idx]
    predicted = classes[best_idx]

    if confidence > threshold:
        return predicted, confidence
    return "unauthorized", confidence


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_voice.py <path_to_wav>")
        sys.exit(1)
    label, conf = verify_voice(sys.argv[1])
    print(f"predicted={label}, confidence={conf:.3f}")
