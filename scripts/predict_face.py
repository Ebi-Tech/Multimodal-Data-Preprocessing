"""
predict_face.py
===============
Importable facial recognition for the CLI app.

Loads the trained face model from models/face_recognition_model.pkl
and exposes a single function:

    from scripts.predict_face import predict_face
    label, confidence = predict_face("images/David/neutral.jpg")

Returns the predicted team member name if confidence exceeds the
threshold, otherwise returns "unauthorized".
"""
import cv2
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from skimage.feature import hog

_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _ROOT / "models" / "face_recognition_model.pkl"
_FACE_SIZE = (128, 128)
_THRESHOLD = 0.55

_bundle = None


def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(_MODEL_PATH)
    return _bundle


def _detect_and_crop(img_rgb):
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        face = img_rgb
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = img_rgb[y:y + h, x:x + w]
    return cv2.resize(face, _FACE_SIZE)


def _extract_features(img_rgb):
    hist_parts = []
    for ch in range(3):
        h = cv2.calcHist([img_rgb], [ch], None, [32], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist_parts.append(h)
    hist_feat = np.concatenate(hist_parts)

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hog_feat = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), feature_vector=True
    )
    return np.concatenate([hist_feat, hog_feat]).astype(np.float32)


def predict_face(image_path, threshold=_THRESHOLD):
    """
    Identify a person from a facial image.

    Returns (predicted_member, confidence) if confidence > threshold,
    otherwise ("unauthorized", confidence).
    """
    bundle = _get_bundle()
    model = bundle["model"]
    feature_cols = bundle["features"]

    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cropped = _detect_and_crop(img_rgb)
    feat = _extract_features(cropped)
    x_row = pd.DataFrame([feat], columns=feature_cols)

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
        print("Usage: python scripts/predict_face.py <path_to_image>")
        sys.exit(1)
    label, conf = predict_face(sys.argv[1])
    print(f"predicted={label}, confidence={conf:.3f}")
