"""
face_features.py
=================
Shared, dependency-light helpers for the image/face pipeline. Kept separate
from 02_image_pipeline.py (training) and predict_face.py (inference) so both
can import the exact same face-detection + feature-extraction logic without
duplicating it or fighting Python's "can't import a module that starts with
a digit" restriction.

No config.py import here on purpose — this module has to work whether it's
run as part of the `scripts` package (`from scripts.face_features import ...`)
or as a plain script sitting next to config.py, so it avoids the ambiguity
by not depending on either import style itself.
"""
import cv2
import numpy as np
from skimage.feature import hog

FACE_SIZE = (128, 128)

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detect_and_crop(img, face_crop=True):
    """Detect the largest face with a Haar cascade and crop to it.

    Falls back to the full frame (resized) if no face is detected, so a
    tricky photo (bad angle, glasses glare, etc.) never crashes the
    pipeline — it just degrades gracefully to whole-image features.
    """
    if not face_crop:
        return cv2.resize(img, FACE_SIZE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                            minSize=(60, 60))
    if len(faces) == 0:
        face = img
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
        face = img[y:y + h, x:x + w]
    return cv2.resize(face, FACE_SIZE)


def extract_features(img, face_crop=True):
    """Turn a BGR image into one flat feature vector: a per-channel color
    histogram (captures skin tone / lighting) concatenated with a HOG
    descriptor (captures facial structure/edges). This doubles as both the
    "histogram" and "embedding" feature types the assignment asks for.
    """
    face = detect_and_crop(img, face_crop=face_crop)

    hist_parts = []
    for ch in range(3):
        h = cv2.calcHist([face], [ch], None, [32], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hist_parts.append(h)
    hist_feat = np.concatenate(hist_parts)

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), feature_vector=True)

    return np.concatenate([hist_feat, hog_feat]).astype(np.float32)


def augment(img):
    """Return >=2 (here: 4) augmented variants of a BGR image: rotation,
    horizontal flip, grayscale, and gaussian blur."""
    h, w = img.shape[:2]
    out = {}

    M = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    out["rot15"] = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    out["flip"] = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out["gray"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    out["blur"] = cv2.GaussianBlur(img, (7, 7), 0)

    return out
