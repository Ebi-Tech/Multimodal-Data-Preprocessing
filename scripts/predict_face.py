"""
predict_face.py
================
The single importable entry point Task 2 exposes for the rest of the team's
command-line app. Loads the face-recognition model trained in
scripts/02_image_pipeline.py (models/face_recognition_model.pkl) and turns
a photo into a predicted team-member name, or "unauthorized" if the model
isn't confident it's anyone on the team.

Usage
-----
    from scripts.predict_face import predict_face
    predict_face("images/David/neutral.jpg")          # -> "David"
    predict_face("images/unauthorized/intruder.jpg")  # -> "unauthorized"

No retraining happens on import: the pickled bundle (model + label encoder +
decision threshold) is loaded once and reused.
"""
from pathlib import Path
import cv2
from joblib import load

try:
    from scripts.face_features import extract_features
except ImportError:
    from face_features import extract_features

_BUNDLE_PATH = Path(__file__).resolve().parents[1] / "models" / "face_recognition_model.pkl"
_bundle = None


def _get_bundle():
    """Lazy-load the model bundle so importing this module is cheap and
    side-effect free."""
    global _bundle
    if _bundle is None:
        if not _BUNDLE_PATH.exists():
            raise FileNotFoundError(
                f"{_BUNDLE_PATH} not found. Run scripts/02_image_pipeline.py "
                "first to train and save the face-recognition model.")
        _bundle = load(_BUNDLE_PATH)
    return _bundle


def predict_face(image_path) -> str:
    """Predict which team member a photo belongs to.

    Returns the member's name if the model's top prediction has confidence
    >= the trained threshold, otherwise returns "unauthorized" — this is
    what lets the CLI deny access to a face the model wasn't trained on.
    """
    b = _get_bundle()
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    feat = extract_features(img, face_crop=b.get("face_crop", True)).reshape(1, -1)
    proba = b["model"].predict_proba(feat)[0]
    top_idx = int(proba.argmax())
    top_prob = float(proba[top_idx])

    if top_prob < b["threshold"]:
        return "unauthorized"
    return str(b["label_encoder"].inverse_transform([top_idx])[0])


def predict_face_proba(image_path) -> dict:
    """Return the full class-probability distribution as {member: probability}."""
    b = _get_bundle()
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    feat = extract_features(img, face_crop=b.get("face_crop", True)).reshape(1, -1)
    proba = b["model"].predict_proba(feat)[0]
    return {cls: round(float(p), 4) for cls, p in zip(b["label_encoder"].classes_, proba)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/predict_face.py <image_path>")
    print("predicted:", predict_face(sys.argv[1]))
    print("proba    :", predict_face_proba(sys.argv[1]))
