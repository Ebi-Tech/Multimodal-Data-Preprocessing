"""
02_image_pipeline.py
=====================
Task 2 (David): image data collection, augmentation, feature extraction,
and face-recognition model training.

What this script does
----------------------
1. Loads real team photos from images/<member>/<expression>.<ext>
   (expressions = neutral, smiling, surprised — see config.EXPRESSIONS).
   No synthetic faces are generated; this only works once every member has
   dropped their own 3 real photos in place. Missing photos are reported,
   not silently skipped.
2. Displays a grid of every member's photos (saved to plots/image_grid.png).
3. Applies 4 augmentations per real photo — rotation, horizontal flip,
   grayscale, gaussian blur — to build up a larger training set (>=2 per
   image required by the rubric).
4. Detects + crops the face in every photo/augmented variant (OpenCV Haar
   cascade, falls back to the full frame if no face is found), then
   extracts a feature vector per image: a per-channel color histogram +
   a HOG descriptor. All feature rows are written to
   data/image_features.csv.
5. Trains a face-recognition classifier (RandomForest) to predict WHICH
   team member a photo belongs to, evaluates it (accuracy, macro-F1,
   confusion matrix), and saves the fitted bundle to
   models/face_recognition_model.pkl.
6. Runs a quick predict_face() smoke test, including an "unauthorized"
   check if images/unauthorized/*.jpg is present.

Run:  python scripts/02_image_pipeline.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump

try:
    from scripts.config import (IMAGES_DIR, DATA_DIR, MODELS_DIR, PLOTS_DIR,
                                 RANDOM_SEED, TEAM_MEMBERS, EXPRESSIONS,
                                 FACE_CROP, FACE_MATCH_THRESHOLD)
    from scripts.face_features import extract_features, augment
except ImportError:
    from config import (IMAGES_DIR, DATA_DIR, MODELS_DIR, PLOTS_DIR,
                         RANDOM_SEED, TEAM_MEMBERS, EXPRESSIONS,
                         FACE_CROP, FACE_MATCH_THRESHOLD)
    from face_features import extract_features, augment

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------
# 1. LOAD REAL PHOTOS
# --------------------------------------------------------------------------
def _find_photo(member_dir, expression):
    for ext in IMG_EXTS:
        p = member_dir / f"{expression}{ext}"
        if p.exists():
            return p
    return None


def load_photos():
    """Returns {member: {expression: path}}; prints anything missing."""
    photos = {}
    missing = []
    for member in TEAM_MEMBERS:
        member_dir = IMAGES_DIR / member
        photos[member] = {}
        for expr in EXPRESSIONS:
            p = _find_photo(member_dir, expr)
            if p:
                photos[member][expr] = p
            else:
                missing.append(f"images/{member}/{expr}.<jpg|png|...>")
    n_found = sum(len(v) for v in photos.values())
    n_required = len(TEAM_MEMBERS) * len(EXPRESSIONS)
    print(f"[load] found {n_found}/{n_required} required photos")
    if missing:
        print("[load] MISSING (add these before the model will be complete):")
        for m in missing:
            print(f"    {m}")
    return photos


# --------------------------------------------------------------------------
# 2. DISPLAY GRID
# --------------------------------------------------------------------------
def display_grid(photos):
    n_rows, n_cols = len(TEAM_MEMBERS), len(EXPRESSIONS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    for i, member in enumerate(TEAM_MEMBERS):
        for j, expr in enumerate(EXPRESSIONS):
            ax = axes[i, j]
            path = photos.get(member, {}).get(expr)
            if path is not None:
                img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        color="red", fontsize=9)
            ax.set_title(f"{member} — {expr}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    out = PLOTS_DIR / "image_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[display] saved photo grid -> {out}")


# --------------------------------------------------------------------------
# 3-4. AUGMENT + EXTRACT FEATURES -> data/image_features.csv
# --------------------------------------------------------------------------
def build_feature_table(photos):
    rows = []
    for member, exprs in photos.items():
        for expr, path in exprs.items():
            img = cv2.imread(str(path))
            if img is None:
                print(f"[warn] could not read {path}, skipping")
                continue
            variants = {"original": img, **augment(img)}
            for aug_name, vimg in variants.items():
                feat = extract_features(vimg, face_crop=FACE_CROP)
                rows.append({
                    "member": member,
                    "expression": expr,
                    "augmentation": aug_name,
                    "source_path": str(path),
                    **{f"f{i}": v for i, v in enumerate(feat)},
                })
    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "image_features.csv"
    df.to_csv(out_path, index=False)
    print(f"[features] {len(df)} rows, {df['member'].nunique() if len(df) else 0} "
          f"members -> {out_path}")
    return df


# --------------------------------------------------------------------------
# 5. TRAIN + EVALUATE FACE RECOGNITION MODEL
# --------------------------------------------------------------------------
def train_face_model(df):
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values
    y = df["member"].values

    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)

    # small dataset -> stratified split still works since every member has
    # 15 rows (3 photos x 5 variants incl. original)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=RANDOM_SEED, stratify=y_enc)

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n[eval] face recognition accuracy={acc:.3f}  macro-F1={f1:.3f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_,
                                 zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha="right")
    ax.set_yticks(range(len(le.classes_)))
    ax.set_yticklabels(le.classes_)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title("Face Recognition — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = PLOTS_DIR / "face_confusion_matrix.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[eval] saved confusion matrix -> {out}")

    bundle = {
        "model": clf,
        "label_encoder": le,
        "feat_cols": feat_cols,
        "face_crop": FACE_CROP,
        "threshold": FACE_MATCH_THRESHOLD,
        "accuracy": acc,
        "f1_macro": f1,
    }
    out_model = MODELS_DIR / "face_recognition_model.pkl"
    dump(bundle, out_model)
    print(f"[save] {out_model} (unauthorized threshold={FACE_MATCH_THRESHOLD})")
    return bundle


# --------------------------------------------------------------------------
# 6. SMOKE TEST: predict_face() on a couple of real / unauthorized photos
# --------------------------------------------------------------------------
def smoke_test(photos):
    try:
        from scripts.predict_face import predict_face
    except ImportError:
        from predict_face import predict_face

    print("\n== predict_face() smoke test ==")
    for member, exprs in photos.items():
        for expr, path in list(exprs.items())[:1]:
            print(f"  {path}  ->  {predict_face(path)}  (expected: {member})")

    unauth_dir = IMAGES_DIR / "unauthorized"
    unauth_photos = []
    if unauth_dir.exists():
        for ext in IMG_EXTS:
            unauth_photos.extend(unauth_dir.glob(f"*{ext}"))
    if unauth_photos:
        for p in unauth_photos:
            print(f"  {p}  ->  {predict_face(p)}  (expected: unauthorized)")
    else:
        print(f"  (no unauthorized sample found in {unauth_dir}; add a photo "
              f"of someone NOT on the team there to test the deny path)")


# --------------------------------------------------------------------------
def main():
    print("== Loading real team photos ==")
    photos = load_photos()
    if sum(len(v) for v in photos.values()) == 0:
        sys.exit(
            "\nERROR: no photos found.\n"
            "Add real photos first: images/<Member>/<neutral|smiling|surprised>.<jpg|png>\n"
            "e.g. images/David/neutral.jpg, images/David/smiling.jpg, images/David/surprised.jpg\n"
            "then re-run: python scripts/02_image_pipeline.py")

    print("\n== Displaying photo grid ==")
    display_grid(photos)

    print("\n== Augmenting + extracting features ==")
    df = build_feature_table(photos)
    if df.empty or df["member"].nunique() < 2:
        sys.exit("\nERROR: need photos from at least 2 members to train a "
                  "classifier. Add more photos and re-run.")

    print("\n== Training face recognition model ==")
    train_face_model(df)

    smoke_test(photos)

    print("\nTask 2 (image pipeline) complete.")


if __name__ == "__main__":
    main()
