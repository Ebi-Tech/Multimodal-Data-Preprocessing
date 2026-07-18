# Multimodal Data Preprocessing (Formative 2)

Group project: user identity and product recommendation system using facial recognition, voice verification, and a product recommendation model.

**Team:** David, Bakhit, Divine, Serein

## Project structure
task1_data_merge.ipynb        — Task 1: data merge, EDA, product recommendation model
scripts/config.py             — shared paths and constants
scripts/01_data_merge_eda.py  — standalone script version of the merge pipeline
scripts/predict_product.py    — importable predict_product() function for the CLI
scripts/face_features.py      — shared face detect/augment/feature-extraction helpers (Task 2)
scripts/02_image_pipeline.py  — Task 2: image pipeline + face recognition model training
scripts/predict_face.py       — importable predict_face() function for the CLI
data/                         — source datasets, merged output, image_features.csv (Task 2)
models/                       — trained model pickles and metrics
plots/                        — EDA and evaluation plots
images/<Member>/<neutral|smiling|surprised>.jpg — real team photos (Task 2)
images/unauthorized/*.jpg     — optional: photo(s) of someone NOT on the team, to test the deny path
audio/                        — voice recordings (Task 3, in progress)

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn scikit-image opencv-python openpyxl joblib
```

## How to run

**Task 1 notebook:**
```bash
jupyter notebook task1_data_merge.ipynb
```

**Task 1 script (standalone):**
```bash
python scripts/01_data_merge_eda.py
```

**Product prediction (import in CLI):**
```python
from scripts.predict_product import predict_product
result = predict_product({"engagement_score": 75, "purchase_interest_score": 3.0, ...})
```

**Task 2 — image pipeline + face recognition:**

Every team member drops their own 3 real photos in `images/<Member>/`:
`neutral.jpg`, `smiling.jpg`, `surprised.jpg` (member names come from
`config.TEAM_MEMBERS`). Then:
```bash
python scripts/02_image_pipeline.py
```
This loads all members' photos, displays a grid (`plots/image_grid.png`),
applies 4 augmentations per photo (rotation, flip, grayscale, blur),
extracts color-histogram + HOG features into `data/image_features.csv`,
trains a RandomForest face-recognition model, evaluates it (accuracy,
macro-F1, `plots/face_confusion_matrix.png`), and saves it to
`models/face_recognition_model.pkl`.

**Face prediction (import in CLI):**
```python
from scripts.predict_face import predict_face
predict_face("images/David/neutral.jpg")  # -> "David" or "unauthorized"
```
