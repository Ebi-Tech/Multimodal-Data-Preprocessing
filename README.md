# Multimodal Data Preprocessing (Formative 2)

Group project: user identity and product recommendation system using facial recognition, voice verification, and a product recommendation model.

**Team:** David, Bakhit, Divine, Serein

## Project structure
task1_data_merge.ipynb       — Task 1: data merge, EDA, product recommendation model
scripts/config.py            — shared paths and constants
scripts/01_data_merge_eda.py — standalone script version of the merge pipeline
scripts/predict_product.py   — importable predict_product() function for the CLI
data/                        — source datasets and merged output
models/                      — trained model pickle and metrics
plots/                       — EDA and evaluation plots
images/                      — facial images (Tasks 2-3, in progress)
audio/                       — voice recordings (Tasks 2-3, in progress)

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl joblib
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
