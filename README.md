# Multimodal Data Preprocessing (Formative 2)

Group project: user identity and product recommendation system using facial recognition, voice verification, and a product recommendation model.

**Team:** David, Bakhit, Divine, Serein

## Project structure
notebooks/
task1_data_merge.ipynb         Task 1: data merge, EDA, product recommendation model
C_audio_voice_model.ipynb      Task 3: audio processing, voiceprint verification model
scripts/
config.py                      Shared paths and constants
01_data_merge_eda.py           Standalone script version of the merge pipeline
predict_product.py             Importable predict_product() for the CLI
verify_voice.py                Importable verify_voice() for the CLI
data/
customer_social_profiles.xlsx  Source dataset
customer_transactions.xlsx     Source dataset
merged_dataset.csv             Merged output from Task 1
models/
product_recommender.pkl        Trained product recommendation model
metrics.json                   Product model evaluation metrics
voice_model.pkl                Trained voiceprint verification model
voice_metrics.json             Voice model evaluation metrics
plots/                           EDA and evaluation plots
audio/                           Team voice recordings (14 clips + 2 stranger)
audio_features.csv               Extracted audio features (56 samples x 28 features)
images/                          Facial images (Task 2, in progress)

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl joblib librosa
```

## How to run

**Task 1 notebook (data merge + product model):**
```bash
jupyter notebook notebooks/task1_data_merge.ipynb
```

**Task 3 notebook (audio pipeline + voice model):**
```bash
jupyter notebook notebooks/C_audio_voice_model.ipynb
```

**Task 1 script (standalone):**
```bash
python scripts/01_data_merge_eda.py
```

**Product prediction (import for CLI):**
```python
from scripts.predict_product import predict_product
result = predict_product({"engagement_score": 75, "purchase_interest_score": 3.0, ...})
```

**Voice verification (import for CLI):**
```python
from scripts.verify_voice import verify_voice
label, confidence = verify_voice("audio/Bakhit1.wav")
```
