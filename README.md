# Multimodal Data Preprocessing (Formative 2)

Group project: user identity and product recommendation system using facial recognition, voice verification, and a product recommendation model.

**Team:** Serein, David Akintoya, Divine Ifechukwude, Bakhit Mahamat

## Project structure
notebooks/
task1_data_merge.ipynb         Task 1: data merge, EDA, product recommendation model (Serein)
B_image_face_model.ipynb       Task 2: image pipeline, facial recognition model (David)
C_audio_voice_model.ipynb      Task 3: audio pipeline, voiceprint verification model (Divine)
D_system_simulation.ipynb      Task 6: full system simulation, authorized + unauthorized demos (Bakhit)
scripts/
config.py                      Shared paths and constants
01_data_merge_eda.py           Standalone script version of the merge pipeline
predict_product.py             Importable predict_product() for the CLI
predict_face.py                Importable predict_face() for the CLI
face_features.py               Original face feature helpers (reference)
verify_voice.py                Importable verify_voice() for the CLI
system_demo.py                 CLI demo chaining all three models
data/
customer_social_profiles.xlsx  Source dataset
customer_transactions.xlsx     Source dataset
merged_dataset.csv             Merged output from Task 1
image_features.csv             Extracted image features (44 rows, 1860 features)
models/
product_recommender.pkl        Trained product recommendation model
metrics.json                   Product model evaluation metrics
face_recognition_model.pkl     Trained facial recognition model
face_metrics.json              Face model evaluation metrics
voice_model.pkl                Trained voiceprint verification model
voice_metrics.json             Voice model evaluation metrics
plots/                           EDA and evaluation plots
audio/                           Team voice recordings (14 clips + 2 stranger)
audio_features.csv               Extracted audio features (56 samples x 28 features)
images/                          Team facial photos (12 photos + 1 unauthorized)
report/                          Final project report (docx/pdf)

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn==1.6.1 openpyxl joblib librosa opencv-python scikit-image
```

## How to run

**Task 1 notebook (data merge + product model):**
```bash
jupyter notebook notebooks/task1_data_merge.ipynb
```

**Task 2 notebook (image pipeline + face model):**
```bash
jupyter notebook notebooks/B_image_face_model.ipynb
```

**Task 3 notebook (audio pipeline + voice model):**
```bash
jupyter notebook notebooks/C_audio_voice_model.ipynb
```

**Task 6 notebook (system simulation):**
```bash
jupyter notebook notebooks/D_system_simulation.ipynb
```

**Task 1 script (standalone):**
```bash
python scripts/01_data_merge_eda.py
```

**Full system demo (CLI):**
```bash
python scripts/system_demo.py
```

**Individual model imports:**
```python
from scripts.predict_product import predict_product
from scripts.predict_face import predict_face
from scripts.verify_voice import verify_voice
```
