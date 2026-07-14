"""
config.py — shared configuration for the multimodal pipeline.

Edit TEAM_MEMBERS to your real group members. Everything else (paths,
seeds, product classes) is derived from here so the whole pipeline stays
consistent across scripts.
"""
from pathlib import Path

# ---- Team ---------------------------------------------------------------
# Replace these with your real names. The image/audio pipelines create one
# identity per member so the face/voice models have per-person classes.
TEAM_MEMBERS = ["David", "Bakhit", "Divine", "Serein"]

# ---- Reproducibility ----------------------------------------------------
RANDOM_SEED = 42

# ---- Paths --------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent      # project root
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"
AUDIO_DIR = ROOT / "audio"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"

# Provided source datasets (drop the two files the assignment gives you here).
SOCIAL_PROFILES_CSV = DATA_DIR / "customer_social_profiles.csv"
TRANSACTIONS_CSV = DATA_DIR / "customer_transactions.csv"
MERGED_CSV = DATA_DIR / "merged_dataset.csv"

for _d in (DATA_DIR, IMAGES_DIR, AUDIO_DIR, MODELS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Product recommendation classes ------------------------------------
PRODUCT_CLASSES = [
    "electronics", "fashion", "home", "beauty", "sports", "books",
]

# ---- Facial expressions required by the assignment ----------------------
EXPRESSIONS = ["neutral", "smiling", "surprised"]

# ---- Image settings (REAL photos only) ----------------------------------
# Drop your photos in images/<member>/<expression>.<ext>. The image pipeline
# uses these real photos; it does NOT synthesize faces.
REAL_IMAGES_DIR = IMAGES_DIR                 # member folders live directly here
SAMPLES_DIR = IMAGES_DIR / "samples"         # cropped/augmented samples go here
STRANGER_IMAGES_DIR = IMAGES_DIR / "stranger"  # non-member photo(s) for the
                                               # unauthorized-attempt demo
FACE_CROP = True            # auto-detect & crop the face before feature extraction
REAL_AUG_PER_PHOTO = 8      # augmented variants created per real photo (for training)

# ---- Audio phrases ------------------------------------------------------
AUDIO_PHRASES = ["yes_approve", "confirm_transaction"]

# ---- Audio settings (REAL recordings only) ------------------------------
# Drop your recordings in audio/<member>/<phrase>.<ext> (same convention as
# images/<member>/<expression>.<ext>). The audio pipeline uses these real
# clips; it does NOT synthesize voices.
REAL_AUDIO_DIR = AUDIO_DIR                    # member folders live directly here
AUDIO_SAMPLES_DIR = AUDIO_DIR / "samples"    # cropped/augmented clips go here
STRANGER_AUDIO_DIR = AUDIO_DIR / "stranger"  # non-member clip(s) for the
                                             # unauthorized-attempt demo
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac")
REAL_AUG_PER_CLIP = 8       # augmented variants created per real clip (for training)

# ---- Authorized set (who the system is allowed to admit) ----------------
# Everyone on the team is authorized; the CLI demo also constructs an
# "unknown" intruder that is NOT in this list.
AUTHORIZED_USERS = list(TEAM_MEMBERS)
