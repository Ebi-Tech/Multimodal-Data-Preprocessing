"""
system_demo.py
==============
CLI system simulation for the Multimodal Data Preprocessing project.

Chains three models in sequence:
    1. Face recognition gate
    2. Product recommendation
    3. Voice verification gate

Usage:
    python scripts/system_demo.py                              (run full demo)
    python scripts/system_demo.py <face_image> <voice_audio>   (single test)
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.predict_face import predict_face
from scripts.predict_product import predict_product
from scripts.verify_voice import verify_voice


def run_transaction(face_path, voice_path, label="TRANSACTION"):
    print("=" * 60)
    print(label)
    print("=" * 60)

    print("\nSTEP 1: Face recognition gate")
    print(f"  input: {face_path}")
    face_result, face_conf = predict_face(face_path)
    print(f"  result: {face_result} (confidence: {face_conf:.3f})")

    if face_result == "unauthorized":
        print("\n  ACCESS DENIED: face not recognized.")
        print("=" * 60)
        return

    user = face_result
    print(f"  ACCESS GRANTED: welcome, {user}.")

    print("\nSTEP 2: Product recommendation")
    sample_features = {
        "engagement_score": 75.0,
        "purchase_interest_score": 3.5,
        "sentiment_score": 0.5,
        "n_platforms": 2,
        "n_transactions": 10,
        "total_spend": 500.0,
        "avg_amount": 50.0,
        "max_amount": 150.0,
        "avg_rating": 4.0,
        "n_categories": 3,
        "spend_per_txn": 50.0,
        "interest_x_engagement": 262.5,
        "is_high_value": 1,
    }
    product = predict_product(sample_features)
    print(f"  predicted product for {user}: {product}")

    print("\nSTEP 3: Voice verification gate")
    print(f"  input: {voice_path}")
    voice_result, voice_conf = verify_voice(voice_path)
    print(f"  result: {voice_result} (confidence: {voice_conf:.3f})")

    if voice_result == "unauthorized":
        print("\n  ACCESS DENIED: voice not verified.")
        print("=" * 60)
        return

    print(f"\n  VOICE VERIFIED: {voice_result}")
    print("\n" + "=" * 60)
    print("TRANSACTION APPROVED")
    print(f"  user: {user}")
    print(f"  recommended product: {product}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_transaction(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        print("\n>>> Running full system demonstration\n")

        run_transaction(
            "images/Bakhit/neutral.jpg",
            "audio/Bakhit2.wav",
            label="SIMULATION 1: Authorized Transaction (Bakhit)"
        )

        print("\n")

        run_transaction(
            "images/unauthorized/attempt1.jpg",
            "audio/Bakhit2.wav",
            label="SIMULATION 2: Unauthorized Face Attempt"
        )

        print("\n")

        run_transaction(
            "images/Bakhit/neutral.jpg",
            "audio/stranger1.wav",
            label="SIMULATION 3: Unauthorized Voice Attempt"
        )
    else:
        print("Usage:")
        print("  python scripts/system_demo.py                              (run full demo)")
        print("  python scripts/system_demo.py <face_image> <voice_audio>   (single test)")
        sys.exit(1)
