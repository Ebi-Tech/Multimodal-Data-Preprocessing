"""
system_demo.py
==============
CLI system simulation for the Multimodal Data Preprocessing project.

Chains three models in sequence:
    1. Face recognition gate
    2. Product recommendation
    3. Voice verification gate

Usage:
    python scripts/system_demo.py <face_image_path> <voice_audio_path>

Example (authorized):
    python scripts/system_demo.py images/Bakhit/neutral.jpg audio/Bakhit1.wav

Example (unauthorized face):
    python scripts/system_demo.py images/unauthorized/attempt1.jpg audio/Bakhit1.wav

Example (unauthorized voice):
    python scripts/system_demo.py images/Bakhit/neutral.jpg audio/stranger1.wav
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.predict_face import predict_face
from scripts.predict_product import predict_product
from scripts.verify_voice import verify_voice


def run_transaction(face_path, voice_path):
    print("=" * 50)
    print("MULTIMODAL AUTHENTICATION SYSTEM")
    print("=" * 50)

    # Step 1: Face recognition gate
    print("\n[STEP 1] Face Recognition")
    print(f"  Input: {face_path}")
    face_result, face_conf = predict_face(face_path)
    print(f"  Result: {face_result} (confidence: {face_conf:.3f})")

    if face_result == "unauthorized":
        print("\n  ACCESS DENIED: face not recognized.")
        print("=" * 50)
        return

    user = face_result
    print(f"  ACCESS GRANTED: welcome, {user}.")

    # Step 2: Product recommendation
    print("\n[STEP 2] Product Recommendation")
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
    print(f"  Predicted product for {user}: {product}")

    # Step 3: Voice verification gate
    print("\n[STEP 3] Voice Verification")
    print(f"  Input: {voice_path}")
    voice_result, voice_conf = verify_voice(voice_path)
    print(f"  Result: {voice_result} (confidence: {voice_conf:.3f})")

    if voice_result == "unauthorized":
        print("\n  ACCESS DENIED: voice not verified.")
        print("=" * 50)
        return

    print(f"\n  VOICE VERIFIED: {voice_result}")
    print("\n" + "=" * 50)
    print(f"TRANSACTION APPROVED")
    print(f"  User: {user}")
    print(f"  Recommended product: {product}")
    print("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/system_demo.py <face_image> <voice_audio>")
        sys.exit(1)
    run_transaction(sys.argv[1], sys.argv[2])
