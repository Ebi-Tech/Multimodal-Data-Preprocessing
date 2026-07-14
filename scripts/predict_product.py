"""
predict_product.py
==================
The single importable entry point Task 1 exposes for the rest of the team's
command-line app. It loads the RandomForest product-recommendation model trained
in ``task1_data_merge.ipynb`` (saved to ``models/product_recommender.pkl``) and
turns a dict of customer features into a predicted ``product_category``.

Usage
-----
    from scripts.predict_product import predict_product
    predict_product({"total_spend": 1200, "n_transactions": 4,
                     "engagement_score": 70, "dominant_platform": "Instagram"})
    # -> "Electronics"

No retraining happens on import: the pickled pipeline (preprocessing plus model) is
loaded once and reused.
"""
from pathlib import Path
import pandas as pd
from joblib import load

_BUNDLE_PATH = Path(__file__).resolve().parents[1] / "models" / "product_recommender.pkl"

_bundle = None


def _get_bundle():
    """Lazy-load the model bundle so importing this module is cheap and side-effect free."""
    global _bundle
    if _bundle is None:
        if not _BUNDLE_PATH.exists():
            raise FileNotFoundError(
                f"{_BUNDLE_PATH} not found. Run task1_data_merge.ipynb first to train "
                "and save the product-recommendation model.")
        _bundle = load(_BUNDLE_PATH)
    return _bundle


def predict_product(features: dict) -> str:
    """Predict a customer's preferred ``product_category`` from a feature dict.

    Only the features the model was trained on are used; any missing feature
    defaults to 0 (numeric) or "" (categorical), so partial dicts are accepted.
    """
    b = _get_bundle()
    row = {c: features.get(c, 0) for c in b["num"]}
    row.update({c: features.get(c, "") for c in b["cat"]})
    return str(b["pipeline"].predict(pd.DataFrame([row]))[0])


def predict_product_proba(features: dict) -> dict:
    """Return the class-probability distribution as {product_category: probability}."""
    b = _get_bundle()
    row = {c: features.get(c, 0) for c in b["num"]}
    row.update({c: features.get(c, "") for c in b["cat"]})
    proba = b["pipeline"].predict_proba(pd.DataFrame([row]))[0]
    return {cls: round(float(p), 4) for cls, p in zip(b["classes"], proba)}


if __name__ == "__main__":
    # tiny self-check against a representative high-spend electronics-style customer
    demo = {"engagement_score": 72, "purchase_interest_score": 65, "sentiment_score": 1,
            "n_platforms": 2, "n_social_rows": 3, "n_transactions": 4, "total_spend": 1400,
            "avg_amount": 350, "max_amount": 600, "avg_rating": 4.1, "n_categories": 3,
            "spend_per_txn": 350, "interest_x_engagement": 4680, "is_high_value": 1,
            "dominant_platform": "Instagram", "dominant_sentiment": "Positive"}
    print("predicted:", predict_product(demo))
    print("proba    :", predict_product_proba(demo))
