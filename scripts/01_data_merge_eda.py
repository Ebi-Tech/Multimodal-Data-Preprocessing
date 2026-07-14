"""
01_data_merge_eda.py
=====================
Phase 1 of the multimodal pipeline.

What this script does
---------------------
1. Loads the two PROVIDED source datasets from data/ (xlsx or csv):
       - customer_social_profiles   (customer_id_new, social_media_platform,
                                      engagement_score, purchase_interest_score,
                                      review_sentiment)
       - customer_transactions      (customer_id_legacy, transaction_id,
                                      purchase_amount, purchase_date,
                                      product_category, customer_rating)
   No synthetic data is generated. If the files are missing the script stops
   with instructions.
2. Cleans each dataset (drops duplicate rows, fills missing ratings).
3. Aggregates BOTH long tables to one row per customer.
4. Reconciles the two id systems and INNER-joins them:
       social `customer_id_new` = "A" + <legacy id>, e.g. A151 <-> 151.
   The numeric part is the shared key. The join is inner because the
   recommendation target (preferred product_category) only exists for customers
   who transacted, and features only exist for customers with a social profile.
5. Engineers features and performs EDA (variable types, summary statistics,
   >=3 labelled plots: distribution, outliers, correlation, target balance).
6. Writes the final modelling table to data/merged_dataset.csv with a
   `product_category` target used by the recommendation model.

Run:  python scripts/01_data_merge_eda.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_DIR, PLOTS_DIR, MERGED_CSV

SENTIMENT_SCORE = {"Negative": -1, "Neutral": 0, "Positive": 1}


# --------------------------------------------------------------------------
# 1. LOAD PROVIDED SOURCE DATA (xlsx or csv)
# --------------------------------------------------------------------------
def _resolve(stem):
    for ext in (".xlsx", ".xls", ".csv"):
        p = DATA_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _read(path):
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_sources():
    soc_p = _resolve("customer_social_profiles")
    tx_p = _resolve("customer_transactions")
    if soc_p is None or tx_p is None:
        sys.exit(
            "ERROR: provided source dataset(s) not found in data/.\n"
            "Expected (xlsx or csv):\n"
            "  data/customer_social_profiles.(xlsx|csv)\n"
            "  data/customer_transactions.(xlsx|csv)\n"
            "then re-run:  python scripts/01_data_merge_eda.py")
    social, tx = _read(soc_p), _read(tx_p)
    print(f"[load] social profiles={social.shape} ({soc_p.name})  "
          f"transactions={tx.shape} ({tx_p.name})")
    return social, tx


# --------------------------------------------------------------------------
# 2. CLEAN + AGGREGATE SOCIAL PROFILES  ->  one row per customer
# --------------------------------------------------------------------------
def prep_social(df):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()                       # remove exact duplicate rows
    # shared numeric key from "A###"
    df["customer_key"] = (df["customer_id_new"].astype(str)
                          .str.extract(r"(\d+)")[0].astype("Int64"))
    df = df.dropna(subset=["customer_key"])
    df["customer_key"] = df["customer_key"].astype(int)
    df["sentiment_score"] = df["review_sentiment"].map(SENTIMENT_SCORE)
    print(f"[social] dropped {before - len(df)} duplicate rows; "
          f"{df['customer_key'].nunique()} unique customers over {len(df)} rows")

    def _mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) else np.nan

    g = df.groupby("customer_key")
    agg = pd.DataFrame({
        "engagement_score": g["engagement_score"].mean(),
        "purchase_interest_score": g["purchase_interest_score"].mean(),
        "sentiment_score": g["sentiment_score"].mean(),
        "n_platforms": g["social_media_platform"].nunique(),
        "n_social_rows": g.size(),
        "dominant_platform": g["social_media_platform"].agg(_mode),
        "dominant_sentiment": g["review_sentiment"].agg(_mode),
    }).reset_index()
    return agg


# --------------------------------------------------------------------------
# 3. CLEAN + AGGREGATE TRANSACTIONS  ->  one row per customer (+ target)
# --------------------------------------------------------------------------
def prep_transactions(df):
    df = df.copy()
    before = len(df)
    subset = "transaction_id" if "transaction_id" in df.columns else None
    df = df.drop_duplicates(subset=subset)
    df["customer_key"] = df["customer_id_legacy"].astype(int)
    if "customer_rating" in df.columns:
        n_null = int(df["customer_rating"].isna().sum())
        df["customer_rating"] = df["customer_rating"].fillna(
            df["customer_rating"].median())
    else:
        n_null = 0
    if "purchase_date" in df.columns:
        df["purchase_date"] = pd.to_datetime(df["purchase_date"],
                                             errors="coerce")
    print(f"[tx] dropped {before - len(df)} duplicate rows; "
          f"filled {n_null} missing ratings; "
          f"{df['customer_key'].nunique()} unique customers over {len(df)} tx")

    g = df.groupby("customer_key")
    agg = pd.DataFrame({
        "n_transactions": g.size(),
        "total_spend": g["purchase_amount"].sum(),
        "avg_amount": g["purchase_amount"].mean(),
        "max_amount": g["purchase_amount"].max(),
        "avg_rating": g["customer_rating"].mean() if "customer_rating"
        in df.columns else np.nan,
        "n_categories": g["product_category"].nunique(),
    })
    # Target: the category the customer buys most often (their preferred product)
    fav = (df.groupby(["customer_key", "product_category"]).size()
             .reset_index(name="cnt")
             .sort_values(["customer_key", "cnt"], ascending=[True, False])
             .drop_duplicates("customer_key"))
    agg["product_category"] = fav.set_index("customer_key")["product_category"]
    return agg.reset_index()


# --------------------------------------------------------------------------
# 4. MERGE + FEATURE ENGINEERING
# --------------------------------------------------------------------------
def merge_datasets(social_agg, tx_agg):
    """
    INNER join on the reconciled numeric customer key.

    Justification: `customer_id_new` ("A151") in the social table and
    `customer_id_legacy` (151) in the transactions table are two encodings of
    the same customer id from an id-system migration; the numeric part is the
    shared key. We inner-join because the supervised target only exists for
    customers who transacted, and predictive features only exist for customers
    with a social profile — so only customers present in BOTH are usable. Row
    counts are logged and post-merge assertions verify the key is unique and no
    target label is missing.
    """
    merged = social_agg.merge(tx_agg, on="customer_key", how="inner")
    print(f"[merge] social={len(social_agg)} tx={len(tx_agg)} "
          f"-> merged={len(merged)} (inner join on customer_key)")
    assert merged["customer_key"].is_unique, "customer_key not unique post-merge!"
    assert merged["product_category"].notna().all(), "missing target labels!"
    return merged


def engineer_features(df):
    df = df.copy()
    df["spend_per_txn"] = df["total_spend"] / df["n_transactions"].clip(lower=1)
    df["interest_x_engagement"] = (df["purchase_interest_score"]
                                   * df["engagement_score"])
    df["is_high_value"] = (df["total_spend"] >
                           df["total_spend"].median()).astype(int)
    # alias so downstream code that expects a customer id column keeps working
    df["customer_id"] = df["customer_key"]
    return df


# --------------------------------------------------------------------------
# 5. EDA
# --------------------------------------------------------------------------
def run_eda(df):
    print("\n===== EDA: variable types =====")
    print(df.dtypes)
    print("\n===== EDA: summary statistics (numeric) =====")
    print(df.describe().T)
    print("\n===== EDA: summary statistics (categorical) =====")
    try:
        print(df.describe(include="object").T)
    except ValueError:
        print("(no categorical columns)")
    print("\n===== EDA: target balance =====")
    print(df["product_category"].value_counts())

    candidate_num = ["engagement_score", "purchase_interest_score",
                     "sentiment_score", "n_platforms", "n_transactions",
                     "total_spend", "avg_amount", "max_amount", "avg_rating",
                     "n_categories", "spend_per_txn", "interest_x_engagement"]
    num_cols = [c for c in candidate_num if c in df.columns]

    # Plot 1: distribution of total spend
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["total_spend"].dropna(), bins=25, color="#4C72B0",
            edgecolor="white")
    ax.set_title("Distribution of Total Spend per Customer")
    ax.set_xlabel("Total spend"); ax.set_ylabel("Number of customers")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "eda_dist_total_spend.png",
                                    dpi=120)
    plt.close(fig)

    # Plot 2: outliers via standardized boxplots
    fig, ax = plt.subplots(figsize=(10, 4))
    z = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std(ddof=0)
    ax.boxplot([z[c].dropna() for c in num_cols], tick_labels=num_cols,
               showfliers=True)
    ax.set_title("Outlier Check (standardized numeric features)")
    ax.set_ylabel("z-score"); plt.xticks(rotation=40, ha="right")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "eda_outliers_box.png", dpi=120)
    plt.close(fig)

    # Plot 3: correlation heatmap
    corr = df[num_cols + ["is_high_value"]].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    ax.set_title("Correlation Matrix of Numeric Features")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "eda_correlation.png", dpi=120)
    plt.close(fig)

    # Plot 4: target balance
    fig, ax = plt.subplots(figsize=(8, 4))
    df["product_category"].value_counts().plot(kind="bar", ax=ax,
                                               color="#55A868")
    ax.set_title("Product Category Balance (recommendation target)")
    ax.set_xlabel("Product category"); ax.set_ylabel("Customers")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "eda_target_balance.png",
                                    dpi=120)
    plt.close(fig)
    print(f"\n[EDA] saved 4 plots to {PLOTS_DIR}")


# --------------------------------------------------------------------------
def main():
    print("== Loading provided source datasets ==")
    social_raw, tx_raw = load_sources()

    print("\n== Cleaning + aggregating per customer ==")
    social_agg = prep_social(social_raw)
    tx_agg = prep_transactions(tx_raw)

    print("\n== Merge + feature engineering ==")
    merged = merge_datasets(social_agg, tx_agg)
    merged = engineer_features(merged)
    merged.to_csv(MERGED_CSV, index=False)
    print(f"[save] merged_dataset.csv shape={merged.shape}")

    run_eda(merged)
    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
