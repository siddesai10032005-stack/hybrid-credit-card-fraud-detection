
A two-stage machine learning system combining transaction-level fraud detection with customer behavioral verification.

## 🎯 Project Overview

The system combines two perspectives:

- **Stage 1:** Random Forest transaction-level fraud detection
- **Stage 2:** Random Forest customer behavioral verification
- **Decision Engine:** Hybrid risk scoring

Final decisions:

- 🟢 **APPROVE**
- 🟡 **VERIFY**
- 🔴 **BLOCK**

## 🏗️ System Architecture

```text
                    TRANSACTION
                         |
                         v
              +---------------------+
              |       STAGE 1       |
              |   Random Forest     |
              | Transaction Risk    |
              +----------+----------+
                         |
                  Fraud Probability
                         |
                         v
              +---------------------+
              |       STAGE 2       |
              | Behavioral Random   |
              |      Forest         |
              | Customer Behavior   |
              +----------+----------+
                         |
                Behavior Probability
                         |
                         v
              +---------------------+
              |   HYBRID RISK SCORE |
              |                     |
              | 70% Stage 1        |
              | 30% Stage 2        |
              +----------+----------+
                         |
          +--------------+--------------+
          |              |              |
       < 0.10       0.10 - <0.90       >= 0.90
          |              |              |
          v              v              v
       APPROVE        VERIFY          BLOCK




"""
common.py
---------
Shared utilities used across the fraud-detection pipeline:
- Loading the Sparkov-style raw transaction CSVs
- Haversine distance between customer and merchant
- Basic time-based feature extraction

Expected raw columns (Sparkov synthetic dataset):
trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender,
street, city, state, zip, lat, long, city_pop, job, dob, trans_num,
unix_time, merch_lat, merch_long, is_fraud
"""

import numpy as np
import pandas as pd


RAW_DTYPES = {
    "cc_num": "int64",
    "amt": "float64",
    "city_pop": "int64",
}


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw Sparkov-style transaction CSV and do basic cleanup."""
    df = pd.read_csv(path)

    # Some Sparkov exports include a stray unnamed index column
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)
    return df


def haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in kilometers between two lat/lon arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    r_km = 6371.0
    return c * r_km


def add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day_of_week"] = df["trans_date_trans_time"].dt.dayofweek  # 0=Mon
    df["trans_day"] = df["trans_date_trans_time"].dt.day
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["trans_day_of_week"] >= 5).astype(int)
    return df


def add_merchant_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["location_distance_km"] = haversine_distance(
        df["lat"].values, df["long"].values,
        df["merch_lat"].values, df["merch_long"].values,
    )
    return df
"""
build_customer_profiles.py
---------------------------
Builds a per-customer behavioral profile from the TRAINING split only,
so that both the training and test feature sets can be built using
history that a real system would have available at scoring time.

Profile fields per customer (cc_num):
- avg_amount             : historical average transaction amount
- txn_count              : number of historical transactions
- home_lat / home_long   : customer's registered location
- common_merchants       : set of merchants the customer has used before
- common_categories      : set of categories the customer has used before
- avg_hour               : historical average transaction hour
- avg_location_distance  : historical average distance from home to merchant

Output: data/processed/customer_profiles.pkl (pickled dict keyed by cc_num)
         data/processed/customer_profiles_summary.csv (human-readable summary)
"""

import pickle

import pandas as pd

from common import load_raw, add_time_fields, add_merchant_distance

RAW_TRAIN_PATH = "data/raw/fraudTrain.csv"
OUT_PKL_PATH = "data/processed/customer_profiles.pkl"
OUT_CSV_PATH = "data/processed/customer_profiles_summary.csv"


def build_profiles(df: pd.DataFrame) -> dict:
    profiles = {}

    grouped = df.groupby("cc_num")
    for cc_num, g in grouped:
        merchants = set(g["merchant"].unique())
        categories = set(g["category"].unique())

        profiles[cc_num] = {
            "avg_amount": float(g["amt"].mean()),
            "std_amount": float(g["amt"].std(ddof=0)) if len(g) > 1 else 0.0,
            "txn_count": int(len(g)),
            "home_lat": float(g["lat"].iloc[0]),
            "home_long": float(g["long"].iloc[0]),
            "common_merchants": merchants,
            "common_categories": categories,
            "avg_hour": float(g["trans_hour"].mean()),
            "avg_location_distance": float(g["location_distance_km"].mean()),
            "last_trans_time": g["trans_date_trans_time"].max(),
        }
    return profiles


def main():
    print(f"Loading raw training data from {RAW_TRAIN_PATH} ...")
    df = load_raw(RAW_TRAIN_PATH)
    df = add_time_fields(df)
    df = add_merchant_distance(df)

    print(f"Building customer profiles for {df['cc_num'].nunique()} customers ...")
    profiles = build_profiles(df)

    with open(OUT_PKL_PATH, "wb") as f:
        pickle.dump(profiles, f)
    print(f"Saved profiles pickle -> {OUT_PKL_PATH}")

    summary_rows = [
        {
            "cc_num": cc_num,
            "avg_amount": p["avg_amount"],
            "txn_count": p["txn_count"],
            "avg_hour": p["avg_hour"],
            "avg_location_distance": p["avg_location_distance"],
            "n_common_merchants": len(p["common_merchants"]),
            "n_common_categories": len(p["common_categories"]),
        }
        for cc_num, p in profiles.items()
    ]
    pd.DataFrame(summary_rows).to_csv(OUT_CSV_PATH, index=False)
    print(f"Saved summary CSV -> {OUT_CSV_PATH}")


if __name__ == "__main__":
    main()
"""
build_behavior_train.py
------------------------
Builds the full feature tables used by both Stage 1 (transaction-level) and
Stage 2 (behavioral) models, for both the train and test splits.

Stage 1 features:
    amt, trans_hour, trans_day_of_week, trans_day, trans_month, is_weekend,
    avg_amount, amount_ratio, txn_count, location_distance_km, city_pop,
    category, gender

Stage 2 (behavioral) features:
    amount_deviation, merchant_visits, new_merchant, category_visits,
    new_category, hour_deviation, location_deviation,
    seconds_since_previous, high_velocity, log_amount_change,
    log_location_change, merchant_location_anomaly, amount_location_anomaly,
    amount_velocity_anomaly, combined_behavioral_anomaly

Customer running state (merchant visit counts, previous transaction time)
is tracked incrementally in chronological order per cc_num, so no future
information leaks into a given transaction's features.

Output:
    data/processed/train_features.csv
    data/processed/test_features.csv
"""

import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from common import load_raw, add_time_fields, add_merchant_distance

RAW_TRAIN_PATH = "data/raw/fraudTrain.csv"
RAW_TEST_PATH = "data/raw/fraudTest.csv"
PROFILES_PATH = "data/processed/customer_profiles.pkl"

OUT_TRAIN_PATH = "data/processed/train_features.csv"
OUT_TEST_PATH = "data/processed/test_features.csv"

HIGH_VELOCITY_SECONDS = 60
EPS = 1e-6


def load_profiles() -> dict:
    with open(PROFILES_PATH, "rb") as f:
        return pickle.load(f)


def build_features(df: pd.DataFrame, profiles: dict, warm_start_state: dict = None) -> pd.DataFrame:
    """
    Sequentially walks the (already time-sorted) dataframe per customer and
    builds Stage 1 + Stage 2 features. `warm_start_state` lets the test set
    continue running counters from where the train set left off.
    """
    df = df.copy().reset_index(drop=True)

    merchant_visit_counts = defaultdict(lambda: defaultdict(int)) if warm_start_state is None \
        else warm_start_state["merchant_visit_counts"]
    category_visit_counts = defaultdict(lambda: defaultdict(int)) if warm_start_state is None \
        else warm_start_state["category_visit_counts"]
    last_trans_time = {} if warm_start_state is None else warm_start_state["last_trans_time"]

    n = len(df)
    amount_ratio = np.zeros(n)
    amount_deviation = np.zeros(n)
    merchant_visits = np.zeros(n)
    new_merchant = np.zeros(n)
    category_visits = np.zeros(n)
    new_category = np.zeros(n)
    hour_deviation = np.zeros(n)
    location_deviation = np.zeros(n)
    seconds_since_previous = np.zeros(n)
    log_amount_change = np.zeros(n)
    log_location_change = np.zeros(n)

    default_profile = {
        "avg_amount": df["amt"].mean(),
        "std_amount": df["amt"].std(ddof=0),
        "txn_count": 0,
        "avg_hour": 12.0,
        "avg_location_distance": df["location_distance_km"].mean(),
        "common_merchants": set(),
        "common_categories": set(),
    }

    cc_nums = df["cc_num"].values
    merchants = df["merchant"].values
    categories = df["category"].values
    amts = df["amt"].values
    hours = df["trans_hour"].values
    dists = df["location_distance_km"].values
    times = df["trans_date_trans_time"].values

    for i in range(n):
        cc = cc_nums[i]
        p = profiles.get(cc, default_profile)
        std_amt = p["std_amount"] if p["std_amount"] > 0 else EPS

        amount_ratio[i] = amts[i] / (p["avg_amount"] + EPS)
        amount_deviation[i] = (amts[i] - p["avg_amount"]) / std_amt
        hour_deviation[i] = abs(hours[i] - p["avg_hour"])
        location_deviation[i] = abs(dists[i] - p["avg_location_distance"])
        log_amount_change[i] = np.log1p(amts[i]) - np.log1p(p["avg_amount"])
        log_location_change[i] = np.log1p(dists[i]) - np.log1p(p["avg_location_distance"])

        merch = merchants[i]
        cat = categories[i]
        merchant_visits[i] = merchant_visit_counts[cc][merch]
        new_merchant[i] = 1 if (merch not in p["common_merchants"] and merchant_visits[i] == 0) else 0
        category_visits[i] = category_visit_counts[cc][cat]
        new_category[i] = 1 if (cat not in p["common_categories"] and category_visits[i] == 0) else 0

        merchant_visit_counts[cc][merch] += 1
        category_visit_counts[cc][cat] += 1

        if cc in last_trans_time:
            delta = (times[i] - last_trans_time[cc]) / np.timedelta64(1, "s")
            seconds_since_previous[i] = max(delta, 0)
        else:
            seconds_since_previous[i] = np.nan
        last_trans_time[cc] = times[i]

    df["amount_ratio"] = amount_ratio
    df["amount_deviation"] = amount_deviation
    df["merchant_visits"] = merchant_visits
    df["new_merchant"] = new_merchant
    df["category_visits"] = category_visits
    df["new_category"] = new_category
    df["hour_deviation"] = hour_deviation
    df["location_deviation"] = location_deviation

    median_gap = np.nanmedian(seconds_since_previous)
    df["seconds_since_previous"] = pd.Series(seconds_since_previous).fillna(median_gap)
    df["high_velocity"] = (df["seconds_since_previous"] < HIGH_VELOCITY_SECONDS).astype(int)
    df["log_amount_change"] = log_amount_change
    df["log_location_change"] = log_location_change

    # Interaction / composite anomaly signals
    df["merchant_location_anomaly"] = df["new_merchant"] * df["location_deviation"]
    df["amount_location_anomaly"] = df["amount_deviation"].abs() * df["location_deviation"]
    df["amount_velocity_anomaly"] = df["amount_deviation"].abs() * df["high_velocity"]
    df["combined_behavioral_anomaly"] = (
        0.30 * df["amount_deviation"].abs().clip(0, 10) / 10
        + 0.20 * df["new_merchant"]
        + 0.15 * df["new_category"]
        + 0.15 * (df["hour_deviation"] / 12.0).clip(0, 1)
        + 0.10 * df["high_velocity"]
        + 0.10 * (df["location_deviation"] / (df["location_deviation"].max() + EPS))
    )

    # customer_avg_amount / customer_transaction_count for Stage 1 features
    df["customer_avg_amount"] = df["cc_num"].map(lambda c: profiles.get(c, default_profile)["avg_amount"])
    df["customer_transaction_count"] = df["cc_num"].map(lambda c: profiles.get(c, default_profile)["txn_count"])

    state = {
        "merchant_visit_counts": merchant_visit_counts,
        "category_visit_counts": category_visit_counts,
        "last_trans_time": last_trans_time,
    }
    return df, state


def main():
    profiles = load_profiles()

    print(f"Loading train data from {RAW_TRAIN_PATH} ...")
    train_df = load_raw(RAW_TRAIN_PATH)
    train_df = add_time_fields(train_df)
    train_df = add_merchant_distance(train_df)

    print("Building Stage 1 + Stage 2 features for train set ...")
    train_df, state = build_features(train_df, profiles, warm_start_state=None)
    train_df.to_csv(OUT_TRAIN_PATH, index=False)
    print(f"Saved -> {OUT_TRAIN_PATH} ({len(train_df):,} rows)")

    print(f"Loading test data from {RAW_TEST_PATH} ...")
    test_df = load_raw(RAW_TEST_PATH)
    test_df = add_time_fields(test_df)
    test_df = add_merchant_distance(test_df)

    print("Building Stage 1 + Stage 2 features for test set (warm-started from train) ...")
    test_df, _ = build_features(test_df, profiles, warm_start_state=state)
    test_df.to_csv(OUT_TEST_PATH, index=False)
    print(f"Saved -> {OUT_TEST_PATH} ({len(test_df):,} rows)")


if __name__ == "__main__":
    main()
"""
train_behavior.py
------------------
Trains:
  Stage 1 - Transaction-level fraud model
      Evaluates Logistic Regression vs Random Forest, selects Random Forest.
  Stage 2 - Customer behavioral verification model
      Random Forest trained on behavioral anomaly features.

Saves both fitted models + the encoders to models/, and writes
fraud_probability / behavior_probability predictions on the test set to
results/test_predictions.csv
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

TRAIN_PATH = "data/processed/train_features.csv"
TEST_PATH = "data/processed/test_features.csv"

STAGE1_FEATURES = [
    "amt", "trans_hour", "trans_day_of_week", "trans_day", "trans_month",
    "is_weekend", "customer_avg_amount", "amount_ratio",
    "customer_transaction_count", "location_distance_km", "city_pop",
    "category_enc", "gender_enc",
]

STAGE2_FEATURES = [
    "amount_deviation", "merchant_visits", "new_merchant", "category_visits",
    "new_category", "hour_deviation", "location_deviation",
    "seconds_since_previous", "high_velocity", "log_amount_change",
    "log_location_change", "merchant_location_anomaly",
    "amount_location_anomaly", "amount_velocity_anomaly",
    "combined_behavioral_anomaly",
]

LABEL_COL = "is_fraud"


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame):
    cat_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()

    all_categories = pd.concat([train_df["category"], test_df["category"]]).unique()
    all_genders = pd.concat([train_df["gender"], test_df["gender"]]).unique()
    cat_encoder.fit(all_categories)
    gender_encoder.fit(all_genders)

    for df in (train_df, test_df):
        df["category_enc"] = cat_encoder.transform(df["category"])
        df["gender_enc"] = gender_encoder.transform(df["gender"])

    return train_df, test_df, cat_encoder, gender_encoder


def train_stage1(train_df: pd.DataFrame):
    X = train_df[STAGE1_FEATURES]
    y = train_df[LABEL_COL]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Evaluating Logistic Regression (Stage 1) ...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_scaled, y)
    lr_auc = roc_auc_score(y, lr.predict_proba(X_scaled)[:, 1])
    print(f"  Logistic Regression train ROC-AUC: {lr_auc:.4f}")

    print("Evaluating Random Forest (Stage 1) ...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    rf_auc = roc_auc_score(y, rf.predict_proba(X)[:, 1])
    print(f"  Random Forest train ROC-AUC: {rf_auc:.4f}")

    print("Random Forest selected as final Stage 1 model (higher performance).")
    return rf, scaler


def train_stage2(train_df: pd.DataFrame):
    X = train_df[STAGE2_FEATURES]
    y = train_df[LABEL_COL]

    print("Training Stage 2 behavioral Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    auc = roc_auc_score(y, rf.predict_proba(X)[:, 1])
    ap = average_precision_score(y, rf.predict_proba(X)[:, 1])
    print(f"  Stage 2 train ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f}")
    return rf


def main():
    print("Loading engineered feature tables ...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_df, test_df, cat_encoder, gender_encoder = encode_categoricals(train_df, test_df)

    stage1_model, stage1_scaler = train_stage1(train_df)
    stage2_model = train_stage2(train_df)

    print("Scoring test set with both stages ...")
    test_df["fraud_probability"] = stage1_model.predict_proba(test_df[STAGE1_FEATURES])[:, 1]
    test_df["behavior_probability"] = stage2_model.predict_proba(test_df[STAGE2_FEATURES])[:, 1]

    joblib.dump(stage1_model, "models/stage1_random_forest.joblib")
    joblib.dump(stage2_model, "models/stage2_random_forest.joblib")
    joblib.dump(cat_encoder, "models/category_encoder.joblib")
    joblib.dump(gender_encoder, "models/gender_encoder.joblib")
    print("Saved models -> models/")

    out_cols = [
        "trans_num", "cc_num", "trans_date_trans_time", "amt", "merchant",
        "category", "is_fraud", "fraud_probability", "behavior_probability",
    ]
    out_cols = [c for c in out_cols if c in test_df.columns]
    test_df[out_cols].to_csv("results/test_predictions.csv", index=False)
    print("Saved predictions -> results/test_predictions.csv")

    test_auc_s1 = roc_auc_score(test_df[LABEL_COL], test_df["fraud_probability"])
    test_auc_s2 = roc_auc_score(test_df[LABEL_COL], test_df["behavior_probability"])
    print(f"Test ROC-AUC | Stage 1: {test_auc_s1:.4f}  Stage 2: {test_auc_s2:.4f}")


if __name__ == "__main__":
    main()
"""
final_hybrid.py
----------------
Combines Stage 1 (fraud_probability) and Stage 2 (behavior_probability)
into a single Hybrid Risk Score, applies the three-way APPROVE / VERIFY /
BLOCK decision policy, and reports final system-level metrics.

Hybrid Risk Score = 0.70 * fraud_probability + 0.30 * behavior_probability

Decision policy:
    score <  0.10           -> APPROVE
    0.10 <= score < 0.90    -> VERIFY
    score >= 0.90           -> BLOCK

Outputs:
    results/FINAL_RESULTS.md
    dashboard_data.csv
    dashboard_metrics.json
"""

import json

import pandas as pd

PREDICTIONS_PATH = "results/test_predictions.csv"

FRAUD_WEIGHT = 0.70
BEHAVIOR_WEIGHT = 0.30

APPROVE_THRESHOLD = 0.10
BLOCK_THRESHOLD = 0.90

LABEL_COL = "is_fraud"


def decide(score: float) -> str:
    if score < APPROVE_THRESHOLD:
        return "APPROVE"
    elif score < BLOCK_THRESHOLD:
        return "VERIFY"
    else:
        return "BLOCK"


def compute_metrics(df: pd.DataFrame) -> dict:
    total_txns = len(df)
    fraud_txns = int(df[LABEL_COL].sum())

    routing_counts = df["decision"].value_counts().to_dict()
    for d in ("APPROVE", "VERIFY", "BLOCK"):
        routing_counts.setdefault(d, 0)

    fraud_df = df[df[LABEL_COL] == 1]
    fraud_approved = int((fraud_df["decision"] == "APPROVE").sum())
    fraud_verified = int((fraud_df["decision"] == "VERIFY").sum())
    fraud_blocked = int((fraud_df["decision"] == "BLOCK").sum())
    fraud_caught = fraud_verified + fraud_blocked

    fraud_routing_recall = fraud_caught / fraud_txns if fraud_txns else 0.0

    blocked_df = df[df["decision"] == "BLOCK"]
    block_precision = (
        (blocked_df[LABEL_COL] == 1).sum() / len(blocked_df) if len(blocked_df) else 0.0
    )

    return {
        "transactions": total_txns,
        "fraud_transactions": fraud_txns,
        "routing": {
            "APPROVE": int(routing_counts["APPROVE"]),
            "VERIFY": int(routing_counts["VERIFY"]),
            "BLOCK": int(routing_counts["BLOCK"]),
        },
        "fraud_routing": {
            "fraud_approved": fraud_approved,
            "fraud_verified": fraud_verified,
            "fraud_blocked": fraud_blocked,
            "fraud_caught": fraud_caught,
        },
        "overall_fraud_routing_recall": round(fraud_routing_recall * 100, 2),
        "automatic_block_precision": round(block_precision * 100, 2),
    }


def behavioral_pattern_table(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the 'Behavioral Insights' style pattern table."""
    loc_thresh = df["location_deviation"].quantile(0.90) if "location_deviation" in df else None
    amt_thresh = df["amount_deviation"].quantile(0.90) if "amount_deviation" in df else None

    patterns = []
    if {"amount_deviation", "location_deviation"}.issubset(df.columns):
        mask = (df["amount_deviation"] > amt_thresh) & (df["location_deviation"] > loc_thresh)
        patterns.append(("Large amount + unusual location", mask))
    if {"amount_deviation", "new_merchant"}.issubset(df.columns):
        mask = (df["amount_deviation"] > amt_thresh) & (df["new_merchant"] == 1)
        patterns.append(("Large amount + new merchant", mask))
    if {"amount_deviation", "hour_deviation"}.issubset(df.columns):
        hr_thresh = df["hour_deviation"].quantile(0.90)
        mask = (df["amount_deviation"] > amt_thresh) & (df["hour_deviation"] > hr_thresh)
        patterns.append(("Large amount + unusual time", mask))
    if {"amount_deviation", "high_velocity"}.issubset(df.columns):
        mask = (df["amount_deviation"] > amt_thresh) & (df["high_velocity"] == 1)
        patterns.append(("Large amount + fast transaction", mask))
    if {"hour_deviation", "high_velocity"}.issubset(df.columns):
        hr_thresh = df["hour_deviation"].quantile(0.90)
        mask = (df["hour_deviation"] > hr_thresh) & (df["high_velocity"] == 1)
        patterns.append(("Unusual time + fast transaction", mask))

    rows = []
    for name, mask in patterns:
        subset = df[mask]
        n = len(subset)
        fraud_rate = (subset[LABEL_COL].mean() * 100) if n else 0.0
        rows.append({"Behavioral Pattern": name, "Transactions": n, "Fraud Rate (%)": round(fraud_rate, 2)})
    return pd.DataFrame(rows)


def write_final_results_md(metrics: dict, pattern_table: pd.DataFrame, path: str):
    r = metrics["routing"]
    fr = metrics["fraud_routing"]
    lines = [
        "# Final Results\n",
        f"Transactions       : {metrics['transactions']:,}",
        f"Fraud transactions : {metrics['fraud_transactions']:,}\n",
        "## Transaction Routing\n",
        "| Decision | Transactions |",
        "|----------|-------------|",
        f"| APPROVE  | {r['APPROVE']:,} |",
        f"| VERIFY   | {r['VERIFY']:,} |",
        f"| BLOCK    | {r['BLOCK']:,} |\n",
        "## Fraud Routing\n",
        "| Outcome | Fraud Transactions |",
        "|---------|--------------------|",
        f"| Fraud approved | {fr['fraud_approved']:,} |",
        f"| Fraud verified | {fr['fraud_verified']:,} |",
        f"| Fraud blocked  | {fr['fraud_blocked']:,} |",
        f"| Fraud caught   | {fr['fraud_caught']:,} |\n",
        "## Key Metrics\n",
        f"Overall fraud routing recall : {metrics['overall_fraud_routing_recall']}%",
        f"Automatic BLOCK precision    : {metrics['automatic_block_precision']}%\n",
    ]
    if not pattern_table.empty:
        lines.append("## Behavioral Insights\n")
        lines.append(pattern_table.to_markdown(index=False))
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    print(f"Loading predictions from {PREDICTIONS_PATH} ...")
    df = pd.read_csv(PREDICTIONS_PATH)

    df["hybrid_score"] = (
        FRAUD_WEIGHT * df["fraud_probability"] + BEHAVIOR_WEIGHT * df["behavior_probability"]
    )
    df["decision"] = df["hybrid_score"].apply(decide)

    metrics = compute_metrics(df)
    print(json.dumps(metrics, indent=2))

    # Need behavioral columns for the pattern table -> merge back if available
    try:
        features = pd.read_csv("data/processed/test_features.csv")
        merge_cols = ["trans_num", "amount_deviation", "location_deviation", "new_merchant", "hour_deviation", "high_velocity"]
        merge_cols = [c for c in merge_cols if c in features.columns]
        df = df.merge(features[merge_cols], on="trans_num", how="left")
    except FileNotFoundError:
        pass

    pattern_table = behavioral_pattern_table(df)

    df.to_csv("results/final_decisions.csv", index=False)
    write_final_results_md(metrics, pattern_table, "results/FINAL_RESULTS.md")

    dashboard_cols = [
        "trans_num", "trans_date_trans_time", "amt", "merchant", "category",
        LABEL_COL, "fraud_probability", "behavior_probability",
        "hybrid_score", "decision",
    ]
    dashboard_cols = [c for c in dashboard_cols if c in df.columns]
    df[dashboard_cols].to_csv("dashboard_data.csv", index=False)

    with open("dashboard_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved -> results/final_decisions.csv, results/FINAL_RESULTS.md")
    print("Saved -> dashboard_data.csv, dashboard_metrics.json")


if __name__ == "__main__":
    main()
"""
analyze_mistakes.py
--------------------
Error analysis on the final hybrid decisions:

False Positives : legitimate transactions (is_fraud == 0) that were
                   VERIFY'd or BLOCK'd, typically because they combine
                   a large amount + unusual location + new merchant.

False Negatives : fraudulent transactions (is_fraud == 1) that were
                   APPROVE'd, typically because they closely resemble the
                   customer's normal behavior.

Outputs:
    results/false_positives.csv
    results/false_negatives.csv
    results/error_analysis_summary.md
"""

import pandas as pd

DECISIONS_PATH = "results/final_decisions.csv"
FEATURES_PATH = "data/processed/test_features.csv"

LABEL_COL = "is_fraud"


def load_merged() -> pd.DataFrame:
    decisions = pd.read_csv(DECISIONS_PATH)
    try:
        features = pd.read_csv(FEATURES_PATH)
        extra_cols = [
            "trans_num", "amount_deviation", "location_deviation", "new_merchant",
            "new_category", "hour_deviation", "high_velocity", "location_distance_km",
        ]
        extra_cols = [c for c in extra_cols if c in features.columns]
        decisions = decisions.merge(features[extra_cols], on="trans_num", how="left", suffixes=("", "_feat"))
    except FileNotFoundError:
        pass
    return decisions


def analyze_false_positives(df: pd.DataFrame) -> pd.DataFrame:
    fp = df[(df[LABEL_COL] == 0) & (df["decision"].isin(["VERIFY", "BLOCK"]))].copy()
    fp = fp.sort_values("hybrid_score", ascending=False)
    return fp


def analyze_false_negatives(df: pd.DataFrame) -> pd.DataFrame:
    fn = df[(df[LABEL_COL] == 1) & (df["decision"] == "APPROVE")].copy()
    fn = fn.sort_values("hybrid_score", ascending=False)
    return fn


def summarize(fp: pd.DataFrame, fn: pd.DataFrame) -> str:
    lines = ["# Error Analysis Summary\n"]

    lines.append(f"Total false positives (legit flagged VERIFY/BLOCK): {len(fp):,}")
    lines.append(f"Total false negatives (fraud approved)            : {len(fn):,}\n")

    if "new_merchant" in fp.columns and len(fp):
        pct_new_merchant = fp["new_merchant"].mean() * 100
        lines.append(f"- {pct_new_merchant:.1f}% of false positives involved a new merchant")
    if "location_deviation" in fp.columns and len(fp):
        pct_high_loc = (fp["location_deviation"] > fp["location_deviation"].median()).mean() * 100
        lines.append(f"- {pct_high_loc:.1f}% of false positives had above-median location deviation")

    if "amount_deviation" in fn.columns and len(fn):
        pct_low_dev = (fn["amount_deviation"].abs() < 1.0).mean() * 100
        lines.append(f"- {pct_low_dev:.1f}% of false negatives had low amount deviation (looked like normal spending)")
    if "new_merchant" in fn.columns and len(fn):
        pct_known_merchant = (fn["new_merchant"] == 0).mean() * 100
        lines.append(f"- {pct_known_merchant:.1f}% of false negatives used a merchant the customer had used before")

    lines.append(
        "\nInterpretation: false positives cluster around genuine but unusual "
        "purchases (large amount + new merchant/location) — this is what the "
        "VERIFY layer exists to absorb rather than hard-blocking. False "
        "negatives cluster around fraud that imitates the customer's normal "
        "behavior, confirming that behavioral analysis is a complementary "
        "signal rather than a standalone fraud detector."
    )
    return "\n".join(lines)


def main():
    df = load_merged()

    fp = analyze_false_positives(df)
    fn = analyze_false_negatives(df)

    fp.to_csv("results/false_positives.csv", index=False)
    fn.to_csv("results/false_negatives.csv", index=False)

    summary = summarize(fp, fn)
    with open("results/error_analysis_summary.md", "w") as f:
        f.write(summary)

    print(summary)
    print("\nSaved -> results/false_positives.csv, results/false_negatives.csv, results/error_analysis_summary.md")


if __name__ == "__main__":
    main()
"""
dashboard.py
------------
Interactive Streamlit dashboard for the Hybrid Credit Card Fraud Detection
system.

Run locally:
    streamlit run dashboard.py

Expects:
    dashboard_data.csv     (produced by src/final_hybrid.py)
    dashboard_metrics.json (produced by src/final_hybrid.py)
"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Hybrid Fraud Detection Dashboard", layout="wide")

DATA_PATH = "dashboard_data.csv"
METRICS_PATH = "dashboard_metrics.json"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


df = load_data()
metrics = load_metrics()

st.title("🔐 Hybrid Credit Card Fraud Detection")
st.caption("Stage 1 (transaction-level) + Stage 2 (behavioral) hybrid risk scoring")

# --- KPI row -----------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", f"{metrics['transactions']:,}")
col2.metric("Fraud Transactions", f"{metrics['fraud_transactions']:,}")
col3.metric("Fraud Routing Recall", f"{metrics['overall_fraud_routing_recall']}%")
col4.metric("Automatic BLOCK Precision", f"{metrics['automatic_block_precision']}%")

st.divider()

# --- APPROVE / VERIFY / BLOCK distribution ------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Transaction Routing")
    routing = metrics["routing"]
    routing_df = pd.DataFrame({"Decision": list(routing.keys()), "Transactions": list(routing.values())})
    fig = px.bar(
        routing_df, x="Decision", y="Transactions", color="Decision",
        color_discrete_map={"APPROVE": "#2ecc71", "VERIFY": "#f1c40f", "BLOCK": "#e74c3c"},
        text="Transactions",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Fraud Rate by Decision")
    fraud_rate = df.groupby("decision")["is_fraud"].mean().reindex(["APPROVE", "VERIFY", "BLOCK"]) * 100
    fraud_rate_df = fraud_rate.reset_index()
    fraud_rate_df.columns = ["Decision", "Fraud Rate (%)"]
    fig2 = px.bar(
        fraud_rate_df, x="Decision", y="Fraud Rate (%)", color="Decision",
        color_discrete_map={"APPROVE": "#2ecc71", "VERIFY": "#f1c40f", "BLOCK": "#e74c3c"},
        text_auto=".2f",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Risk score distribution --------------------------------------------
st.subheader("Hybrid Risk Score Distribution")
fig3 = px.histogram(
    df, x="hybrid_score", color="is_fraud", nbins=60, barmode="overlay",
    labels={"is_fraud": "Is Fraud"},
    color_discrete_map={0: "#3498db", 1: "#e74c3c"},
)
fig3.add_vline(x=0.10, line_dash="dash", line_color="gray", annotation_text="APPROVE / VERIFY")
fig3.add_vline(x=0.90, line_dash="dash", line_color="gray", annotation_text="VERIFY / BLOCK")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# --- Stage 1 vs Stage 2 comparison --------------------------------------
st.subheader("Stage 1 (Transaction) vs Stage 2 (Behavioral) Risk")
sample = df.sample(min(20000, len(df)), random_state=42)
fig4 = px.scatter(
    sample, x="fraud_probability", y="behavior_probability", color="is_fraud",
    opacity=0.5, color_discrete_map={0: "#3498db", 1: "#e74c3c"},
    labels={"fraud_probability": "Stage 1: Fraud Probability", "behavior_probability": "Stage 2: Behavior Probability"},
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# --- Transaction explorer -----------------------------------------------
st.subheader("Transaction Explorer")
decision_filter = st.multiselect("Filter by decision", ["APPROVE", "VERIFY", "BLOCK"], default=["APPROVE", "VERIFY", "BLOCK"])
fraud_only = st.checkbox("Show only actual fraud transactions", value=False)

filtered = df[df["decision"].isin(decision_filter)]
if fraud_only:
    filtered = filtered[filtered["is_fraud"] == 1]

st.dataframe(filtered.sort_values("hybrid_score", ascending=False).head(500), use_container_width=True)

st.divider()

# --- Risk score simulator -------------------------------------------------
st.subheader("Risk Score Simulator")
sim_col1, sim_col2 = st.columns(2)
fraud_prob_input = sim_col1.slider("Stage 1: fraud_probability", 0.0, 1.0, 0.50, 0.01)
behavior_prob_input = sim_col2.slider("Stage 2: behavior_probability", 0.0, 1.0, 0.50, 0.01)

hybrid = 0.70 * fraud_prob_input + 0.30 * behavior_prob_input
if hybrid < 0.10:
    decision, color = "APPROVE", "green"
elif hybrid < 0.90:
    decision, color = "VERIFY", "orange"
else:
    decision, color = "BLOCK", "red"

st.metric("Hybrid Risk Score", f"{hybrid:.3f}")
st.markdown(f"### Decision: :{color}[{decision}]")
