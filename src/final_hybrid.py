import pandas as pd
import numpy as np
import joblib

print("=" * 70)
print("HYBRID CREDIT CARD FRAUD DETECTION")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/6] Loading datasets...")

train = pd.read_csv("data/raw/fraudTrain.csv")
test = pd.read_csv("data/raw/fraudTest.csv")

train["trans_date_trans_time"] = pd.to_datetime(
    train["trans_date_trans_time"]
)

test["trans_date_trans_time"] = pd.to_datetime(
    test["trans_date_trans_time"]
)

# ------------------------------------------------------------
# STAGE 1 PREDICTIONS MUST BE GENERATED BEFORE SORTING
# ------------------------------------------------------------
# test_features.csv is in the original transaction order.
# Generate predictions while test is still in that same order,
# then attach them to the transactions before behavioral sorting.

stage1_features = [
    "amt",
    "hour",
    "day_of_week",
    "day",
    "month",
    "is_weekend",
    "customer_avg_amt",
    "amount_ratio",
    "customer_transaction_count",
    "location_distance",
    "city_pop",
    "category",
    "gender"
]

stage1_data = pd.read_csv(
    "data/processed/test_features.csv"
)

stage1_model = joblib.load(
    "models/random_forest_model.pkl"
)

test["fraud_probability"] = (
    stage1_model.predict_proba(
        stage1_data[stage1_features]
    )[:, 1]
)

# NOW sort test for chronological customer behavior processing.
test = (
    test
    .sort_values(["cc_num", "trans_date_trans_time"])
    .reset_index(drop=True)
)

# ============================================================
# CUSTOMER BEHAVIORAL PROFILE
# ============================================================

print("[2/6] Building customer profiles...")

profile = (
    train.groupby("cc_num")
    .agg(
        avg_amount=("amt", "mean"),
        std_amount=("amt", "std"),
        avg_hour=(
            "trans_date_trans_time",
            lambda x: x.dt.hour.mean()
        ),
        avg_lat=("lat", "mean"),
        avg_long=("long", "mean")
    )
    .reset_index()
)

profile["std_amount"] = profile["std_amount"].fillna(0)

merchant = (
    train.groupby(["cc_num", "merchant"])
    .size()
    .reset_index(name="merchant_visits")
)

category = (
    train.groupby(["cc_num", "category"])
    .size()
    .reset_index(name="category_visits")
)

last = (
    train
    .sort_values("trans_date_trans_time")
    .groupby("cc_num")
    .tail(1)
    [
        [
            "cc_num",
            "trans_date_trans_time",
            "amt",
            "merch_lat",
            "merch_long"
        ]
    ]
)

last.columns = [
    "cc_num",
    "previous_time",
    "previous_amount",
    "previous_lat",
    "previous_long"
]

# ============================================================
# CREATE BEHAVIOR FEATURES
# ============================================================

print("[3/6] Creating behavioral features...")

d = test.copy()

d["hour"] = d["trans_date_trans_time"].dt.hour

d = d.merge(profile, on="cc_num", how="left")
d = d.merge(
    merchant,
    on=["cc_num", "merchant"],
    how="left"
)
d = d.merge(
    category,
    on=["cc_num", "category"],
    how="left"
)
d = d.merge(last, on="cc_num", how="left")

d["merchant_visits"] = d["merchant_visits"].fillna(0)
d["category_visits"] = d["category_visits"].fillna(0)

d["new_merchant"] = (
    d["merchant_visits"] == 0
).astype(int)

d["new_category"] = (
    d["category_visits"] == 0
).astype(int)

d["amount_z"] = (
    (d["amt"] - d["avg_amount"]) /
    (d["std_amount"] + 1e-6)
)

d["amount_z"] = (
    d["amount_z"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

d["amount_deviation"] = d["amount_z"].abs()

hour_difference = (
    d["hour"] - d["avg_hour"]
).abs()

d["hour_deviation"] = np.minimum(
    hour_difference,
    24 - hour_difference
)

d["location_deviation"] = np.sqrt(
    (d["merch_lat"] - d["avg_lat"]) ** 2 +
    (d["merch_long"] - d["avg_long"]) ** 2
)

# ============================================================
# PREVIOUS TRANSACTION / VELOCITY
# ============================================================

d["current_previous_time"] = (
    d.groupby("cc_num")[
        "trans_date_trans_time"
    ].shift(1)
)

d["current_previous_amount"] = (
    d.groupby("cc_num")["amt"].shift(1)
)

d["current_previous_lat"] = (
    d.groupby("cc_num")["merch_lat"].shift(1)
)

d["current_previous_long"] = (
    d.groupby("cc_num")["merch_long"].shift(1)
)

d["prev_time"] = (
    d["current_previous_time"]
    .fillna(d["previous_time"])
)

d["prev_amount"] = (
    d["current_previous_amount"]
    .fillna(d["previous_amount"])
)

d["prev_lat"] = (
    d["current_previous_lat"]
    .fillna(d["previous_lat"])
)

d["prev_long"] = (
    d["current_previous_long"]
    .fillna(d["previous_long"])
)

d["seconds_since_previous"] = (
    d["trans_date_trans_time"] -
    d["prev_time"]
).dt.total_seconds()

d["seconds_since_previous"] = (
    d["seconds_since_previous"]
    .fillna(999999)
)

d["high_velocity"] = (
    d["seconds_since_previous"] < 300
).astype(int)

# ============================================================
# AMOUNT CHANGE
# ============================================================

d["amount_change"] = (
    d["amt"] /
    (d["prev_amount"] + 1e-6)
)

d["amount_change"] = (
    d["amount_change"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

d["log_amount_change"] = np.log1p(
    d["amount_change"].clip(upper=20)
)

# ============================================================
# LOCATION CHANGE
# ============================================================

d["location_change"] = np.sqrt(
    (d["merch_lat"] - d["prev_lat"]) ** 2 +
    (d["merch_long"] - d["prev_long"]) ** 2
)

d["log_location_change"] = np.log1p(
    d["location_change"]
)

# ============================================================
# COMBINED BEHAVIOR SIGNALS
# ============================================================

d["merchant_location_anomaly"] = (
    d["new_merchant"] *
    (d["location_deviation"] > 0.8)
).astype(int)

d["amount_location_anomaly"] = (
    (d["amount_deviation"] > 2) *
    (d["location_deviation"] > 0.8)
).astype(int)

d["amount_velocity_anomaly"] = (
    (d["amount_deviation"] > 2) *
    d["high_velocity"]
).astype(int)

d["combined_behavior_anomaly"] = (
    d["new_merchant"] +
    (d["amount_deviation"] > 2).astype(int) +
    (d["location_deviation"] > 0.8).astype(int) +
    d["high_velocity"] +
    (d["amount_change"] > 5).astype(int)
)

# ============================================================
# STAGE 2 — BEHAVIOR MODEL
# ============================================================

print("[4/6] Running behavioral model...")

behavior_features = [
    "amount_deviation",
    "merchant_visits",
    "new_merchant",
    "category_visits",
    "new_category",
    "hour_deviation",
    "location_deviation",
    "seconds_since_previous",
    "high_velocity",
    "log_amount_change",
    "log_location_change",
    "merchant_location_anomaly",
    "amount_location_anomaly",
    "amount_velocity_anomaly",
    "combined_behavior_anomaly"
]

X_behavior = (
    d[behavior_features]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

behavior_model = joblib.load(
    "models/behavior_model_v3.pkl"
)

d["behavior_probability"] = (
    behavior_model.predict_proba(
        X_behavior
    )[:, 1]
)

# ============================================================
# STAGE 1 — RANDOM FOREST
# ============================================================

print("[5/6] Using correctly aligned Stage-1 predictions...")

# fraud_probability was calculated BEFORE test sorting,
# so it remains attached to the correct transaction.

d["fraud_probability"] = test["fraud_probability"].values

# ============================================================
# HYBRID MODEL
# ============================================================

d["hybrid_probability"] = (
    0.70 * d["fraud_probability"] +
    0.30 * d["behavior_probability"]
)

# ============================================================
# FINAL DECISION ENGINE
# ============================================================

def make_decision(score):

    if score < 0.10:
        return "APPROVE"

    if score < 0.90:
        return "VERIFY"

    return "BLOCK"


d["decision"] = (
    d["hybrid_probability"]
    .apply(make_decision)
)

# ============================================================
# SAVE FINAL RESULTS
# ============================================================

output_cols = [
    "cc_num",
    "trans_date_trans_time",
    "amt",
    "merchant",
    "category",
    "is_fraud",
    "fraud_probability",
    "behavior_probability",
    "hybrid_probability",
    "decision"
]

d[output_cols].to_csv(
    "data/processed/final_hybrid_results.csv",
    index=False
)

# ============================================================
# EVALUATION
# ============================================================

print("[6/6] Evaluating final decision system...")

fraud = d["is_fraud"] == 1
legitimate = d["is_fraud"] == 0

approve = d["decision"] == "APPROVE"
verify = d["decision"] == "VERIFY"
block = d["decision"] == "BLOCK"

total_fraud = int(fraud.sum())

fraud_approved = int((fraud & approve).sum())
fraud_verified = int((fraud & verify).sum())
fraud_blocked = int((fraud & block).sum())

legitimate_blocked = int(
    (legitimate & block).sum()
)

fraud_caught = int(
    (fraud & ~approve).sum()
)

routing_recall = (
    fraud_caught / total_fraud
)

block_count = int(block.sum())

block_precision = (
    fraud_blocked / block_count
    if block_count > 0
    else 0
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL PROJECT RESULTS")
print("=" * 70)

print("\nDATASET")
print(f"Transactions       : {len(d):,}")
print(f"Fraud transactions : {total_fraud:,}")

print("\nDECISIONS")
print(f"APPROVE : {int(approve.sum()):,}")
print(f"VERIFY  : {int(verify.sum()):,}")
print(f"BLOCK   : {int(block.sum()):,}")

print("\nFRAUD ROUTING")
print(f"Fraud approved : {fraud_approved:,}")
print(f"Fraud verified : {fraud_verified:,}")
print(f"Fraud blocked  : {fraud_blocked:,}")
print(f"Fraud caught   : {fraud_caught:,}")

print(
    f"Overall routing recall : "
    f"{routing_recall:.2%}"
)

print("\nAUTOMATIC BLOCKING")
print(
    f"Legitimate blocked : "
    f"{legitimate_blocked:,}"
)

print(
    f"Block precision : "
    f"{block_precision:.2%}"
)

print("\nDECISION THRESHOLDS")
print("APPROVE : score < 0.10")
print("VERIFY  : 0.10 <= score < 0.90")
print("BLOCK   : score >= 0.90")

print("\nOUTPUT")
print(
    "data/processed/final_hybrid_results.csv"
)

print("=" * 70)
