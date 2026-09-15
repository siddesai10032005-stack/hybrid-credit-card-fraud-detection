import pandas as pd
import numpy as np

print("Loading data...")

test = pd.read_csv("data/raw/fraudTest.csv")
train = pd.read_csv("data/raw/fraudTrain.csv")

test["trans_date_trans_time"] = pd.to_datetime(test["trans_date_trans_time"])
train["trans_date_trans_time"] = pd.to_datetime(train["trans_date_trans_time"])

test = test.sort_values(
    ["cc_num", "trans_date_trans_time"]
).reset_index(drop=True)

# ==================================================
# RECREATE EXACT FEATURES FROM final_hybrid.py
# ==================================================

profile = train.groupby("cc_num").agg(
    avg_amount=("amt", "mean"),
    std_amount=("amt", "std"),
    avg_hour=("trans_date_trans_time", lambda x: x.dt.hour.mean()),
    avg_lat=("lat", "mean"),
    avg_long=("long", "mean")
).reset_index()

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
    train.sort_values("trans_date_trans_time")
    .groupby("cc_num")
    .tail(1)
    [["cc_num", "trans_date_trans_time", "amt",
      "merch_lat", "merch_long"]]
)

last.columns = [
    "cc_num", "previous_time", "previous_amount",
    "previous_lat", "previous_long"
]

d = test.copy()

d["hour"] = d["trans_date_trans_time"].dt.hour

d = d.merge(profile, on="cc_num", how="left")
d = d.merge(merchant, on=["cc_num", "merchant"], how="left")
d = d.merge(category, on=["cc_num", "category"], how="left")
d = d.merge(last, on="cc_num", how="left")

d["merchant_visits"] = d["merchant_visits"].fillna(0)
d["category_visits"] = d["category_visits"].fillna(0)

d["new_merchant"] = (d["merchant_visits"] == 0).astype(int)
d["new_category"] = (d["category_visits"] == 0).astype(int)

# Amount
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

# Time
hd = (d["hour"] - d["avg_hour"]).abs()

d["hour_deviation"] = np.minimum(
    hd,
    24 - hd
)

# Location
d["location_deviation"] = np.sqrt(
    (d["merch_lat"] - d["avg_lat"]) ** 2 +
    (d["merch_long"] - d["avg_long"]) ** 2
)

# Previous transaction
d["current_previous_time"] = (
    d.groupby("cc_num")["trans_date_trans_time"].shift(1)
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

# Velocity
d["seconds_since_previous"] = (
    d["trans_date_trans_time"] - d["prev_time"]
).dt.total_seconds()

d["seconds_since_previous"] = (
    d["seconds_since_previous"]
    .fillna(999999)
)

d["high_velocity"] = (
    d["seconds_since_previous"] < 300
).astype(int)

# Amount change
d["amount_change"] = (
    d["amt"] / (d["prev_amount"] + 1e-6)
)

d["amount_change"] = (
    d["amount_change"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
)

d["log_amount_change"] = np.log1p(
    d["amount_change"].clip(upper=20)
)

# Location change
d["location_change"] = np.sqrt(
    (d["merch_lat"] - d["prev_lat"]) ** 2 +
    (d["merch_long"] - d["prev_long"]) ** 2
)

d["log_location_change"] = np.log1p(
    d["location_change"]
)

# Interactions
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
)

d["combined_behavior_anomaly"] = (
    d["new_merchant"] +
    (d["amount_deviation"] > 2).astype(int) +
    (d["location_deviation"] > 0.8).astype(int) +
    d["high_velocity"] +
    (d["amount_change"] > 5).astype(int)
)

# ==================================================
# ATTACH FINAL MODEL RESULTS
# ==================================================

results = pd.read_csv(
    "data/processed/final_hybrid_results.csv"
)

results["trans_date_trans_time"] = pd.to_datetime(
    results["trans_date_trans_time"]
)

d = d.merge(
    results[
        [
            "cc_num",
            "trans_date_trans_time",
            "fraud_probability",
            "behavior_probability",
            "hybrid_probability",
            "decision"
        ]
    ],
    on=["cc_num", "trans_date_trans_time"],
    how="left"
)

# ==================================================
# CLASSIFY ERRORS
# ==================================================

d["predicted_fraud"] = (
    d["hybrid_probability"] >= 0.51
).astype(int)

false_positives = d[
    (d["is_fraud"] == 0) &
    (d["predicted_fraud"] == 1)
].copy()

false_negatives = d[
    (d["is_fraud"] == 1) &
    (d["predicted_fraud"] == 0)
].copy()

true_positives = d[
    (d["is_fraud"] == 1) &
    (d["predicted_fraud"] == 1)
].copy()

true_negatives = d[
    (d["is_fraud"] == 0) &
    (d["predicted_fraud"] == 0)
].copy()

print("\n========================================")
print("MODEL ERROR SUMMARY")
print("========================================")
print(f"False positives : {len(false_positives):,}")
print(f"False negatives : {len(false_negatives):,}")
print(f"True positives  : {len(true_positives):,}")
print(f"True negatives  : {len(true_negatives):,}")

features = [
    "amt",
    "fraud_probability",
    "behavior_probability",
    "hybrid_probability",
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

# ==================================================
# MAIN COMPARISON
# ==================================================

comparison = pd.DataFrame({
    "FALSE_POSITIVE": false_positives[features].mean(numeric_only=True),
    "FALSE_NEGATIVE": false_negatives[features].mean(numeric_only=True),
    "TRUE_POSITIVE": true_positives[features].mean(numeric_only=True),
    "TRUE_NEGATIVE": true_negatives[features].mean(numeric_only=True)
})

print("\n========================================")
print("FEATURE COMPARISON")
print("========================================")
print(comparison.round(3).to_string())

# ==================================================
# MEDIANS
# ==================================================

print("\n========================================")
print("MEDIAN COMPARISON")
print("========================================")

medians = pd.DataFrame({
    "FALSE_POSITIVE": false_positives[features].median(numeric_only=True),
    "FALSE_NEGATIVE": false_negatives[features].median(numeric_only=True),
    "TRUE_POSITIVE": true_positives[features].median(numeric_only=True),
    "TRUE_NEGATIVE": true_negatives[features].median(numeric_only=True)
})

print(medians.round(3).to_string())

# ==================================================
# BEHAVIOR FLAGS
# ==================================================

flags = [
    "new_merchant",
    "new_category",
    "high_velocity",
    "merchant_location_anomaly",
    "amount_location_anomaly",
    "amount_velocity_anomaly"
]

print("\n========================================")
print("BEHAVIOR FLAG RATES")
print("========================================")

for col in flags:
    print(f"\n{col}")
    print(
        pd.DataFrame({
            "FALSE_POSITIVE": [
                false_positives[col].mean()
            ],
            "FALSE_NEGATIVE": [
                false_negatives[col].mean()
            ],
            "TRUE_POSITIVE": [
                true_positives[col].mean()
            ],
            "TRUE_NEGATIVE": [
                true_negatives[col].mean()
            ]
        }).round(3).to_string(index=False)
    )

# ==================================================
# RISK BANDS
# ==================================================

d["risk_band"] = pd.cut(
    d["hybrid_probability"],
    bins=[-np.inf, 0.306, 0.51, np.inf],
    labels=["LOW", "MEDIUM", "HIGH"]
)

print("\n========================================")
print("RISK BAND BREAKDOWN")
print("========================================")

risk = d.groupby(
    ["risk_band", "is_fraud"],
    observed=False
).size().unstack(fill_value=0)

print(risk)

print("\nFraud rate within each risk band:")

for band in ["LOW", "MEDIUM", "HIGH"]:
    subset = d[d["risk_band"] == band]

    if len(subset):
        print(
            f"{band}: "
            f"{subset['is_fraud'].sum():,} fraud / "
            f"{len(subset):,} transactions = "
            f"{subset['is_fraud'].mean()*100:.2f}% fraud"
        )

# ==================================================
# SAVE FOR FURTHER ANALYSIS
# ==================================================

false_positives.to_csv(
    "results/false_positives.csv",
    index=False
)

false_negatives.to_csv(
    "results/false_negatives.csv",
    index=False
)

comparison.to_csv(
    "results/error_feature_comparison.csv"
)

print("\n========================================")
print("SAVED")
print("========================================")
print("results/false_positives.csv")
print("results/false_negatives.csv")
print("results/error_feature_comparison.csv")

# ==================================================
# BEHAVIOR COMBINATION ANALYSIS
# ==================================================

print("\n========================================")
print("BEHAVIOR COMBINATION ANALYSIS")
print("========================================")

# Create simple behavioral flags
d["large_amount"] = (
    d["amount_deviation"] > 2
).astype(int)

d["unusual_time"] = (
    d["hour_deviation"] > 6
).astype(int)

d["unusual_location"] = (
    d["location_deviation"] > 0.8
).astype(int)

d["new_merchant_flag"] = d["new_merchant"]

d["fast_transaction"] = d["high_velocity"]

# --------------------------------------------------
# Analyze combinations
# --------------------------------------------------

combinations = [
    ("large_amount", "unusual_location"),
    ("large_amount", "new_merchant_flag"),
    ("large_amount", "unusual_time"),
    ("large_amount", "fast_transaction"),
    ("new_merchant_flag", "unusual_location"),
    ("new_merchant_flag", "unusual_time"),
    ("unusual_location", "unusual_time"),
    ("unusual_time", "fast_transaction"),
]

for a, b in combinations:

    temp = d[
        (d[a] == 1) &
        (d[b] == 1)
    ]

    if len(temp) == 0:
        continue

    fraud_rate = temp["is_fraud"].mean() * 100

    print(
        f"\n{a} + {b}"
    )
    print(
        f"Transactions : {len(temp):,}"
    )
    print(
        f"Fraud        : {temp['is_fraud'].sum():,}"
    )
    print(
        f"Fraud rate   : {fraud_rate:.2f}%"
    )

# ==================================================
# BEHAVIOR SCORE / FRAUD RATE
# ==================================================

print("\n========================================")
print("COMBINED BEHAVIOR ANOMALY")
print("========================================")

behavior_summary = (
    d.groupby("combined_behavior_anomaly")
    .agg(
        transactions=("is_fraud", "size"),
        frauds=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean"),
        avg_hybrid=("hybrid_probability", "mean")
    )
    .reset_index()
)

behavior_summary["fraud_rate"] *= 100

print(
    behavior_summary.round(3).to_string(index=False)
)

# ==================================================
# HYBRID SCORE BANDS
# ==================================================

print("\n========================================")
print("HYBRID SCORE BANDS")
print("========================================")

d["score_band"] = pd.cut(
    d["hybrid_probability"],
    bins=[
        -np.inf,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        np.inf
    ]
)

score_summary = (
    d.groupby("score_band", observed=False)
    .agg(
        transactions=("is_fraud", "size"),
        frauds=("is_fraud", "sum"),
        fraud_rate=("is_fraud", "mean")
    )
    .reset_index()
)

score_summary["fraud_rate"] *= 100

print(
    score_summary.round(3).to_string(index=False)
)

# ==================================================
# SAVE
# ==================================================

behavior_summary.to_csv(
    "results/behavior_combination_summary.csv",
    index=False
)

score_summary.to_csv(
    "results/hybrid_score_bands.csv",
    index=False
)

print("\nSaved:")
print("results/behavior_combination_summary.csv")
print("results/hybrid_score_bands.csv")

# ==================================================
# DECISION RULE SIMULATION
# ==================================================

print("\n========================================")
print("DECISION RULE SIMULATION")
print("========================================")

# Candidate rules
rules = {
    "Current: hybrid >= 0.51":
        d["hybrid_probability"] >= 0.51,

    "High ML >= 0.80":
        d["fraud_probability"] >= 0.80,

    "ML >= 0.70 + behavior >= 0.50":
        (d["fraud_probability"] >= 0.70) &
        (d["behavior_probability"] >= 0.50),

    "ML >= 0.70 + 2+ behavior anomalies":
        (d["fraud_probability"] >= 0.70) &
        (d["combined_behavior_anomaly"] >= 2),

    "ML >= 0.75 + unusual time":
        (d["fraud_probability"] >= 0.75) &
        (d["hour_deviation"] > 6),

    "ML >= 0.75 + amount/location":
        (d["fraud_probability"] >= 0.75) &
        (d["amount_location_anomaly"] == 1),

    "ML >= 0.75 + strong behavior":
        (d["fraud_probability"] >= 0.75) &
        (
            (d["amount_location_anomaly"] == 1) |
            (d["amount_velocity_anomaly"] == 1) |
            (
                (d["hour_deviation"] > 6) &
                (d["amount_deviation"] > 2)
            )
        ),

    "Very high ML >= 0.90":
        d["fraud_probability"] >= 0.90,
}

rows = []

for name, block in rules.items():

    actual = d["is_fraud"]

    tp = ((block == 1) & (actual == 1)).sum()
    fp = ((block == 1) & (actual == 0)).sum()
    fn = ((block == 0) & (actual == 1)).sum()

    blocked = block.sum()

    precision = tp / blocked if blocked else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    rows.append({
        "Rule": name,
        "Blocked": int(blocked),
        "Fraud_Caught": int(tp),
        "Legitimate_Blocked": int(fp),
        "Fraud_Missed": int(fn),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    })

simulation = pd.DataFrame(rows)

print(
    simulation.to_string(index=False)
)

simulation.to_csv(
    "results/decision_rule_simulation.csv",
    index=False
)

print("\nSaved:")
print("results/decision_rule_simulation.csv")
