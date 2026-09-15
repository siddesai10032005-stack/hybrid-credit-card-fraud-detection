import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/processed/behavior_train.csv")

features = [
    "amount_deviation","merchant_visits","new_merchant",
    "category_visits","new_category","hour_deviation",
    "location_deviation","seconds_since_previous",
    "high_velocity","log_amount_change",
    "log_location_change","merchant_location_anomaly",
    "amount_location_anomaly","amount_velocity_anomaly",
    "combined_behavior_anomaly"
]

X = df[features].replace([float("inf"),float("-inf")],0).fillna(0)
y = df["is_fraud"]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

print("Training behavioral model...")
model.fit(X,y)

joblib.dump(model,"models/behavior_model_v3.pkl")

print("Saved models/behavior_model_v3.pkl")

imp = pd.DataFrame({
    "feature":features,
    "importance":model.feature_importances_
}).sort_values("importance",ascending=False)

print(imp.to_string(index=False))
