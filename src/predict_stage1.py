import pandas as pd
import joblib

test = pd.read_csv("data/processed/test_features.csv")

features = [
    "amt","hour","day_of_week","day","month","is_weekend",
    "customer_avg_amt","amount_ratio",
    "customer_transaction_count","location_distance",
    "city_pop","category","gender"
]

model = joblib.load("models/random_forest_model.pkl")

test["fraud_probability"] = model.predict_proba(
    test[features]
)[:,1]

test.to_csv(
    "data/processed/test_predictions.csv",
    index=False
)

print(test["fraud_probability"].describe())
print("Saved Stage 1 predictions.")
