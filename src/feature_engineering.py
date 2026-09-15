import pandas as pd
import numpy as np

def make_features(path, output):
    df = pd.read_csv(path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    customer_avg = df.groupby("cc_num")["amt"].transform("mean")
    customer_count = df.groupby("cc_num").cumcount()

    df["customer_avg_amt"] = customer_avg
    df["amount_ratio"] = df["amt"] / (customer_avg + 1e-6)
    df["customer_transaction_count"] = customer_count

    df["location_distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    )

    cols = [
        "amt","hour","day_of_week","day","month","is_weekend",
        "customer_avg_amt","amount_ratio",
        "customer_transaction_count","location_distance",
        "city_pop","category","gender","is_fraud"
    ]

    df[cols].to_csv(output,index=False)
    print("Saved:",output,len(df))

make_features(
    "data/raw/fraudTrain.csv",
    "data/processed/train_features.csv"
)

make_features(
    "data/raw/fraudTest.csv",
    "data/processed/test_features.csv"
)
