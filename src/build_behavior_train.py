import pandas as pd
import numpy as np

print("Loading train...")

df = pd.read_csv("data/raw/fraudTrain.csv")
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df = df.sort_values("trans_date_trans_time").reset_index(drop=True)

split = int(len(df)*0.8)

history = df.iloc[:split].copy()
current = df.iloc[split:].copy()

print("History:",len(history))
print("Behavior training:",len(current))

profile = history.groupby("cc_num").agg(
    avg_amount=("amt","mean"),
    std_amount=("amt","std"),
    avg_hour=("trans_date_trans_time",lambda x:x.dt.hour.mean()),
    avg_lat=("lat","mean"),
    avg_long=("long","mean")
).reset_index()

profile["std_amount"] = profile["std_amount"].fillna(0)

merchant = history.groupby(
    ["cc_num","merchant"]
).size().reset_index(name="merchant_visits")

category = history.groupby(
    ["cc_num","category"]
).size().reset_index(name="category_visits")

last = (
    history.sort_values("trans_date_trans_time")
    .groupby("cc_num")
    .tail(1)
    [["cc_num","trans_date_trans_time","amt","merch_lat","merch_long"]]
)

last.columns = [
    "cc_num","previous_time","previous_amount",
    "previous_lat","previous_long"
]

d = current.copy()
d["hour"] = d["trans_date_trans_time"].dt.hour

d = d.merge(profile,on="cc_num",how="left")
d = d.merge(merchant,on=["cc_num","merchant"],how="left")
d = d.merge(category,on=["cc_num","category"],how="left")
d = d.merge(last,on="cc_num",how="left")

d["merchant_visits"] = d["merchant_visits"].fillna(0)
d["category_visits"] = d["category_visits"].fillna(0)

d["new_merchant"] = (d["merchant_visits"]==0).astype(int)
d["new_category"] = (d["category_visits"]==0).astype(int)

d["amount_z"] = (
    (d["amt"]-d["avg_amount"]) /
    (d["std_amount"]+1e-6)
)

d["amount_deviation"] = d["amount_z"].abs()

hd = (d["hour"]-d["avg_hour"]).abs()
d["hour_deviation"] = np.minimum(hd,24-hd)

d["location_deviation"] = np.sqrt(
    (d["merch_lat"]-d["avg_lat"])**2+
    (d["merch_long"]-d["avg_long"])**2
)

d = d.sort_values(["cc_num","trans_date_trans_time"])

d["prev_time_current"] = (
    d.groupby("cc_num")["trans_date_trans_time"].shift(1)
)

d["prev_amount_current"] = (
    d.groupby("cc_num")["amt"].shift(1)
)

d["prev_lat_current"] = (
    d.groupby("cc_num")["merch_lat"].shift(1)
)

d["prev_long_current"] = (
    d.groupby("cc_num")["merch_long"].shift(1)
)

d["prev_time"] = d["prev_time_current"].fillna(d["previous_time"])
d["prev_amount"] = d["prev_amount_current"].fillna(d["previous_amount"])
d["prev_lat"] = d["prev_lat_current"].fillna(d["previous_lat"])
d["prev_long"] = d["prev_long_current"].fillna(d["previous_long"])

d["seconds_since_previous"] = (
    d["trans_date_trans_time"]-d["prev_time"]
).dt.total_seconds().fillna(999999)

d["high_velocity"] = (
    d["seconds_since_previous"]<300
).astype(int)

d["amount_change"] = (
    d["amt"]/(d["prev_amount"]+1e-6)
).replace([np.inf,-np.inf],0).fillna(0)

d["log_amount_change"] = np.log1p(
    d["amount_change"].clip(upper=20)
)

d["location_change"] = np.sqrt(
    (d["merch_lat"]-d["prev_lat"])**2+
    (d["merch_long"]-d["prev_long"])**2
)

d["log_location_change"] = np.log1p(d["location_change"])

d["merchant_location_anomaly"] = (
    d["new_merchant"]*
    (d["location_deviation"]>0.8)
).astype(int)

d["amount_location_anomaly"] = (
    (d["amount_deviation"]>2)*
    (d["location_deviation"]>0.8)
).astype(int)

d["amount_velocity_anomaly"] = (
    (d["amount_deviation"]>2)*
    d["high_velocity"]
)

d["combined_behavior_anomaly"] = (
    d["new_merchant"]+
    (d["amount_deviation"]>2).astype(int)+
    (d["location_deviation"]>0.8).astype(int)+
    d["high_velocity"]+
    (d["amount_change"]>5).astype(int)
)

features = [
    "amount_deviation","merchant_visits","new_merchant",
    "category_visits","new_category","hour_deviation",
    "location_deviation","seconds_since_previous",
    "high_velocity","log_amount_change",
    "log_location_change","merchant_location_anomaly",
    "amount_location_anomaly","amount_velocity_anomaly",
    "combined_behavior_anomaly","is_fraud"
]

d[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_csv(
    "data/processed/behavior_train.csv",
    index=False
)

print("Saved behavior_train.csv")
