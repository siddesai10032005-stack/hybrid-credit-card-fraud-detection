import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("data/processed/train_features.csv")

features = [
    "amt","hour","day_of_week","day","month","is_weekend",
    "customer_avg_amt","amount_ratio",
    "customer_transaction_count","location_distance",
    "city_pop","category","gender"
]

X = train[features]
y = train["is_fraud"]

numeric = [
    "amt","hour","day_of_week","day","month","is_weekend",
    "customer_avg_amt","amount_ratio",
    "customer_transaction_count","location_distance",
    "city_pop"
]

categorical = ["category","gender"]

preprocessor = ColumnTransformer([
    ("num",StandardScaler(),numeric),
    ("cat",OneHotEncoder(handle_unknown="ignore"),categorical)
])

lr = Pipeline([
    ("prep",preprocessor),
    ("model",LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1
    ))
])

rf = Pipeline([
    ("prep",preprocessor),
    ("model",RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ))
])

print("Training Logistic Regression...")
lr.fit(X,y)

print("Training Random Forest...")
rf.fit(X,y)

joblib.dump(lr,"models/logistic_model.pkl")
joblib.dump(rf,"models/random_forest_model.pkl")

print("Stage 1 models saved.")
