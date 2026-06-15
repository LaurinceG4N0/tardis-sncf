import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

df = pd.read_csv("cleaned_dataset.csv", sep=";")

# Tes features exactes du model_metadata.json
FEATURES = [
    "Service", "Departure station", "Arrival station",
    "Average journey time", "Number of scheduled trains",
    "Number of cancelled trains", "Average delay of all trains at departure",
    "Pct delay due to external causes", "Pct delay due to infrastructure",
    "Pct delay due to traffic management", "Pct delay due to rolling stock",
    "Pct delay due to station management and equipment reuse",
    "Pct delay due to passenger handling (crowding, disabled persons, connections)",
    "Year", "Month", "Delay categories",
    "month_sin", "month_cos", "season", "is_peak_month",
    "cancellation_rate", "pct_heavily_delayed", "stress_score",
]
TARGET = "Average delay of all trains at arrival"

CAT_COLS = ["Service", "Departure station", "Arrival station", "Delay categories"]
NUM_COLS = [f for f in FEATURES if f not in CAT_COLS]

X = df[FEATURES]
y = df[TARGET]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        subsample=0.7, n_estimators=300, min_samples_split=5,
        min_samples_leaf=1, max_depth=4, learning_rate=0.1,
        random_state=42,
    )),
])

pipeline.fit(X, y)
joblib.dump(pipeline, "model.joblib")
print("model.joblib regenerated successfully")
