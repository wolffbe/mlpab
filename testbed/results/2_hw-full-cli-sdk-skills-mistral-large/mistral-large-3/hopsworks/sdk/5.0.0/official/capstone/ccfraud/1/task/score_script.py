import hopsworks
import pandas as pd
import numpy as np
import joblib

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load model
model = mr.get_model("ccmodelc444ca", version=1)
model_dir = model.download()
model = joblib.load(f"{model_dir}/model.pkl")

# Read scoring data
score_df = fs.get_feature_group("temp_score_transactions", version=1).read()

# Prepare features for scoring
feature_columns = [
    "transaction_velocity_1h", "transaction_velocity_24h", 
    "amount_mean_24h", "amount_std_24h", 
    "geo_distance", "time_since_last_tx", "amount"
]
X_score = score_df[feature_columns]

# Predict
score_df["fraud_probability"] = model.predict_proba(X_score)[:, 1]

# Write predictions to feature table
predictions_fg = fs.get_or_create_feature_group(
    name="ccpredc444ca",
    version=1,
    description="Fraud probability predictions for transactions",
    primary_key=["transaction_id"],
    event_time="datetime",
    online_enabled=True,
)

# Insert predictions
predictions_fg.insert(score_df[["transaction_id", "datetime", "fraud_probability"]], write_options={"wait_for_job": True})