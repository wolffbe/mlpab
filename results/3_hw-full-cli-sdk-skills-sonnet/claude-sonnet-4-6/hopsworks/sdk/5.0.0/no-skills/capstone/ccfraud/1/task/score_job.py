"""Runs ON the Hopsworks platform.
Loads model ccmodel89f322, reads cctxnscore89f322, writes ccpred89f322.
"""
import os
import pickle
import numpy as np
import pandas as pd
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load model
hw_model = mr.get_model("ccmodel89f322", version=1)
model_dir = hw_model.download()
with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(model_dir, "feature_cols.txt")) as f:
    feat_cols = [line.strip() for line in f if line.strip()]
print(f"Model loaded, features: {feat_cols}")

# Read score features
fg_score = fs.get_feature_group("cctxnscore89f322", version=1)
score_df = fg_score.read()
print(f"Score rows: {len(score_df)}")

X_score = score_df[feat_cols].fillna(0).values
fraud_probs = model.predict_proba(X_score)[:, 1]

pred_df = pd.DataFrame({
    "transaction_id": score_df["transaction_id"].astype(str).values,
    "fraud_probability": fraud_probs.astype(float),
})
print(pred_df.describe())

fg_pred = fs.get_or_create_feature_group(
    name="ccpred89f322",
    version=1,
    primary_key=["transaction_id"],
    description="CC fraud predictions",
    online_enabled=True,
)
fg_pred.insert(pred_df, write_options={"wait_for_job": True})
print(f"ccpred89f322 written: {len(pred_df)} rows")
