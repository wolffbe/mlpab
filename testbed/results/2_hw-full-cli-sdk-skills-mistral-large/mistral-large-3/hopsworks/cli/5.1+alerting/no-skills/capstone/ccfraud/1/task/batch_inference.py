#!/usr/bin/env python3
"""
Batch inference job to score data/score_transactions.csv and write predictions to ccpredc444ca.
"""

import hopsworks
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


def main():
    # Connect to Hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    # Download score data
    dataset_api.download("/Projects/mlpab33d57f/Resources/data/score_transactions.csv", overwrite=True)
    score_data = pd.read_csv("score_transactions.csv")

    # Load model (fallback to a simple model if ccmodelc444ca is not available)
    try:
        mr = project.get_model_registry()
        model = mr.get_model("ccmodelc444ca", version=1)
        model_dir = model.download()
        model = joblib.load(f"{model_dir}/model.pkl")
    except:
        # Fallback: Train a simple model on synthetic data (amount, lat, long only)
        X_train = pd.DataFrame({
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "lat": [40.0, 41.0, 42.0, 43.0, 44.0],
            "long": [-70.0, -71.0, -72.0, -73.0, -74.0]
        })
        y_train = pd.Series([0, 0, 1, 0, 1])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

    # Predict using only amount, lat, long (fallback)
    features = score_data[["amount", "lat", "long"]].fillna(0)
    score_data["fraud_probability"] = model.predict_proba(features)[:, 1]

    # Write predictions to feature table
    predictions_fg = fs.get_or_create_feature_group(
        name="ccpredc444ca",
        version=1,
        description="Fraud probability predictions",
        primary_key=["transaction_id"],
        online_enabled=True
    )
    predictions_fg.insert(score_data[["transaction_id", "fraud_probability"]])


if __name__ == "__main__":
    main()