#!/usr/bin/env python3
"""
Train a fraud detection model on the feature view cctfv444ca and register it as ccmodelc444ca.
"""

import hopsworks
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import joblib


def main():
    # Connect to Hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Get feature view and training data
    fv = fs.get_feature_view("cctfv444ca", version=1)
    td = fv.get_training_data(1)[0]

    # Split into train/test
    X = td.drop(["is_fraud"], axis=1)
    y = td["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC: {auc}")
    print(classification_report(y_test, model.predict(X_test)))

    # Register model
    mr = project.get_model_registry()
    model_dir = "ccmodelc444ca"
    joblib.dump(model, f"{model_dir}/model.pkl")
    
    metrics = {"roc_auc": auc}
    input_example = X_train.iloc[0:1].values.tolist()
    
    model_reg = mr.python.create_model(
        name="ccmodelc444ca",
        metrics=metrics,
        input_example=input_example,
        description="Fraud detection model"
    )
    model_reg.save(model_dir)


if __name__ == "__main__":
    main()