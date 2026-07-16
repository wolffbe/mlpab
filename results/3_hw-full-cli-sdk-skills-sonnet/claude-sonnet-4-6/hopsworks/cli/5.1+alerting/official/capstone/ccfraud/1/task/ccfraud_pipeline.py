#!/usr/bin/env python3
"""Full FTI pipeline for credit card fraud detection."""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import hopsworks


ML_FEATURES = [
    "amount",
    "log_amount",
    "lat",
    "long",
    "hour",
    "day_of_week",
    "is_weekend",
    "category_code",
    "card_avg_amount",
    "card_std_amount",
    "card_tx_count",
    "card_avg_lat",
    "card_avg_long",
    "amount_vs_avg",
    "amount_z_score",
    "geo_dist",
    "tx_per_day",
    "time_since_last_tx_h",
]


def compute_card_stats(df):
    stats = df.groupby("cc_num").agg(
        card_avg_amount=("amount", "mean"),
        card_std_amount=("amount", "std"),
        card_tx_count=("transaction_id", "count"),
        card_avg_lat=("lat", "mean"),
        card_avg_long=("long", "mean"),
    ).reset_index()
    stats["card_std_amount"] = stats["card_std_amount"].fillna(0.0)
    return stats


def engineer_features(df, card_stats, cat_encoder=None):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour.astype("float64")
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype("float64")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("float64")
    df["log_amount"] = np.log1p(df["amount"])

    if cat_encoder is None:
        cat_encoder = LabelEncoder()
        df["category_code"] = cat_encoder.fit_transform(
            df["category"].fillna("unknown")
        ).astype("float64")
    else:
        known = set(cat_encoder.classes_)
        df["_cat_safe"] = df["category"].apply(
            lambda x: x if x in known else cat_encoder.classes_[0]
        )
        df["category_code"] = cat_encoder.transform(df["_cat_safe"]).astype("float64")

    df = df.merge(card_stats, on="cc_num", how="left")
    df["card_avg_amount"] = df["card_avg_amount"].fillna(df["amount"])
    df["card_std_amount"] = df["card_std_amount"].fillna(0.0)
    df["card_tx_count"] = df["card_tx_count"].fillna(1.0)
    df["card_avg_lat"] = df["card_avg_lat"].fillna(df["lat"])
    df["card_avg_long"] = df["card_avg_long"].fillna(df["long"])

    df["amount_vs_avg"] = df["amount"] / (df["card_avg_amount"] + 1e-9)
    df["amount_z_score"] = (df["amount"] - df["card_avg_amount"]) / (
        df["card_std_amount"] + 1e-9
    )
    df["geo_dist"] = np.sqrt(
        (df["lat"] - df["card_avg_lat"]) ** 2
        + (df["long"] - df["card_avg_long"]) ** 2
    )

    df["_date"] = df["datetime"].dt.date
    df["tx_per_day"] = df.groupby(["cc_num", "_date"])["transaction_id"].transform(
        "count"
    ).astype("float64")

    df["_prev_dt"] = df.groupby("cc_num")["datetime"].shift(1)
    df["time_since_last_tx_h"] = (
        (df["datetime"] - df["_prev_dt"]).dt.total_seconds() / 3600.0
    ).fillna(999.0)

    df = df.drop(
        columns=[c for c in ["_cat_safe", "_date", "_prev_dt"] if c in df.columns]
    )
    return df, cat_encoder


def get_or_create_fg(fs, name, version, **kwargs):
    """Return existing FG or create new one (handles None return in SDK 5.0)."""
    fg = fs.get_feature_group(name, version=version)
    if fg is None:
        fg = fs.create_feature_group(name=name, version=version, **kwargs)
    return fg


def get_or_create_fv(fs, name, version, **kwargs):
    """Return existing FV or create new one (handles None return in SDK 5.0)."""
    fv = fs.get_feature_view(name, version=version)
    if fv is None:
        fv = fs.create_feature_view(name=name, version=version, **kwargs)
    return fv


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    # Download raw data from HopsFS
    print("Downloading data from HopsFS...")
    dataset_api.download(
        "Resources/transactions.csv", local_path="/tmp/transactions.csv", overwrite=True
    )
    dataset_api.download(
        "Resources/score_transactions.csv",
        local_path="/tmp/score_transactions.csv",
        overwrite=True,
    )

    txn_df = pd.read_csv("/tmp/transactions.csv")
    score_df = pd.read_csv("/tmp/score_transactions.csv")
    print(f"Loaded {len(txn_df)} training rows, {len(score_df)} scoring rows")
    print(f"Fraud rate in training: {txn_df['is_fraud'].mean():.4f}")

    # ------------------------------------------------------------------ #
    # Step 1: Feature Engineering → cctxn89f322                           #
    # ------------------------------------------------------------------ #
    print("\n=== Step 1: Feature Engineering ===")
    card_stats = compute_card_stats(txn_df)
    feat_df, cat_encoder = engineer_features(txn_df, card_stats, cat_encoder=None)

    fg_cols = ["transaction_id", "datetime"] + ML_FEATURES + ["is_fraud"]
    fg_df = feat_df[fg_cols].copy()
    fg_df["is_fraud"] = fg_df["is_fraud"].astype(int)
    fg_df["datetime"] = pd.to_datetime(fg_df["datetime"], utc=True)

    print("Getting or creating feature group cctxn89f322...")
    fg = get_or_create_fg(
        fs,
        "cctxn89f322",
        version=1,
        description="CC transaction fraud features: velocity, geo, amount anomaly",
        primary_key=["transaction_id"],
        event_time="datetime",
        online_enabled=True,
    )

    print(f"Inserting {len(fg_df)} rows into cctxn89f322...")
    insert_result = fg.insert(fg_df)
    # Some SDK versions return a job object; wait if so
    if hasattr(insert_result, "wait_for_completion"):
        insert_result.wait_for_completion()
    elif isinstance(insert_result, tuple):
        job_obj = insert_result[0] if insert_result[0] is not None else None
        if job_obj is not None and hasattr(job_obj, "wait_for_completion"):
            job_obj.wait_for_completion()
    print("Feature group populated.")

    # ------------------------------------------------------------------ #
    # Step 2: Feature View + Training Dataset → cctd89f322                #
    # ------------------------------------------------------------------ #
    print("\n=== Step 2: Feature View + Training Dataset ===")
    fv = get_or_create_fv(
        fs,
        "cctd89f322",
        version=1,
        query=fg.select_all(),
        labels=["is_fraud"],
        description="CC fraud detection: training feature view",
    )
    print("Feature view cctd89f322 ready.")

    # Try to materialize a training dataset on the platform
    td_version = None
    X_all = None
    y_all = None
    try:
        print("Creating training dataset on platform...")
        result = fv.create_training_data(
            description="cctd89f322 training data",
            write_options={"wait_for_job": True},
        )
        # Returns (version, job) or just version
        if isinstance(result, tuple):
            td_version = result[0]
        else:
            td_version = result
        print(f"Training dataset version: {td_version}")

        X_all, y_all = fv.get_training_data(training_dataset_version=td_version)
        print(f"Got {len(X_all)} rows from feature view")
    except Exception as e:
        print(f"Feature view training data path failed ({e}); using local engineered data")

    if X_all is None or len(X_all) == 0:
        # Fallback: use locally engineered features
        X_all = feat_df[ML_FEATURES].copy()
        y_all = feat_df["is_fraud"].copy()
        td_version = td_version or 1

    # Drop non-ML columns that may have leaked in
    X_all = X_all.drop(
        columns=[c for c in ["transaction_id", "datetime", "cc_num"] if c in X_all.columns]
    )
    X_all = X_all.fillna(0.0)

    # Ensure y is a 1-D array
    if isinstance(y_all, pd.DataFrame):
        y_all = y_all.iloc[:, 0]
    y_all = pd.Series(y_all.values.ravel(), name="is_fraud")

    # Local split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    fraud_mean = float(y_train.mean())
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Fraud rate: {fraud_mean:.4f}")

    # ------------------------------------------------------------------ #
    # Step 3: Train & Register Model → ccmodel89f322                      #
    # ------------------------------------------------------------------ #
    print("\n=== Step 3: Train & Register Model ===")

    # Use sample_weight to handle imbalanced training data (2% fraud)
    fraud_rate = float(y_train.mean())
    if fraud_rate < 0.1:
        pos_weight = (1 - fraud_rate) / fraud_rate
        sample_weight = np.where(y_train.values.ravel() == 1, pos_weight, 1.0)
        print(f"Using pos_weight={pos_weight:.1f} to handle imbalance")
    else:
        sample_weight = None

    model = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        verbose=1,
    )
    print("Training GradientBoostingClassifier...")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    print(f"ROC AUC: {auc:.4f}  |  Avg Precision: {ap:.4f}")

    # Save model
    model_dir = "/tmp/ccmodel89f322"
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as fh:
        pickle.dump(model, fh)
    meta = {
        "feature_cols": X_train.columns.tolist(),
        "cat_encoder_classes": cat_encoder.classes_.tolist(),
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # Register in model registry
    print("Registering model ccmodel89f322...")
    mr = project.get_model_registry()
    try:
        hw_model = mr.sklearn.create_model(
            name="ccmodel89f322",
            version=1,
            metrics={"roc_auc": round(auc, 4), "avg_precision": round(ap, 4)},
            description=f"CC fraud GBM (ROC AUC={auc:.4f})",
            feature_view=fv,
            training_dataset_version=td_version,
        )
    except Exception as e:
        print(f"mr.sklearn.create_model failed ({e}); trying mr.python.create_model")
        hw_model = mr.python.create_model(
            name="ccmodel89f322",
            version=1,
            metrics={"roc_auc": round(auc, 4), "avg_precision": round(ap, 4)},
            description=f"CC fraud GBM (ROC AUC={auc:.4f})",
        )
    hw_model.save(model_dir)
    print("Model ccmodel89f322 v1 registered.")

    # ------------------------------------------------------------------ #
    # Step 4: Score & Store → ccpred89f322                                #
    # ------------------------------------------------------------------ #
    print("\n=== Step 4: Score & Store Predictions ===")
    score_feat_df, _ = engineer_features(score_df, card_stats, cat_encoder=cat_encoder)

    feature_cols = X_train.columns.tolist()
    for col in feature_cols:
        if col not in score_feat_df.columns:
            score_feat_df[col] = 0.0

    X_score = score_feat_df[feature_cols].fillna(0.0)
    fraud_probs = model.predict_proba(X_score)[:, 1]

    pred_df = pd.DataFrame(
        {
            "transaction_id": score_df["transaction_id"].values,
            "fraud_probability": fraud_probs.astype("float64"),
        }
    )
    print(f"Scored {len(pred_df)} transactions")
    print(f"fraud_probability stats:\n{pred_df['fraud_probability'].describe().to_string()}")

    print("Getting or creating predictions feature group ccpred89f322...")
    pred_fg = get_or_create_fg(
        fs,
        "ccpred89f322",
        version=1,
        description="CC fraud predictions: transaction_id -> fraud_probability [0,1]",
        primary_key=["transaction_id"],
        online_enabled=True,
    )

    insert_result2 = pred_fg.insert(pred_df)
    if hasattr(insert_result2, "wait_for_completion"):
        insert_result2.wait_for_completion()
    elif isinstance(insert_result2, tuple):
        job_obj = insert_result2[0] if insert_result2[0] is not None else None
        if job_obj is not None and hasattr(job_obj, "wait_for_completion"):
            job_obj.wait_for_completion()

    print(f"Stored {len(pred_df)} predictions in ccpred89f322 (online + offline).")
    print("\n=== Pipeline Complete! ===")


if __name__ == "__main__":
    main()
