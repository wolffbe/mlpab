#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Full FTI Pipeline
Feature Group: cctxn89f322
Feature View/Training Dataset: cctd89f322
Model: ccmodel89f322
Predictions: ccpred89f322
"""

import os
import json
import math
import numpy as np
import pandas as pd
import joblib
import hopsworks


def compute_features(df, card_stats=None, cat_freq=None, merch_freq=None):
    """Compute fraud detection features from raw transactions."""
    df = df.copy()

    # Datetime features
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['tx_hour'] = df['datetime'].dt.hour.astype(float)
    df['tx_day_of_week'] = df['datetime'].dt.dayofweek.astype(float)
    df['tx_hour_sin'] = np.sin(2 * math.pi * df['tx_hour'] / 24.0)
    df['tx_hour_cos'] = np.cos(2 * math.pi * df['tx_hour'] / 24.0)

    # Amount features
    df['amount'] = df['amount'].astype(float)
    df['amount_log'] = np.log1p(df['amount'])

    # Card-level statistics
    if card_stats is None:
        card_stats = (
            df.groupby('cc_num')
            .agg(
                card_avg_amount=('amount', 'mean'),
                card_std_amount=('amount', 'std'),
                card_tx_count=('transaction_id', 'count'),
                card_home_lat=('lat', 'mean'),
                card_home_long=('long', 'mean'),
            )
            .reset_index()
        )
        card_stats['card_std_amount'] = card_stats['card_std_amount'].fillna(0.0)

    merge_cols = ['cc_num', 'card_avg_amount', 'card_std_amount',
                  'card_tx_count', 'card_home_lat', 'card_home_long']
    df = df.merge(card_stats[merge_cols], on='cc_num', how='left')

    # Fill for unseen cards using global averages
    df['card_avg_amount'] = df['card_avg_amount'].fillna(card_stats['card_avg_amount'].mean())
    df['card_std_amount'] = df['card_std_amount'].fillna(card_stats['card_std_amount'].mean())
    df['card_tx_count'] = df['card_tx_count'].fillna(1.0)
    df['card_home_lat'] = df['card_home_lat'].fillna(df['lat'])
    df['card_home_long'] = df['card_home_long'].fillna(df['long'])

    # Geo distance from card home (Euclidean approximation in degrees)
    df['geo_dist'] = np.sqrt(
        (df['lat'].astype(float) - df['card_home_lat']) ** 2
        + (df['long'].astype(float) - df['card_home_long']) ** 2
    )

    # Amount relative to card history
    df['amount_vs_avg'] = df['amount'] / (df['card_avg_amount'] + 1e-6)

    # Categorical frequency encoding (model-independent proxy for risk)
    if cat_freq is None:
        cat_freq = df['category'].value_counts(normalize=True).to_dict()
    if merch_freq is None:
        merch_freq = df['merchant'].value_counts(normalize=True).to_dict()

    df['category_freq'] = df['category'].map(cat_freq).fillna(0.0)
    df['merchant_freq'] = df['merchant'].map(merch_freq).fillna(0.0)

    return df, card_stats, cat_freq, merch_freq


FEATURE_COLS = [
    'transaction_id', 'cc_num', 'datetime',
    'amount', 'amount_log',
    'tx_hour', 'tx_day_of_week', 'tx_hour_sin', 'tx_hour_cos',
    'card_avg_amount', 'card_std_amount', 'card_tx_count',
    'card_home_lat', 'card_home_long', 'geo_dist',
    'amount_vs_avg', 'category_freq', 'merchant_freq',
    'is_fraud',
]

MODEL_FEATURES = [c for c in FEATURE_COLS
                  if c not in ('transaction_id', 'cc_num', 'datetime', 'is_fraud')]


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    # ── Download data from HopsFS ──────────────────────────────────────────────
    print("Downloading data from HopsFS …")
    dataset_api.download(
        "Resources/ccfraud/transactions.csv",
        local_path="/tmp/transactions.csv",
        overwrite=True,
    )
    dataset_api.download(
        "Resources/ccfraud/score_transactions.csv",
        local_path="/tmp/score_transactions.csv",
        overwrite=True,
    )

    df_train = pd.read_csv("/tmp/transactions.csv")
    df_score = pd.read_csv("/tmp/score_transactions.csv")
    print(f"Training: {len(df_train)} rows | Scoring: {len(df_score)} rows")

    # ── Feature Engineering ────────────────────────────────────────────────────
    df_feat, card_stats, cat_freq, merch_freq = compute_features(df_train)

    fg_df = df_feat[FEATURE_COLS].copy()
    fg_df['is_fraud'] = fg_df['is_fraud'].astype(int)
    # Keep datetime as pandas Timestamp so Hopsworks infers TIMESTAMP type (not string)

    # ── Feature Group ──────────────────────────────────────────────────────────
    print("Creating feature group cctxn89f322 …")
    fg = fs.get_or_create_feature_group(
        name="cctxn89f322",
        version=1,
        primary_key=["transaction_id"],
        event_time="datetime",
        online_enabled=False,
        description="Credit card transaction fraud features (velocity, geo, amount signals)",
    )
    fg.insert(fg_df, write_options={"wait_for_job": True})
    print("Feature group ready")

    # ── Feature View ──────────────────────────────────────────────────────────
    print("Creating feature view cctd89f322 …")
    fv = None
    try:
        fv = fs.get_feature_view(name="cctd89f322", version=1)
        if fv is None:
            raise ValueError("get_feature_view returned None")
        print("Feature view already exists")
    except Exception as e:
        print(f"Feature view get failed ({e}), creating …")
        query = fg.select_except(["transaction_id", "cc_num", "datetime"])
        fv = fs.create_feature_view(
            name="cctd89f322",
            version=1,
            description="Credit card fraud detection feature view",
            labels=["is_fraud"],
            query=query,
        )
        print("Feature view created")
    assert fv is not None, "Feature view is None after create — aborting"

    # ── Training Dataset ───────────────────────────────────────────────────────
    print("Splitting training data …")
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    y_tr = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
    y_te = y_test.values.ravel() if hasattr(y_test, 'values') else y_test

    neg = int((y_tr == 0).sum())
    pos = int((y_tr == 1).sum())
    spw = neg / max(pos, 1)
    print(f"Class balance — neg: {neg}, pos: {pos}, scale_pos_weight: {spw:.2f}")

    # ── Train Model ────────────────────────────────────────────────────────────
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=spw,
        random_state=42,
        tree_method='hist',
    )
    model.fit(X_train, y_tr, eval_set=[(X_test, y_te)], verbose=False)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_te, y_pred_proba)
    print(f"ROC AUC (test): {auc:.4f}")

    metrics = {
        "roc_auc": float(auc),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "fraud_pct": float(pos / (neg + pos) * 100),
    }

    # ── Register Model ─────────────────────────────────────────────────────────
    model_dir = "/tmp/ccmodel89f322"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/model.pkl")

    feature_names = list(X_train.columns)
    with open(f"{model_dir}/feature_names.json", "w") as f:
        json.dump(feature_names, f)
    with open(f"{model_dir}/encodings.json", "w") as f:
        json.dump({"cat_freq": cat_freq, "merch_freq": merch_freq}, f)
    card_stats.to_csv(f"{model_dir}/card_stats.csv", index=False)

    mr = project.get_model_registry()
    hw_model = mr.python.create_model(
        name="ccmodel89f322",
        metrics=metrics,
        description=(
            "XGBoost classifier for credit card fraud detection. "
            "Features: geo distance from home, tx velocity, amount vs history, "
            "category/merchant frequency, hour/day-of-week signals."
        ),
        input_example=X_train.head(1),
        feature_view=fv,
    )
    hw_model.save(model_dir)
    print(f"Model registered: ccmodel89f322 (ROC AUC = {auc:.4f})")

    # ── Score Transactions ────────────────────────────────────────────────────
    print("Scoring …")
    df_sf, _, _, _ = compute_features(
        df_score,
        card_stats=card_stats,
        cat_freq=cat_freq,
        merch_freq=merch_freq,
    )

    # Ensure all model features are present
    for col in feature_names:
        if col not in df_sf.columns:
            df_sf[col] = 0.0

    X_score = df_sf[feature_names]
    fraud_proba = model.predict_proba(X_score)[:, 1]

    pred_df = pd.DataFrame({
        "transaction_id": df_score["transaction_id"].values,
        "fraud_probability": fraud_proba.tolist(),
    })

    # ── Predictions Feature Group ─────────────────────────────────────────────
    print("Writing predictions to ccpred89f322 …")
    pred_fg = fs.get_or_create_feature_group(
        name="ccpred89f322",
        version=1,
        primary_key=["transaction_id"],
        online_enabled=True,
        description="Credit card fraud predictions. fraud_probability in [0, 1].",
    )
    pred_fg.insert(pred_df, write_options={"wait_for_job": True})

    print(f"Done! {len(pred_df)} predictions written.")
    print(f"Predicted fraud rate (>0.5): {(fraud_proba > 0.5).mean():.2%}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
