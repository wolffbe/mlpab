"""Full FTI pipeline for credit-card fraud detection, run as a Hopsworks job.

Feature engineering -> feature group cctxn690a32 -> feature view/training
dataset cctd690a32 -> sklearn classifier ccmodel690a32 (with metrics) ->
predictions feature group ccpred690a32 (online + offline).
"""
import os
import joblib
import numpy as np
import pandas as pd

import hopsworks

FG_NAME = "cctxn690a32"
FV_NAME = "cctd690a32"
MODEL_NAME = "ccmodel690a32"
PRED_FG_NAME = "ccpred690a32"

EARTH_R = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


CATEGORIES = [
    "cash_advance", "clothing", "electronics", "entertainment", "fuel",
    "grocery", "health", "online", "restaurant", "travel",
]


def base_clean(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["amount"] = df["amount"].astype(float)
    df["lat"] = df["lat"].astype(float)
    df["long"] = df["long"].astype(float)
    df["cc_num"] = df["cc_num"].astype(str)
    return df


def engineer(df, card_stats):
    """Add engineered features. card_stats is a per-card aggregate frame
    computed from the training history (so it transfers to the score set)."""
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["log_amount"] = np.log1p(df["amount"])

    df = df.merge(card_stats, on="cc_num", how="left")
    # global fallbacks for any unseen card
    gmean = card_stats["card_amt_mean"].mean()
    gstd = card_stats["card_amt_std"].mean()
    glat = card_stats["card_lat"].mean()
    glong = card_stats["card_long"].mean()
    df["card_amt_mean"] = df["card_amt_mean"].fillna(gmean)
    df["card_amt_std"] = df["card_amt_std"].fillna(gstd).replace(0, gstd)
    df["card_lat"] = df["card_lat"].fillna(glat)
    df["card_long"] = df["card_long"].fillna(glong)

    df["amt_dev"] = (df["amount"] - df["card_amt_mean"]) / (df["card_amt_std"] + 1.0)
    df["amt_ratio"] = df["amount"] / (df["card_amt_mean"] + 1.0)
    df["geo_dist"] = haversine(df["lat"], df["long"], df["card_lat"], df["card_long"])

    # velocity features within each dataset, per card
    dt_prev = df.groupby("cc_num")["datetime"].diff().dt.total_seconds()
    df["secs_since_prev"] = dt_prev.fillna(7 * 24 * 3600.0).clip(upper=7 * 24 * 3600.0)
    df["txn_rate_1h"] = (df["secs_since_prev"] <= 3600).astype(int)

    # category one-hot
    for c in CATEGORIES:
        df[f"cat_{c}"] = (df["category"] == c).astype(int)

    return df


FEATURES = [
    "amount", "log_amount", "hour", "dayofweek", "is_night",
    "amt_dev", "amt_ratio", "geo_dist", "card_amt_mean", "card_amt_std",
    "secs_since_prev", "txn_rate_1h",
] + [f"cat_{c}" for c in CATEGORIES]


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dsapi = project.get_dataset_api()

    print("Downloading input CSVs from HopsFS ...", flush=True)
    for fn in ["transactions.csv", "score_transactions.csv"]:
        if os.path.exists(fn):
            os.remove(fn)
        dsapi.download(f"Resources/ccfraud/{fn}", local_path=".", overwrite=True)

    train_raw = base_clean(pd.read_csv("transactions.csv"))
    score_raw = base_clean(pd.read_csv("score_transactions.csv"))
    print(f"train rows={len(train_raw)} score rows={len(score_raw)}", flush=True)

    # per-card aggregates from training history (transfer to score set)
    card_stats = (
        train_raw.groupby("cc_num")
        .agg(card_amt_mean=("amount", "mean"),
             card_amt_std=("amount", "std"),
             card_lat=("lat", "median"),
             card_long=("long", "median"))
        .reset_index()
    )
    card_stats["card_amt_std"] = card_stats["card_amt_std"].fillna(0.0)

    train_fe = engineer(train_raw, card_stats)
    train_fe["is_fraud"] = train_raw.set_index("transaction_id")["is_fraud"].reindex(
        train_fe["transaction_id"]).values
    train_fe["is_fraud"] = train_fe["is_fraud"].astype(int)

    # ---- Feature group cctxn690a32 (engineered features + label) ----
    fg_cols = ["transaction_id", "cc_num", "datetime"] + FEATURES + ["is_fraud"]
    fg_df = train_fe[fg_cols].copy()
    print("Creating feature group", FG_NAME, flush=True)
    fg = fs.get_or_create_feature_group(
        name=FG_NAME, version=1,
        description="Engineered credit-card fraud features with label",
        primary_key=["transaction_id"], event_time="datetime",
        online_enabled=False,
    )
    fg.insert(fg_df, write_options={"wait_for_job": True})
    print("Feature group inserted", flush=True)

    # ---- Feature view / training dataset cctd690a32 ----
    print("Creating feature view", FV_NAME, flush=True)
    try:
        existing = fs.get_feature_view(name=FV_NAME, version=1)
        existing.delete()
    except Exception as e:
        print("no existing fv:", e, flush=True)
    query = fg.select_except(["transaction_id", "cc_num", "datetime"])
    fv = fs.create_feature_view(
        name=FV_NAME, version=1, query=query, labels=["is_fraud"],
        description="Training dataset feature view for fraud classifier",
    )
    td_version, _ = fv.create_train_test_split(
        test_size=0.2, description="fraud training dataset",
        write_options={"wait_for_job": True},
    )
    print("Training dataset version:", td_version, flush=True)

    # ---- Train classifier in-memory (platform-side) ----
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X = train_fe[FEATURES]
    y = train_fe["is_fraud"]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, class_weight="balanced", random_state=42,
    )
    clf.fit(Xtr, ytr)
    proba_te = clf.predict_proba(Xte)[:, 1]
    test_auc = float(roc_auc_score(yte, proba_te))
    print(f"Held-out test ROC AUC = {test_auc:.4f}", flush=True)

    # refit on all training data for final scoring
    clf.fit(X, y)

    # ---- Register model ccmodel690a32 with metrics ----
    mr = project.get_model_registry()
    mdir = "ccmodel_artifact"
    os.makedirs(mdir, exist_ok=True)
    joblib.dump(clf, os.path.join(mdir, "model.pkl"))
    input_example = X.iloc[:1].to_dict(orient="records")[0]
    metrics = {"roc_auc": test_auc}
    print("Registering model", MODEL_NAME, flush=True)
    try:
        model = mr.sklearn.create_model(
            name=MODEL_NAME, metrics=metrics,
            description="Credit-card fraud classifier (HistGradientBoosting)",
            input_example=input_example, feature_view=fv,
        )
    except Exception as e:
        print("sklearn.create_model failed, falling back to python:", e, flush=True)
        model = mr.python.create_model(
            name=MODEL_NAME, metrics=metrics,
            description="Credit-card fraud classifier (HistGradientBoosting)",
            input_example=input_example,
        )
    model.save(mdir)
    print("Model registered with metrics", metrics, flush=True)

    # ---- Score score_transactions.csv ----
    score_fe = engineer(score_raw, card_stats)
    score_proba = clf.predict_proba(score_fe[FEATURES])[:, 1]
    pred_df = pd.DataFrame({
        "transaction_id": score_fe["transaction_id"].values,
        "fraud_probability": np.clip(score_proba, 0.0, 1.0).astype(float),
    })
    print("Scored rows:", len(pred_df), "prob range",
          float(pred_df.fraud_probability.min()),
          float(pred_df.fraud_probability.max()), flush=True)

    # ---- Predictions feature group ccpred690a32 (online + offline) ----
    print("Creating predictions feature group", PRED_FG_NAME, flush=True)
    pred_fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME, version=1,
        description="Fraud probability predictions for scoring set",
        primary_key=["transaction_id"], online_enabled=True,
    )
    pred_fg.insert(pred_df, write_options={"wait_for_job": True})
    print("Predictions written. DONE.", flush=True)


if __name__ == "__main__":
    main()
