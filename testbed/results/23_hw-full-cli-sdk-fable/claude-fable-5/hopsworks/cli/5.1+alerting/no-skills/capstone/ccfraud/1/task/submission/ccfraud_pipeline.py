"""Credit-card fraud FTI pipeline — runs as a Hopsworks job.

Stages (sys.argv[1]):
  features — engineer fraud features from raw CSVs into feature group cctxn4b8521
  train    — read training dataset from feature view cctd4b8521, train and
             register classifier ccmodel4b8521 with metrics
  score    — score score_transactions.csv with the registered model and write
             predictions to online-enabled feature group ccpred4b8521
"""
import os
import sys

import numpy as np
import pandas as pd
import hopsworks

FG_NAME = "cctxn4b8521"
FV_NAME = "cctd4b8521"
MODEL_NAME = "ccmodel4b8521"
PRED_FG_NAME = "ccpred4b8521"
DATA_DIR = "Resources/ccfraud"

CATEGORIES = [
    "cash_advance", "clothing", "electronics", "entertainment", "fuel",
    "grocery", "health", "online", "restaurant", "travel",
]

BASE_FEATURES = [
    "amount", "log_amount", "hour", "dow", "is_night", "mins_since_prev",
    "txn_1h", "txn_24h", "txn_7d", "dist_prev_km", "speed_kmh",
    "card_amt_mean", "card_amt_std", "amt_z", "amt_over_mean",
    "dist_home_km", "lat", "long",
]
FEATURES = BASE_FEATURES + [f"cat_{c}" for c in CATEGORIES]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.radians(np.asarray(lat1, dtype=float))
    p2 = np.radians(np.asarray(lat2, dtype=float))
    dl = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_raw(project):
    ds = project.get_dataset_api()
    hist_path = ds.download(f"{DATA_DIR}/transactions.csv", overwrite=True)
    score_path = ds.download(f"{DATA_DIR}/score_transactions.csv", overwrite=True)
    hist = pd.read_csv(hist_path)
    score = pd.read_csv(score_path)
    for df in (hist, score):
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return hist, score


def engineer(hist, score):
    """Causal per-card features over the combined timeline; returns (hist_df, score_df)."""
    hist = hist.copy()
    score = score.copy()
    hist["_split"] = "hist"
    score["_split"] = "score"
    score["is_fraud"] = -1
    df = pd.concat([hist, score], ignore_index=True)
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

    home = (
        hist.groupby("cc_num")[["lat", "long"]]
        .median()
        .rename(columns={"lat": "home_lat", "long": "home_long"})
        .reset_index()
    )
    df = df.merge(home, on="cc_num", how="left")
    df["home_lat"] = df["home_lat"].fillna(df["lat"])
    df["home_long"] = df["home_long"].fillna(df["long"])

    parts = []
    for _, g in df.groupby("cc_num", sort=False):
        g = g.sort_values("datetime").copy()
        t = g["datetime"]
        g["mins_since_prev"] = (t - t.shift(1)).dt.total_seconds() / 60.0
        g["prev_lat"] = g["lat"].shift(1)
        g["prev_long"] = g["long"].shift(1)
        ones = pd.Series(1.0, index=t)
        g["txn_1h"] = ones.rolling("1h").count().values - 1.0
        g["txn_24h"] = ones.rolling("24h").count().values - 1.0
        g["txn_7d"] = ones.rolling("7d").count().values - 1.0
        g["card_amt_mean"] = g["amount"].expanding().mean().shift(1).values
        g["card_amt_std"] = g["amount"].expanding().std().shift(1).values
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    dist_prev = haversine_km(df["lat"], df["long"], df["prev_lat"], df["prev_long"])
    df["dist_prev_km"] = pd.Series(dist_prev).fillna(0.0)
    hours_gap = (df["mins_since_prev"] / 60.0).clip(lower=1.0 / 60.0)
    df["speed_kmh"] = (df["dist_prev_km"] / hours_gap).fillna(0.0).clip(upper=5000.0)
    df["mins_since_prev"] = df["mins_since_prev"].fillna(1e5).clip(upper=1e5)

    df["card_amt_mean"] = df["card_amt_mean"].fillna(df["amount"])
    df["card_amt_std"] = df["card_amt_std"].fillna(0.0)
    std = df["card_amt_std"].replace(0.0, np.nan)
    df["amt_z"] = ((df["amount"] - df["card_amt_mean"]) / std).fillna(0.0).clip(-50, 50)
    df["amt_over_mean"] = (df["amount"] / df["card_amt_mean"].clip(lower=0.01)).clip(upper=100.0)

    df["dist_home_km"] = haversine_km(df["lat"], df["long"], df["home_lat"], df["home_long"])
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["datetime"].dt.hour.astype(float)
    df["dow"] = df["datetime"].dt.dayofweek.astype(float)
    df["is_night"] = df["hour"].between(0, 5).astype(float)
    for c in CATEGORIES:
        df[f"cat_{c}"] = (df["category"] == c).astype(float)

    keep = ["transaction_id", "cc_num", "datetime"] + FEATURES + ["is_fraud", "_split"]
    df = df[keep]
    hist_out = df[df["_split"] == "hist"].drop(columns=["_split"]).reset_index(drop=True)
    score_out = df[df["_split"] == "score"].drop(columns=["_split", "is_fraud"]).reset_index(drop=True)
    hist_out["is_fraud"] = hist_out["is_fraud"].astype("int64")
    return hist_out, score_out


def stage_features(project):
    fs = project.get_feature_store()
    hist, score = load_raw(project)
    hist_feat, _ = engineer(hist, score)
    print(f"engineered {len(hist_feat)} labelled rows, {hist_feat['is_fraud'].sum()} fraud")
    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=1,
        primary_key=["transaction_id"],
        event_time="datetime",
        online_enabled=False,
        description="Engineered credit-card fraud features (velocity, geo, amount signals) per transaction",
    )
    fg.insert(hist_feat, write_options={"wait_for_job": True})
    print("feature group written:", FG_NAME)


def stage_train(project):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    fs = project.get_feature_store()
    fv = fs.get_feature_view(FV_NAME, 1)
    try:
        x_train, x_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=1)
    except Exception as e:
        print(f"query-service read failed ({e}); falling back to direct parquet read")

        def read_split(split):
            ds = project.get_dataset_api()
            base = f"{project.name}_Training_Datasets/{FV_NAME}_1_1/{split}"
            frames = []
            for entry in ds.list_files(base, 0, 100)[1]:
                path = entry.attributes.path
                if path.endswith(".parquet"):
                    local = ds.download(path, overwrite=True)
                    frames.append(pd.read_parquet(local))
            return pd.concat(frames, ignore_index=True)

        train_df = read_split("train")
        test_df = read_split("test")
        y_train = train_df[["is_fraud"]]
        y_test = test_df[["is_fraud"]]
        x_train = train_df.drop(columns=["is_fraud"])
        x_test = test_df.drop(columns=["is_fraud"])

    drop_cols = [c for c in ("transaction_id", "cc_num", "datetime") if c in x_train.columns]
    x_train = x_train.drop(columns=drop_cols)
    x_test = x_test.drop(columns=drop_cols)
    feature_cols = list(x_train.columns)
    y_tr = np.asarray(y_train).ravel().astype(int)
    y_te = np.asarray(y_test).ravel().astype(int)
    print(f"train {x_train.shape}, test {x_test.shape}, fraud rate train {y_tr.mean():.4f}")

    clf = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    clf.fit(x_train, y_tr)
    proba = clf.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
    }
    print("metrics:", metrics)

    model_dir = "ccfraud_model_artifact"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": feature_cols}, os.path.join(model_dir, "model.pkl"))

    mr = project.get_model_registry()
    model = mr.sklearn.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description="HistGradientBoosting credit-card fraud classifier trained on cctd4b8521",
        feature_view=fv,
        training_dataset_version=1,
    )
    model.save(model_dir)
    print("registered model", MODEL_NAME)


def stage_score(project):
    import joblib

    fs = project.get_feature_store()
    hist, score = load_raw(project)
    _, score_feat = engineer(hist, score)
    print(f"engineered {len(score_feat)} score rows")

    mr = project.get_model_registry()
    model = mr.get_model(MODEL_NAME, version=1)
    model_dir = model.download()
    bundle = joblib.load(os.path.join(model_dir, "model.pkl"))
    clf = bundle["model"]
    feature_cols = bundle["feature_cols"]

    x = score_feat.reindex(columns=feature_cols, fill_value=0.0).astype(float)
    proba = clf.predict_proba(x)[:, 1]
    preds = pd.DataFrame({
        "transaction_id": score_feat["transaction_id"].astype(str),
        "fraud_probability": np.clip(proba.astype(float), 0.0, 1.0),
    })
    print(preds["fraud_probability"].describe())

    fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=1,
        primary_key=["transaction_id"],
        online_enabled=True,
        description="Fraud probability predictions for scored transactions",
    )
    fg.insert(preds, write_options={"wait_for_job": True})
    print("predictions written:", PRED_FG_NAME, len(preds))


def stage_verify(project):
    fs = project.get_feature_store()
    fg = fs.get_feature_group(PRED_FG_NAME, 1)
    online_df = fg.read(online=True)
    print("online rows:", len(online_df))
    print(online_df.head(3))
    print("range ok:", bool(online_df["fraud_probability"].between(0, 1).all()))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "features"
    project = hopsworks.login()
    if stage == "features":
        stage_features(project)
    elif stage == "train":
        stage_train(project)
    elif stage == "score":
        stage_score(project)
    elif stage == "verify":
        stage_verify(project)
    else:
        raise SystemExit(f"unknown stage: {stage}")


if __name__ == "__main__":
    main()
