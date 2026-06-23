"""Full FTI pipeline for credit-card fraud detection, runs as a Hopsworks job.

Feature engineering -> feature group cctxnfe5424
Feature view + training dataset cctdfe5424
Train + register sklearn classifier ccmodelfe5424 (with metrics)
Score score_transactions.csv -> online+offline feature table ccpredfe5424
"""
import os
import numpy as np
import pandas as pd
import hopsworks

FG_NAME = "cctxnfe5424"
FV_NAME = "cctdfe5424"
MODEL_NAME = "ccmodelfe5424"
PRED_FG = "ccpredfe5424"

FEATURES = [
    "amount", "log_amount", "hour", "dow",
    "geo_dist_home", "dist_from_prev", "secs_since_prev",
    "cnt_1h", "cnt_24h", "amount_to_mean_ratio", "cat_fraud_rate",
]


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def add_card_history_features(df, card_mean):
    """df must be sorted by cc_num, datetime. Adds per-card rolling features."""
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

    # home location & mean amount per card (from training history)
    df["home_lat"] = df["cc_num"].map(card_mean["lat"]).fillna(card_mean["lat"].mean())
    df["home_long"] = df["cc_num"].map(card_mean["long"]).fillna(card_mean["long"].mean())
    df["card_mean_amount"] = df["cc_num"].map(card_mean["amount"]).fillna(card_mean["amount"].mean())

    df["geo_dist_home"] = haversine(df["lat"], df["long"], df["home_lat"], df["home_long"])
    df["amount_to_mean_ratio"] = df["amount"] / df["card_mean_amount"].replace(0, np.nan)
    df["amount_to_mean_ratio"] = df["amount_to_mean_ratio"].fillna(1.0)

    # previous-transaction features
    g = df.groupby("cc_num", sort=False)
    df["prev_lat"] = g["lat"].shift(1)
    df["prev_long"] = g["long"].shift(1)
    df["secs_since_prev"] = g["datetime"].diff().dt.total_seconds()
    df["dist_from_prev"] = haversine(df["lat"], df["long"], df["prev_lat"], df["prev_long"])

    # time-window velocity counts (prior transactions within window)
    cnt1, cnt24 = [], []
    for _, grp in df.groupby("cc_num", sort=False):
        s = grp.set_index("datetime")["amount"]
        c1 = s.rolling("1h").count() - 1
        c24 = s.rolling("24h").count() - 1
        cnt1.append(pd.Series(c1.values, index=grp.index))
        cnt24.append(pd.Series(c24.values, index=grp.index))
    df["cnt_1h"] = pd.concat(cnt1).sort_index()
    df["cnt_24h"] = pd.concat(cnt24).sort_index()

    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    df["secs_since_prev"] = df["secs_since_prev"].fillna(1e7)
    df["dist_from_prev"] = df["dist_from_prev"].fillna(0.0)
    df["cnt_1h"] = df["cnt_1h"].fillna(0.0).clip(lower=0)
    df["cnt_24h"] = df["cnt_24h"].fillna(0.0).clip(lower=0)
    return df


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dsapi = project.get_dataset_api()

    for f in ["transactions.csv", "score_transactions.csv"]:
        try:
            dsapi.download(f"Resources/ccfraud/{f}", local_path=f, overwrite=True)
        except Exception as e:
            print("download note:", e)

    train = pd.read_csv("transactions.csv")
    score = pd.read_csv("score_transactions.csv")
    train["datetime"] = pd.to_datetime(train["datetime"], utc=True).dt.tz_localize(None)
    score["datetime"] = pd.to_datetime(score["datetime"], utc=True).dt.tz_localize(None)

    # card-level statistics from labelled history (no leakage from score set)
    card_mean = train.groupby("cc_num")[["lat", "long", "amount"]].mean()
    cat_rate = train.groupby("category")["is_fraud"].mean()
    global_rate = train["is_fraud"].mean()

    train_fe = add_card_history_features(train.copy(), card_mean)
    train_fe["cat_fraud_rate"] = train_fe["category"].map(cat_rate).fillna(global_rate)

    # build score features against combined per-card history for correct velocity/recency
    combined = pd.concat([
        train[["transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "long"]],
        score[["transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "long"]],
    ], ignore_index=True)
    comb_fe = add_card_history_features(combined, card_mean)
    comb_fe["cat_fraud_rate"] = comb_fe["category"].map(cat_rate).fillna(global_rate)
    score_ids = set(score["transaction_id"])
    score_fe = comb_fe[comb_fe["transaction_id"].isin(score_ids)].copy()

    # ---------- Feature group cctxnfe5424 (labelled engineered features) ----------
    fg_cols = ["transaction_id", "cc_num", "datetime"] + FEATURES + ["is_fraud"]
    fg_df = train_fe[fg_cols].copy()
    fg_df["transaction_id"] = fg_df["transaction_id"].astype(str)
    fg_df["is_fraud"] = fg_df["is_fraud"].astype("int64")

    fg = fs.get_or_create_feature_group(
        name=FG_NAME, version=1,
        description="Engineered credit-card fraud features",
        primary_key=["transaction_id"], event_time="datetime",
        online_enabled=False,
    )
    fg.insert(fg_df, write_options={"wait_for_job": True})
    print("Inserted", len(fg_df), "rows into", FG_NAME)

    # ---------- Feature view + training dataset cctdfe5424 ----------
    try:
        existing = fs.get_feature_view(name=FV_NAME, version=1)
        existing.delete()
    except Exception:
        pass
    query = fg.select(FEATURES + ["is_fraud"])
    fv = fs.create_feature_view(
        name=FV_NAME, version=1, query=query, labels=["is_fraud"],
        description="Fraud features for ccmodelfe5424",
    )
    print("Created feature view", FV_NAME)
    td_version, _ = fv.create_train_test_split(
        test_size=0.2, description="cctdfe5424 training dataset",
        write_options={"wait_for_job": True},
    )
    print("Created training dataset version", td_version)

    # ---------- Train + register classifier ccmodelfe5424 ----------
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    import joblib

    X = train_fe[FEATURES].astype("float64").fillna(0.0)
    y = train_fe["is_fraud"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=42)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "accuracy": float(accuracy_score(y_te, (proba >= 0.5).astype(int))),
        "f1": float(f1_score(y_te, (proba >= 0.5).astype(int))),
    }
    print("Held-out metrics:", metrics)

    # refit on full labelled data for scoring
    final_clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=42)
    final_clf.fit(X, y)

    model_dir = "ccmodel_artifact"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_clf, os.path.join(model_dir, "model.pkl"))

    mr = project.get_model_registry()
    input_example = X.iloc[:1].to_dict(orient="records")[0]
    model = mr.python.create_model(
        name=MODEL_NAME, metrics=metrics,
        description="Credit-card fraud GradientBoosting classifier",
        feature_view=fv, input_example=input_example,
    )
    model.save(model_dir)
    print("Registered model", MODEL_NAME, "version", model.version)

    # ---------- Score score_transactions.csv -> ccpredfe5424 (online + offline) ----------
    Xs = score_fe[FEATURES].astype("float64").fillna(0.0)
    score_fe = score_fe.copy()
    score_fe["fraud_probability"] = final_clf.predict_proba(Xs)[:, 1].clip(0.0, 1.0)

    pred_df = score_fe[["transaction_id", "cc_num", "datetime", "fraud_probability"]].copy()
    pred_df["transaction_id"] = pred_df["transaction_id"].astype(str)
    pred_df["fraud_probability"] = pred_df["fraud_probability"].astype("float64")

    pred_fg = fs.get_or_create_feature_group(
        name=PRED_FG, version=1,
        description="Fraud probability predictions for scoring set",
        primary_key=["transaction_id"], event_time="datetime",
        online_enabled=True,
    )
    pred_fg.insert(pred_df, write_options={"wait_for_job": True})
    print("Wrote", len(pred_df), "predictions into", PRED_FG)
    print("PIPELINE_DONE roc_auc=", metrics["roc_auc"])


if __name__ == "__main__":
    main()
