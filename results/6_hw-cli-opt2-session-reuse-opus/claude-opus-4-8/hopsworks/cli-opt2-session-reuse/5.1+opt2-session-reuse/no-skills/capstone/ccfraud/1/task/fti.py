"""Full FTI pipeline for credit-card fraud detection, run as a Hopsworks job."""
import os
import numpy as np
import pandas as pd

import hopsworks

FG_NAME = "cctxnadd9bd"
FV_NAME = "cctdadd9bd"
MODEL_NAME = "ccmodeladd9bd"
PRED_FG_NAME = "ccpredadd9bd"


def log(*a):
    print("[FTI]", *a, flush=True)


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2.0) ** 2)
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_csv(ds, remote, local):
    for kwargs in ({"overwrite": True}, {}):
        try:
            ds.download(remote, local, **kwargs)
            break
        except TypeError:
            continue
        except Exception as e:
            log("download attempt failed", remote, repr(e))
    return pd.read_csv(local)


def add_velocity(df):
    """Within-set temporal velocity features per card."""
    df = df.sort_values(["cc_num", "datetime"]).copy()
    df["time_since_last_sec"] = (
        df.groupby("cc_num")["datetime"].diff().dt.total_seconds()
    )
    df["time_since_last_sec"] = df["time_since_last_sec"].fillna(1e6)
    idx = df.set_index("datetime")
    g = idx.groupby("cc_num")["amount"]
    df["cnt_1h"] = g.rolling("1h").count().reset_index(level=0, drop=True).values
    df["cnt_24h"] = g.rolling("24h").count().reset_index(level=0, drop=True).values
    df["sum_1h"] = g.rolling("1h").sum().reset_index(level=0, drop=True).values
    return df


def engineer(df, card_stats, cat_te, merch_te, global_cards, global_amt, global_rate):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df["amount"] = df["amount"].astype(float)
    df["lat"] = df["lat"].astype(float)
    df["long"] = df["long"].astype(float)

    df = add_velocity(df)

    df = df.merge(card_stats, on="cc_num", how="left")
    df["card_mean_amount"] = df["card_mean_amount"].fillna(global_amt)
    df["card_std_amount"] = df["card_std_amount"].fillna(0.0)
    df["card_mean_lat"] = df["card_mean_lat"].fillna(df["lat"])
    df["card_mean_long"] = df["card_mean_long"].fillna(df["long"])
    df["card_txn_count"] = df["card_txn_count"].fillna(0.0)

    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_night"] = (df["hour"] < 6).astype(int)
    df["amt_minus_card_mean"] = df["amount"] - df["card_mean_amount"]
    df["amt_over_card_mean"] = df["amount"] / (df["card_mean_amount"] + 1.0)
    df["amt_z"] = df["amt_minus_card_mean"] / (df["card_std_amount"] + 1.0)
    df["geo_dist"] = haversine(
        df["lat"], df["long"], df["card_mean_lat"], df["card_mean_long"]
    )
    df["cat_te"] = df["category"].map(cat_te).fillna(global_rate)
    df["merch_te"] = df["merchant"].map(merch_te).fillna(global_rate)
    return df


FEATURES = [
    "amount", "log_amount", "hour", "dayofweek", "is_night",
    "amt_minus_card_mean", "amt_over_card_mean", "amt_z", "geo_dist",
    "cnt_1h", "cnt_24h", "sum_1h", "time_since_last_sec",
    "cat_te", "merch_te", "card_txn_count",
]


def main():
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    ds = proj.get_dataset_api()

    log("loading data")
    train = load_csv(ds, "Resources/ccdata/transactions.csv", "transactions.csv")
    score = load_csv(ds, "Resources/ccdata/score_transactions.csv", "score_transactions.csv")
    train["datetime"] = pd.to_datetime(train["datetime"], utc=True).dt.tz_localize(None)
    log("train", train.shape, "score", score.shape)

    # Per-card aggregates and target encodings derived from labelled history only.
    grp = train.groupby("cc_num")
    card_stats = pd.DataFrame({
        "card_mean_amount": grp["amount"].mean(),
        "card_std_amount": grp["amount"].std().fillna(0.0),
        "card_mean_lat": grp["lat"].mean(),
        "card_mean_long": grp["long"].mean(),
        "card_txn_count": grp.size(),
    }).reset_index()

    global_rate = float(train["is_fraud"].mean())
    global_amt = float(train["amount"].mean())
    global_cards = float(card_stats["card_txn_count"].mean())

    def smoothed_te(col, m=20.0):
        agg = train.groupby(col)["is_fraud"].agg(["mean", "count"])
        return ((agg["mean"] * agg["count"] + global_rate * m)
                / (agg["count"] + m)).to_dict()

    cat_te = smoothed_te("category")
    merch_te = smoothed_te("merchant")

    log("engineering train features")
    tr = engineer(train, card_stats, cat_te, merch_te, global_cards, global_amt, global_rate)
    tr["is_fraud"] = train.set_index("transaction_id").loc[tr["transaction_id"], "is_fraud"].values
    tr["event_time"] = tr["datetime"]

    fg_df = tr[["transaction_id", "event_time"] + FEATURES + ["is_fraud"]].copy()

    # ---- Feature group ----
    log("creating feature group", FG_NAME)
    fg = fs.get_or_create_feature_group(
        name=FG_NAME, version=1,
        description="Engineered fraud features per transaction",
        primary_key=["transaction_id"], event_time="event_time",
        online_enabled=False,
    )
    fg.insert(fg_df, write_options={"wait_for_job": True})
    log("feature group inserted")

    # ---- Feature view + training dataset ----
    log("creating feature view", FV_NAME)
    try:
        old = fs.get_feature_view(name=FV_NAME, version=1)
        old.delete()
    except Exception:
        pass
    query = fg.select(FEATURES + ["is_fraud"])
    fv = fs.create_feature_view(
        name=FV_NAME, version=1, query=query, labels=["is_fraud"],
        description="Fraud training dataset feature view",
    )
    log("materializing training dataset cctdadd9bd")
    td_version = 1
    try:
        td_version, _ = fv.create_train_test_split(
            test_size=0.2, description="cctdadd9bd training dataset",
            write_options={"wait_for_job": True},
        )
    except Exception as e:
        log("create_train_test_split failed, falling back to training_data", repr(e))
        try:
            td_version, _ = fv.create_training_data(
                description="cctdadd9bd training dataset",
                write_options={"wait_for_job": True},
            )
        except Exception as e2:
            log("create_training_data also failed", repr(e2))

    # ---- Train ----
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    X = tr[FEATURES].astype(float).values
    y = tr["is_fraud"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.07, max_depth=None,
        max_leaf_nodes=31, l2_regularization=1.0, random_state=42,
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))
    acc = float(accuracy_score(yte, (proba >= 0.5).astype(int)))
    f1 = float(f1_score(yte, (proba >= 0.5).astype(int)))
    log(f"holdout AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f}")

    # Refit on all data for final scoring.
    clf.fit(X, y)

    # ---- Register model ----
    import joblib
    mr = proj.get_model_registry()
    model_dir = "ccmodel_dir"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))
    input_example = tr[FEATURES].iloc[:1].astype(float).values.tolist()
    metrics = {"roc_auc": auc, "accuracy": acc, "f1": f1}
    model = mr.python.create_model(
        name=MODEL_NAME, metrics=metrics,
        description="Fraud classifier (HistGradientBoosting)",
        input_example=input_example,
        feature_view=fv,
    )
    model.save(model_dir)
    log("model registered", MODEL_NAME, "metrics", metrics)

    # ---- Score & write predictions ----
    log("scoring score_transactions")
    sc = engineer(score, card_stats, cat_te, merch_te, global_cards, global_amt, global_rate)
    sc_proba = clf.predict_proba(sc[FEATURES].astype(float).values)[:, 1]
    pred_df = pd.DataFrame({
        "transaction_id": sc["transaction_id"].values,
        "fraud_probability": np.clip(sc_proba, 0.0, 1.0).astype(float),
    })
    log("pred sample", pred_df.head().to_dict("records"))

    pred_fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME, version=1,
        description="Fraud probability predictions for scored transactions",
        primary_key=["transaction_id"], online_enabled=True,
    )
    pred_fg.insert(pred_df, write_options={"wait_for_job": True})
    log("predictions written to", PRED_FG_NAME, "rows", len(pred_df))
    log("DONE")


if __name__ == "__main__":
    main()
