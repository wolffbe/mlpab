"""Full FTI fraud-detection pipeline, runs as a Hopsworks job (platform-side)."""
import os
import numpy as np
import pandas as pd
import hopsworks


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2.0) ** 2)
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def rolling_counts(df):
    """Causal per-card velocity features over a time-sorted union frame."""
    df = df.sort_values(["cc_num", "datetime"]).copy()
    out_tsl = np.zeros(len(df))
    out_1h = np.zeros(len(df))
    out_24h = np.zeros(len(df))
    pos = 0
    for _, g in df.groupby("cc_num", sort=False):
        t = g["datetime"].values.astype("datetime64[s]").astype(np.int64)
        n = len(t)
        for i in range(n):
            ti = t[i]
            out_tsl[pos + i] = (ti - t[i - 1]) if i > 0 else 1e7
            c1 = 0
            c24 = 0
            j = i - 1
            while j >= 0 and ti - t[j] <= 86400:
                if ti - t[j] <= 3600:
                    c1 += 1
                c24 += 1
                j -= 1
            out_1h[pos + i] = c1
            out_24h[pos + i] = c24
        pos += n
    df["time_since_last"] = out_tsl
    df["cnt_1h"] = out_1h
    df["cnt_24h"] = out_24h
    return df


def build_features(raw, card_stats, cat_rate, global_lat, global_long,
                   global_amt_mean, global_amt_std, global_cat_rate):
    df = raw.copy()
    df = df.merge(card_stats, on="cc_num", how="left")
    df["home_lat"] = df["home_lat"].fillna(global_lat)
    df["home_long"] = df["home_long"].fillna(global_long)
    df["card_amt_mean"] = df["card_amt_mean"].fillna(global_amt_mean)
    df["card_amt_std"] = df["card_amt_std"].fillna(global_amt_std).replace(0, global_amt_std)
    df["geo_dist"] = haversine(df["lat"], df["long"], df["home_lat"], df["home_long"])
    df["amount_z"] = (df["amount"] - df["card_amt_mean"]) / (df["card_amt_std"] + 1e-6)
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["cat_fraud_rate"] = df["category"].map(cat_rate).fillna(global_cat_rate)
    return df


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    ds = project.get_dataset_api()

    ds.download("Resources/ccfraud/transactions.csv", "transactions.csv", overwrite=True)
    ds.download("Resources/ccfraud/score_transactions.csv", "score_transactions.csv", overwrite=True)

    train = pd.read_csv("transactions.csv")
    score = pd.read_csv("score_transactions.csv")
    train["datetime"] = pd.to_datetime(train["datetime"])
    score["datetime"] = pd.to_datetime(score["datetime"])

    # Per-card aggregates from TRAINING history only (avoid leakage)
    card_stats = train.groupby("cc_num").agg(
        home_lat=("lat", "median"),
        home_long=("long", "median"),
        card_amt_mean=("amount", "mean"),
        card_amt_std=("amount", "std"),
    ).reset_index()
    card_stats["card_amt_std"] = card_stats["card_amt_std"].fillna(0.0)

    # smoothed category target encoding from training
    glob = train["is_fraud"].mean()
    cat_grp = train.groupby("category")["is_fraud"].agg(["mean", "count"])
    k = 20.0
    cat_rate = ((cat_grp["mean"] * cat_grp["count"] + glob * k)
                / (cat_grp["count"] + k)).to_dict()

    g_lat = train["lat"].median()
    g_long = train["long"].median()
    g_amt_mean = train["amount"].mean()
    g_amt_std = train["amount"].std()

    # velocity features over union (causal: only earlier txns counted)
    union = pd.concat([train.drop(columns=["is_fraud"]), score], ignore_index=True)
    union = rolling_counts(union)
    vel = union[["transaction_id", "time_since_last", "cnt_1h", "cnt_24h"]]

    train = train.merge(vel, on="transaction_id", how="left")
    score = score.merge(vel, on="transaction_id", how="left")

    feat_train = build_features(train, card_stats, cat_rate, g_lat, g_long,
                                g_amt_mean, g_amt_std, glob)
    feat_score = build_features(score, card_stats, cat_rate, g_lat, g_long,
                                g_amt_mean, g_amt_std, glob)

    FEATS = ["amount", "log_amount", "geo_dist", "amount_z", "hour", "dayofweek",
             "is_night", "cat_fraud_rate", "time_since_last", "cnt_1h", "cnt_24h"]

    # ---- Feature group with engineered features + label ----
    fg_cols = ["transaction_id", "datetime"] + FEATS + ["is_fraud"]
    fg_df = feat_train[fg_cols].copy()
    fg = fs.get_or_create_feature_group(
        name="cctxn82e46e", version=1,
        primary_key=["transaction_id"], event_time="datetime",
        online_enabled=True,
        description="Engineered credit-card fraud features",
    )
    fg.insert(fg_df, write_options={"wait_for_job": True})

    # ---- Feature view + training dataset cctd82e46e ----
    query = fg.select(FEATS + ["is_fraud"])
    fv = fs.get_or_create_feature_view(
        name="cctd82e46e", version=1, query=query, labels=["is_fraud"],
        description="Fraud training feature view",
    )
    X_train, X_test, y_train, y_test = fv.train_test_split(
        test_size=0.2, description="cctd82e46e training dataset")
    X_train = X_train[FEATS]
    X_test = X_test[FEATS]
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # ---- Train classifier ----
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                     learning_rate=0.1, subsample=0.9,
                                     random_state=42)
    clf.fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, p_test))
    acc = float(accuracy_score(y_test, (p_test >= 0.5).astype(int)))
    f1 = float(f1_score(y_test, (p_test >= 0.5).astype(int)))
    print(f"HELDOUT_ROC_AUC={auc:.4f} ACC={acc:.4f} F1={f1:.4f}")

    # ---- Register model with metrics ----
    import joblib
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    os.makedirs("ccmodel_dir", exist_ok=True)
    joblib.dump(clf, "ccmodel_dir/model.pkl")
    mr = project.get_model_registry()
    model_schema = ModelSchema(
        input_schema=Schema(X_train),
        output_schema=Schema(pd.DataFrame({"fraud_probability": [0.0]})),
    )
    model = mr.python.create_model(
        name="ccmodel82e46e",
        metrics={"roc_auc": auc, "accuracy": acc, "f1": f1},
        description="Credit-card fraud classifier",
        model_schema=model_schema,
        input_example=X_train.head(1),
    )
    model.save("ccmodel_dir")

    # ---- Score every row & write predictions feature table ----
    Xs = feat_score[FEATS]
    proba = clf.predict_proba(Xs)[:, 1]
    pred_df = pd.DataFrame({
        "transaction_id": feat_score["transaction_id"].values,
        "fraud_probability": np.clip(proba, 0.0, 1.0).astype(float),
    })
    pred_fg = fs.get_or_create_feature_group(
        name="ccpred82e46e", version=1,
        primary_key=["transaction_id"],
        online_enabled=True,
        description="Fraud probability predictions for scoring set",
    )
    pred_fg.insert(pred_df, write_options={"wait_for_job": True})
    print(f"SCORED_ROWS={len(pred_df)}")
    print("PIPELINE_DONE")


if __name__ == "__main__":
    main()
