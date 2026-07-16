"""Full FTI fraud pipeline — runs as a Hopsworks PYTHON job (on-platform).

Reads raw FGs (ccraw52cef3 labelled, ccrawscore52cef3 to score), engineers
fraud features, writes feature group cctxn52cef3, builds feature view +
training dataset cctd52cef3, trains & registers ccmodel52cef3 with metrics,
scores every row and writes predictions FG ccpred52cef3 (online + offline).
"""
import os
import numpy as np
import pandas as pd
import hopsworks


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def base_frame(df):
    df = df.copy()
    df["ts"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df["amount"] = df["amount"].astype(float)
    df["lat"] = df["lat"].astype(float)
    df["long"] = df["long"].astype(float)
    df["cc_num"] = df["cc_num"].astype("int64")
    return df


def add_velocity(union):
    """Causal velocity features computed over the combined sorted timeline."""
    u = union.sort_values(["cc_num", "ts"]).reset_index(drop=True)
    u["secs_since_prev"] = u.groupby("cc_num")["ts"].diff().dt.total_seconds()
    u["txns_1h"] = 0
    u["txns_24h"] = 0
    for _, g in u.groupby("cc_num"):
        t = (g["ts"].astype("int64").to_numpy() // 10**9).astype("int64")
        idx = g.index.to_numpy()
        for i in range(len(t)):
            u.at[idx[i], "txns_1h"] = i - np.searchsorted(t, t[i] - 3600, side="left")
            u.at[idx[i], "txns_24h"] = i - np.searchsorted(t, t[i] - 86400, side="left")
    u["secs_since_prev"] = u["secs_since_prev"].fillna(1e7).clip(upper=1e7)
    return u


def engineer(df, ref, glob):
    df = df.merge(ref, on="cc_num", how="left")
    for c, v in glob.items():
        df[c] = df[c].fillna(v)
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["ts"].dt.hour.astype("int64")
    df["dow"] = df["ts"].dt.dayofweek.astype("int64")
    df["amt_ratio"] = (df["amount"] / (df["amt_mean"] + 1e-6)).clip(upper=1000)
    df["amt_z"] = ((df["amount"] - df["amt_mean"]) / (df["amt_std"] + 1e-6)).clip(-50, 50)
    df["geo_dist"] = haversine(df["lat"], df["long"], df["home_lat"], df["home_long"])
    df["txns_1h"] = df["txns_1h"].astype("int64")
    df["txns_24h"] = df["txns_24h"].astype("int64")
    return df


NUM = ["amount", "log_amount", "hour", "dow", "amt_ratio", "amt_z",
       "geo_dist", "secs_since_prev", "txns_1h", "txns_24h"]
CAT = ["category"]
KEEP = ["transaction_id", "cc_num", "ts"] + NUM + CAT


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    train_raw = base_frame(fs.get_feature_group("ccraw52cef3", version=1).read())
    score_raw = base_frame(fs.get_feature_group("ccrawscore52cef3", version=1).read())
    train_raw["is_fraud"] = train_raw["is_fraud"].astype("int64")
    print("loaded", len(train_raw), "train,", len(score_raw), "score")

    # per-card reference stats from TRAINING data only
    ref = (train_raw.groupby("cc_num")
           .agg(home_lat=("lat", "mean"), home_long=("long", "mean"),
                amt_mean=("amount", "mean"), amt_std=("amount", "std"))
           .reset_index())
    glob = {"home_lat": train_raw["lat"].mean(), "home_long": train_raw["long"].mean(),
            "amt_mean": train_raw["amount"].mean(),
            "amt_std": train_raw["amount"].std()}

    # velocity over the union (causal — only looks at past transactions)
    train_raw["_src"] = "train"
    score_raw["_src"] = "score"
    union = add_velocity(pd.concat([train_raw, score_raw], ignore_index=True, sort=False))
    union = engineer(union, ref, glob)

    train = union[union["_src"] == "train"].copy()
    score = union[union["_src"] == "score"].copy()

    # ---- Feature group cctxn52cef3 (engineered features + label) ----
    feat_cols = KEEP + ["is_fraud"]
    train_fg_df = train[feat_cols].rename(columns={"ts": "event_time"}).copy()
    train_fg_df["is_fraud"] = train_fg_df["is_fraud"].astype("int64")
    cctxn = fs.get_or_create_feature_group(
        name="cctxn52cef3", version=1, primary_key=["transaction_id"],
        event_time="event_time", online_enabled=False,
        description="Engineered fraud features per transaction")
    cctxn.insert(train_fg_df, write_options={"wait_for_job": True})
    print("inserted features into cctxn52cef3")

    # ---- Feature view + training dataset cctd52cef3 ----
    try:
        fs.get_feature_view("cctd52cef3", version=1).delete()
    except Exception:
        pass
    query = cctxn.select_all()
    fv = fs.create_feature_view(name="cctd52cef3", version=1, query=query,
                                labels=["is_fraud"],
                                description="Fraud training dataset definition")
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    print("td split:", len(X_train), len(X_test))

    # ---- Train classifier ----
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
        remainder="passthrough")
    clf = Pipeline([("pre", pre),
                    ("gb", HistGradientBoostingClassifier(
                        max_iter=400, learning_rate=0.08,
                        max_leaf_nodes=63, l2_regularization=1.0,
                        random_state=42))])
    yt = y_train["is_fraud"].astype(int).to_numpy()
    yv = y_test["is_fraud"].astype(int).to_numpy()
    clf.fit(X_train[NUM + CAT], yt)
    proba = clf.predict_proba(X_test[NUM + CAT])[:, 1]
    auc = float(roc_auc_score(yv, proba))
    acc = float(accuracy_score(yv, (proba >= 0.5).astype(int)))
    f1 = float(f1_score(yv, (proba >= 0.5).astype(int)))
    metrics = {"roc_auc": auc, "accuracy": acc, "f1": f1}
    print("METRICS", metrics)

    # ---- Register model ccmodel52cef3 ----
    import joblib
    mr = project.get_model_registry()
    mdir = "ccmodel52cef3_artifact"
    os.makedirs(mdir, exist_ok=True)
    joblib.dump(clf, os.path.join(mdir, "model.pkl"))
    input_example = X_test[NUM + CAT].head(2)
    try:
        model = mr.sklearn.create_model(
            name="ccmodel52cef3", metrics=metrics,
            feature_view=fv, input_example=input_example,
            description="Credit-card fraud classifier (HistGradientBoosting)")
    except Exception as e:
        print("sklearn registry fallback:", e)
        model = mr.python.create_model(
            name="ccmodel52cef3", metrics=metrics,
            description="Credit-card fraud classifier (HistGradientBoosting)")
    model.save(mdir)
    print("registered model ccmodel52cef3 v", model.version)

    # ---- Score every row -> ccpred52cef3 ----
    score_proba = clf.predict_proba(score[NUM + CAT])[:, 1]
    pred_df = pd.DataFrame({
        "transaction_id": score["transaction_id"].astype(str).values,
        "fraud_probability": np.clip(score_proba, 0.0, 1.0).astype(float),
    })
    print("scored", len(pred_df), "rows; proba range",
          float(pred_df.fraud_probability.min()), float(pred_df.fraud_probability.max()))
    ccpred = fs.get_or_create_feature_group(
        name="ccpred52cef3", version=1, primary_key=["transaction_id"],
        online_enabled=True,
        description="Fraud probability predictions for scored transactions")
    ccpred.insert(pred_df, write_options={"wait_for_job": True})
    print("inserted predictions into ccpred52cef3 (online+offline)")
    print("DONE roc_auc=", auc)


if __name__ == "__main__":
    main()
