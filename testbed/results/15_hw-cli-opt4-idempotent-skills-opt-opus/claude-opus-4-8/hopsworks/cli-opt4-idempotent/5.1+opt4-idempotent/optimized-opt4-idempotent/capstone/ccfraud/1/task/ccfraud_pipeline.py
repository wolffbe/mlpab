"""End-to-end FTI fraud pipeline. Runs ON the Hopsworks platform as a PYTHON job."""
import os, sys, math, json, tempfile, traceback
import numpy as np
import pandas as pd
import hopsworks

CAT_LIST = ["cash_advance", "clothing", "electronics", "entertainment", "fuel",
            "grocery", "health", "online", "restaurant", "travel"]

def log(*a):
    print("[ccfraud]", *a, flush=True)

def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return r * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def velocity(df, hours):
    """Count of prior transactions per card within the trailing `hours` window."""
    out = np.zeros(len(df), dtype=np.int64)
    win = pd.Timedelta(hours=hours)
    for cc, idx in df.groupby("cc_num").groups.items():
        idx = list(idx)
        ts = df.loc[idx, "datetime"].values
        order = np.argsort(ts)
        ts_sorted = ts[order]
        # for each, count earlier txns within window
        j = 0
        for k in range(len(ts_sorted)):
            while ts_sorted[k] - ts_sorted[j] > win.to_timedelta64():
                j += 1
            out[idx[order[k]]] = k - j  # prior txns strictly before in window
    return out

def engineer(df, card_stats, global_amt, global_lat, global_lon):
    df = df.reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["amount"] = df["amount"].astype(float)
    df["lat"] = df["lat"].astype(float)
    df["long"] = df["long"].astype(float)
    df["hour"] = df["datetime"].dt.hour.astype("int64")
    df["log_amount"] = np.log1p(df["amount"])
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 22, 23]).astype("int64")
    # card-level reference stats (from labelled history only -> no leakage)
    cm = df["cc_num"].map(lambda c: card_stats.get(c, {}).get("mean_amt", global_amt))
    hlat = df["cc_num"].map(lambda c: card_stats.get(c, {}).get("home_lat", global_lat))
    hlon = df["cc_num"].map(lambda c: card_stats.get(c, {}).get("home_lon", global_lon))
    df["amount_ratio"] = (df["amount"] / (cm.astype(float) + 1.0)).astype(float)
    df["geo_distance"] = haversine(df["lat"].values, df["long"].values,
                                   hlat.astype(float).values, hlon.astype(float).values).astype(float)
    df["velocity_1h"] = velocity(df, 1)
    df["velocity_24h"] = velocity(df, 24)
    for c in CAT_LIST:
        df["cat_" + c] = (df["category"] == c).astype("int64")
    return df

FEATURES = (["amount", "log_amount", "hour", "is_night", "amount_ratio",
             "geo_distance", "velocity_1h", "velocity_24h"]
            + ["cat_" + c for c in CAT_LIST])

def main():
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    base = "/hopsfs/Projects/{}/Resources/ccfraud".format(proj.name)
    if not os.path.exists(base):
        # fallback common mount
        base = os.path.join(os.environ.get("PROJECT_PATH", ""), "Resources", "ccfraud")
    log("data dir", base, os.listdir(base) if os.path.exists(base) else "MISSING")

    train = pd.read_csv(os.path.join(base, "transactions.csv"))
    score = pd.read_csv(os.path.join(base, "score_transactions.csv"))
    log("train", train.shape, "score", score.shape)

    # card reference stats from labelled history
    train_dt = train.copy()
    card_stats = {}
    g = train_dt.groupby("cc_num")
    for cc, sub in g:
        card_stats[cc] = {
            "mean_amt": float(sub["amount"].mean()),
            "home_lat": float(sub["lat"].median()),
            "home_lon": float(sub["long"].median()),
        }
    global_amt = float(train_dt["amount"].mean())
    global_lat = float(train_dt["lat"].median())
    global_lon = float(train_dt["long"].median())

    train_f = engineer(train, card_stats, global_amt, global_lat, global_lon)
    train_f["is_fraud"] = train["is_fraud"].astype("int64").values
    score_f = engineer(score, card_stats, global_amt, global_lat, global_lon)

    # ---------- Feature group cctxn61210b ----------
    fg_cols = ["transaction_id", "cc_num", "datetime"] + FEATURES + ["is_fraud"]
    fg_df = train_f[fg_cols].copy()
    fg_df["cc_num"] = fg_df["cc_num"].astype(str)
    fg = fs.get_or_create_feature_group(
        name="cctxn61210b", version=1,
        description="Engineered credit-card fraud features (labelled history)",
        primary_key=["transaction_id"], event_time="datetime",
        online_enabled=False,
    )
    fg.insert(fg_df, write_options={"wait_for_job": True})
    log("inserted FG cctxn61210b", fg_df.shape)

    # ---------- Feature view cctd61210b + training dataset ----------
    try:
        fs.delete_feature_view(name="cctd61210b", version=1)
    except Exception as e:
        log("no existing fv to delete:", e)
    query = fg.select(FEATURES + ["is_fraud"])
    fv = fs.create_feature_view(
        name="cctd61210b", version=1, query=query, labels=["is_fraud"],
        description="Fraud training feature view",
    )
    log("created FV cctd61210b")

    X_train, X_test, y_train, y_test = fv.train_test_split(
        test_size=0.2, description="cctd61210b training dataset")
    td_version = fv.get_training_datasets()[-1].version if hasattr(fv, "get_training_datasets") else 1
    log("training dataset version", td_version, "Xtrain", X_train.shape, "Xtest", X_test.shape)

    X_train = X_train[FEATURES].astype(float)
    X_test = X_test[FEATURES].astype(float)
    y_train = np.ravel(y_train).astype(int)
    y_test = np.ravel(y_test).astype(int)

    # ---------- Train classifier ----------
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              max_depth=6, l2_regularization=1.0,
                                              class_weight="balanced", random_state=42)
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    ap = float(average_precision_score(y_test, proba))
    f1 = float(f1_score(y_test, (proba >= 0.5).astype(int)))
    log("HELD-OUT ROC AUC", auc, "AP", ap, "F1", f1)

    # ---------- Register model ccmodel61210b ----------
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    mr = proj.get_model_registry()
    mdir = tempfile.mkdtemp()
    joblib.dump(clf, os.path.join(mdir, "model.pkl"))
    input_schema = Schema(X_train)
    output_schema = Schema(np.array([0]))
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    model = mr.python.create_model(
        name="ccmodel61210b",
        metrics={"roc_auc": auc, "average_precision": ap, "f1": f1},
        description="Credit-card fraud classifier",
        input_example=X_train.iloc[:1],
        model_schema=model_schema,
        feature_view=fv,
    )
    model.save(mdir)
    log("registered model ccmodel61210b with roc_auc", auc)

    # ---------- Score + write ccpred61210b (online + offline) ----------
    sproba = clf.predict_proba(score_f[FEATURES].astype(float))[:, 1]
    pred_df = pd.DataFrame({
        "transaction_id": score["transaction_id"].astype(str).values,
        "fraud_probability": np.clip(sproba, 0.0, 1.0).astype(float),
    })
    pred_fg = fs.get_or_create_feature_group(
        name="ccpred61210b", version=1,
        description="Fraud probability predictions for scoring set",
        primary_key=["transaction_id"], online_enabled=True,
    )
    pred_fg.insert(pred_df, write_options={"wait_for_job": True})
    log("wrote ccpred61210b", pred_df.shape, "fraud_prob range",
        float(pred_df.fraud_probability.min()), float(pred_df.fraud_probability.max()))
    log("DONE auc=%.4f" % auc)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
