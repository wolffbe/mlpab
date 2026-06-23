"""Full FTI fraud-detection pipeline, runs as a Hopsworks PYTHON job (platform-side)."""
import os
import numpy as np
import pandas as pd

import hopsworks

print(">>> logging in", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
ds = project.get_dataset_api()

# ----------------------------------------------------------------------------
# 0. Pull the raw CSVs that were uploaded to HopsFS
# ----------------------------------------------------------------------------
def fetch(remote, local):
    if os.path.exists(local):
        os.remove(local)
    p = ds.download(remote, local_path=local, overwrite=True)
    return p or local

train_csv = fetch("Resources/ccfraud/transactions.csv", "transactions.csv")
score_csv = fetch("Resources/ccfraud/score_transactions.csv", "score_transactions.csv")
print(">>> downloaded", train_csv, score_csv, flush=True)

raw_train = pd.read_csv(train_csv)
raw_score = pd.read_csv(score_csv)
print(">>> raw shapes", raw_train.shape, raw_score.shape, flush=True)

# ----------------------------------------------------------------------------
# 1. Feature engineering
# ----------------------------------------------------------------------------
CATEGORIES = sorted(raw_train["category"].dropna().unique().tolist())

# per-card priors computed from the labelled history (the "usual" behaviour)
g = raw_train.groupby("cc_num")
card_stats = pd.DataFrame({
    "card_mean_lat": g["lat"].mean(),
    "card_mean_long": g["long"].mean(),
    "card_mean_amount": g["amount"].mean(),
    "card_std_amount": g["amount"].std().fillna(0.0),
    "card_med_amount": g["amount"].median(),
}).reset_index()
GLOBAL = {
    "lat": raw_train["lat"].mean(),
    "long": raw_train["long"].mean(),
    "amount": raw_train["amount"].mean(),
    "std": raw_train["amount"].std(),
    "med": raw_train["amount"].median(),
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def engineer(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

    df["amount"] = df["amount"].astype(float)
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["datetime"].dt.hour.astype(int)
    df["dayofweek"] = df["datetime"].dt.dayofweek.astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)

    # join card priors
    df = df.merge(card_stats, on="cc_num", how="left")
    df["card_mean_lat"] = df["card_mean_lat"].fillna(GLOBAL["lat"])
    df["card_mean_long"] = df["card_mean_long"].fillna(GLOBAL["long"])
    df["card_mean_amount"] = df["card_mean_amount"].fillna(GLOBAL["amount"])
    df["card_std_amount"] = df["card_std_amount"].fillna(GLOBAL["std"]).replace(0, GLOBAL["std"])
    df["card_med_amount"] = df["card_med_amount"].fillna(GLOBAL["med"])

    # geo distance from the card's usual location
    df["geo_dist"] = haversine(df["lat"], df["long"], df["card_mean_lat"], df["card_mean_long"])

    # amount signals
    df["amount_z"] = (df["amount"] - df["card_mean_amount"]) / df["card_std_amount"].replace(0, 1)
    df["amount_ratio"] = df["amount"] / df["card_med_amount"].replace(0, 1)

    # velocity within the card timeline
    df["dt_prev_sec"] = df.groupby("cc_num")["datetime"].diff().dt.total_seconds()
    df["time_since_prev"] = df["dt_prev_sec"].fillna(7 * 24 * 3600.0).clip(upper=7 * 24 * 3600.0)
    # distance jump from previous txn
    df["prev_lat"] = df.groupby("cc_num")["lat"].shift(1)
    df["prev_long"] = df.groupby("cc_num")["long"].shift(1)
    df["step_dist"] = haversine(df["lat"], df["long"], df["prev_lat"].fillna(df["lat"]),
                                df["prev_long"].fillna(df["long"]))
    # implied speed (km/h) between consecutive txns
    df["speed"] = df["step_dist"] / (df["time_since_prev"] / 3600.0 + 1e-3)

    # rolling count in last hour per card
    counts = []
    for _, sub in df.groupby("cc_num"):
        t = sub["datetime"].values.astype("datetime64[s]").astype(np.int64)
        c = np.zeros(len(t), dtype=float)
        j = 0
        for i in range(len(t)):
            while t[i] - t[j] > 3600:
                j += 1
            c[i] = i - j
        counts.append(pd.Series(c, index=sub.index))
    df["txn_count_1h"] = pd.concat(counts).sort_index()

    # category one-hot (fixed columns from training)
    for c in CATEGORIES:
        df[f"cat_{c}"] = (df["category"] == c).astype(int)

    return df


FEATS = ["amount", "log_amount", "hour", "dayofweek", "is_night",
         "geo_dist", "amount_z", "amount_ratio", "time_since_prev",
         "step_dist", "speed", "txn_count_1h"] + [f"cat_{c}" for c in CATEGORIES]

train_fe = engineer(raw_train)
# re-map label by transaction_id (rows were re-sorted)
label_map = raw_train.set_index("transaction_id")["is_fraud"]
train_fe["is_fraud"] = train_fe["transaction_id"].map(label_map).astype(int)

score_fe = engineer(raw_score)
print(">>> engineered", train_fe.shape, score_fe.shape, flush=True)
print(">>> fraud rate", train_fe["is_fraud"].mean(), flush=True)

# ----------------------------------------------------------------------------
# 2. Feature group cctxn23ca19  (engineered features + label)
# ----------------------------------------------------------------------------
fg_cols = ["transaction_id", "cc_num", "datetime"] + FEATS + ["is_fraud"]
fg_df = train_fe[fg_cols].copy()
fg_df["cc_num"] = fg_df["cc_num"].astype(str)

fg = fs.get_or_create_feature_group(
    name="cctxn23ca19", version=1,
    description="Engineered credit-card fraud features",
    primary_key=["transaction_id"],
    event_time="datetime",
    online_enabled=False,
)
fg.insert(fg_df, write_options={"wait_for_job": True})
print(">>> inserted feature group cctxn23ca19", flush=True)

# ----------------------------------------------------------------------------
# 3. Feature view + training dataset cctd23ca19
# ----------------------------------------------------------------------------
fv = None
try:
    query = fg.select(FEATS + ["is_fraud"])
    fv = fs.get_or_create_feature_view(
        name="cctd23ca19", version=1,
        description="Fraud training feature view",
        query=query,
        labels=["is_fraud"],
    )
    print(">>> created feature view cctd23ca19", flush=True)
    try:
        td_version, _ = fv.create_train_test_split(
            test_size=0.2,
            description="cctd23ca19 training dataset",
            write_options={"wait_for_job": True},
        )
        print(">>> materialized training dataset version", td_version, flush=True)
    except Exception as e:
        print(">>> train_test_split warning:", repr(e), flush=True)
except Exception as e:
    print(">>> feature view warning:", repr(e), flush=True)

# ----------------------------------------------------------------------------
# 4. Train classifier + evaluate (held-out ROC AUC)
# ----------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

X = train_fe[FEATS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
y = train_fe["is_fraud"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

params = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
              l2_regularization=1.0, validation_fraction=0.1, random_state=42)
clf = HistGradientBoostingClassifier(**params)
clf.fit(Xtr, ytr)
proba_te = clf.predict_proba(Xte)[:, 1]
roc = float(roc_auc_score(yte, proba_te))
acc = float(accuracy_score(yte, (proba_te >= 0.5).astype(int)))
ap = float(average_precision_score(yte, proba_te))
print(f">>> HELD-OUT ROC AUC = {roc:.4f}  ACC = {acc:.4f}  AP = {ap:.4f}", flush=True)

# refit on all labelled data for the final scoring model
clf_full = HistGradientBoostingClassifier(**params)
clf_full.fit(X, y)

# ----------------------------------------------------------------------------
# 5. Register model ccmodel23ca19 with metrics
# ----------------------------------------------------------------------------
import joblib
model_dir = "ccmodel_artifact"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(clf_full, os.path.join(model_dir, "model.pkl"))
pd.Series(FEATS).to_json(os.path.join(model_dir, "features.json"))

mr = project.get_model_registry()
input_example = X.iloc[:1].to_dict(orient="records")[0]
model = mr.sklearn.create_model(
    name="ccmodel23ca19",
    metrics={"roc_auc": roc, "accuracy": acc, "average_precision": ap},
    description="Credit-card fraud classifier (HistGradientBoosting)",
    input_example=input_example,
    feature_view=fv,
)
model.save(model_dir)
print(">>> registered model ccmodel23ca19", flush=True)

# ----------------------------------------------------------------------------
# 6. Score every row of score_transactions.csv -> ccpred23ca19 (online+offline)
# ----------------------------------------------------------------------------
Xs = score_fe[FEATS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
score_proba = clf_full.predict_proba(Xs)[:, 1]
pred_df = pd.DataFrame({
    "transaction_id": score_fe["transaction_id"].astype(str).values,
    "fraud_probability": np.clip(score_proba, 0.0, 1.0).astype(float),
})
print(">>> scored", pred_df.shape, "mean prob", pred_df["fraud_probability"].mean(), flush=True)

pred_fg = fs.get_or_create_feature_group(
    name="ccpred23ca19", version=1,
    description="Fraud probability predictions for scored transactions",
    primary_key=["transaction_id"],
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print(">>> inserted predictions feature group ccpred23ca19 (online+offline)", flush=True)
print(">>> PIPELINE COMPLETE", flush=True)
