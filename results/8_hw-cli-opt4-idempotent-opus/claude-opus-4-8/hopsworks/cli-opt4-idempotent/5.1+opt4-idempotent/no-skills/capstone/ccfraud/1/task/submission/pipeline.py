"""Full FTI fraud pipeline, runs ON the Hopsworks platform as a PYTHON job."""
import os
import numpy as np
import pandas as pd
import joblib
import hopsworks

SUFFIX = "6c34e0"
FG_NAME = "cctxn" + SUFFIX
FV_NAME = "cctd" + SUFFIX          # feature view; its training dataset is the deliverable
MODEL_NAME = "ccmodel" + SUFFIX
PRED_NAME = "ccpred" + SUFFIX

print("Logging in...", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
ds = project.get_dataset_api()

# ---------------------------------------------------------------- load inputs
def fetch(name):
    for remote in ["Resources/ccfraud/" + name, "Resources/" + name]:
        try:
            local = ds.download(remote, local_path=name, overwrite=True)
            print("downloaded", remote, "->", local, flush=True)
            return local or name
        except Exception as e:  # noqa
            print("download miss", remote, e, flush=True)
    return name

hist_path = fetch("transactions.csv")
score_path = fetch("score_transactions.csv")

hist = pd.read_csv(hist_path, parse_dates=["datetime"])
score = pd.read_csv(score_path, parse_dates=["datetime"])
print("hist", hist.shape, "score", score.shape, flush=True)

# ---------------------------------------------------------- feature engineering
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2.0) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# per-card reference stats derived ONLY from labelled history (no leakage)
ref = hist.groupby("cc_num").agg(
    home_lat=("lat", "median"),
    home_long=("long", "median"),
    amt_mean=("amount", "mean"),
    amt_std=("amount", "std"),
).reset_index()
g_lat = hist["lat"].median()
g_long = hist["long"].median()
g_mean = hist["amount"].mean()
g_std = hist["amount"].std()

CATS = sorted(hist["category"].dropna().unique().tolist())

allc = pd.concat([hist.assign(_part="h"), score.assign(_part="s")],
                 ignore_index=True, sort=False)
allc = allc.merge(ref, on="cc_num", how="left")
allc["home_lat"] = allc["home_lat"].fillna(g_lat)
allc["home_long"] = allc["home_long"].fillna(g_long)
allc["amt_mean"] = allc["amt_mean"].fillna(g_mean)
allc["amt_std"] = allc["amt_std"].fillna(g_std).fillna(0.0)

allc["hour"] = allc["datetime"].dt.hour
allc["dow"] = allc["datetime"].dt.dayofweek
allc["log_amount"] = np.log1p(allc["amount"])
allc["amt_vs_mean"] = allc["amount"] / (allc["amt_mean"] + 1.0)
allc["amt_z"] = (allc["amount"] - allc["amt_mean"]) / (allc["amt_std"] + 1.0)
allc["dist_home"] = haversine(allc["lat"], allc["long"],
                              allc["home_lat"], allc["home_long"])

# trailing per-card velocity / geo signals over the combined timeline
allc = allc.sort_values(["cc_num", "datetime"]).reset_index(drop=True)
grp = allc.groupby("cc_num")
allc["prev_dt"] = grp["datetime"].shift(1)
allc["prev_lat"] = grp["lat"].shift(1)
allc["prev_long"] = grp["long"].shift(1)
allc["time_since_prev"] = (allc["datetime"] - allc["prev_dt"]).dt.total_seconds()
allc["dist_prev"] = haversine(allc["lat"], allc["long"],
                              allc["prev_lat"], allc["prev_long"])
allc["speed"] = allc["dist_prev"] / ((allc["time_since_prev"] / 3600.0) + 0.05)

def add_count(sub):
    sub = sub.sort_values("datetime")
    sub["txn_count_1h"] = sub.rolling("1h", on="datetime")["amount"].count() - 1
    sub["txn_count_24h"] = sub.rolling("24h", on="datetime")["amount"].count() - 1
    return sub

allc = allc.groupby("cc_num", group_keys=False).apply(add_count)

allc["time_since_prev"] = allc["time_since_prev"].fillna(1e7).clip(upper=1e7)
allc["dist_prev"] = allc["dist_prev"].fillna(0.0)
allc["speed"] = allc["speed"].fillna(0.0)
allc["txn_count_1h"] = allc["txn_count_1h"].fillna(0.0)
allc["txn_count_24h"] = allc["txn_count_24h"].fillna(0.0)

# category one-hot (fixed schema across hist + score)
for c in CATS:
    allc["cat_" + c] = (allc["category"] == c).astype("int64")

FEATURES = (["amount", "log_amount", "hour", "dow", "amt_vs_mean", "amt_z",
             "dist_home", "time_since_prev", "dist_prev", "speed",
             "txn_count_1h", "txn_count_24h"]
            + ["cat_" + c for c in CATS])

for col in FEATURES:
    allc[col] = pd.to_numeric(allc[col], errors="coerce").fillna(0.0).astype("float64")

hist_f = allc[allc["_part"] == "h"].copy()
score_f = allc[allc["_part"] == "s"].copy()
hist_f["is_fraud"] = hist_f["is_fraud"].astype("int64")
print("engineered hist", hist_f.shape, "score", score_f.shape, flush=True)

# ----------------------------------------------------- feature group (offline)
fg_cols = ["transaction_id", "cc_num", "datetime"] + FEATURES + ["is_fraud"]
fg_df = hist_f[fg_cols].copy()
fg_df["transaction_id"] = fg_df["transaction_id"].astype(str)
fg_df["cc_num"] = fg_df["cc_num"].astype("int64")

fg = fs.get_or_create_feature_group(
    name=FG_NAME, version=1,
    primary_key=["transaction_id"],
    event_time="datetime",
    online_enabled=False,
    description="Engineered credit-card fraud features (velocity, geo, amount signals).",
)
print("inserting feature group...", flush=True)
fg.insert(fg_df, write_options={"wait_for_job": True})
print("feature group inserted", flush=True)

# ----------------------------------------------- feature view + training dataset
try:
    query = fg.select_all()
    fv = fs.get_or_create_feature_view(
        name=FV_NAME, version=1, query=query, labels=["is_fraud"],
        description="Fraud feature view; training dataset cctd" + SUFFIX,
    )
    print("feature view ready", flush=True)
except Exception as e:
    print("fv create error", e, flush=True)
    fv = fs.get_feature_view(name=FV_NAME, version=1)

X_train = X_test = y_train = y_test = None
try:
    X_train, X_test, y_train, y_test = fv.train_test_split(
        test_size=0.2,
        description="Fraud training dataset (train/test split).",
    )
    print("materialized td via fv:", X_train.shape, X_test.shape, flush=True)
except Exception as e:
    print("train_test_split error", e, flush=True)

def to_matrix(df):
    return df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float64")

# fall back to in-memory split if offline read was empty/failed
if X_train is None or len(X_train) == 0 or y_train is None:
    print("falling back to in-memory split", flush=True)
    from numpy.random import RandomState
    rs = RandomState(42)
    idx = rs.permutation(len(hist_f))
    cut = int(len(idx) * 0.8)
    tr, te = hist_f.iloc[idx[:cut]], hist_f.iloc[idx[cut:]]
    Xtr, Xte = to_matrix(tr), to_matrix(te)
    ytr, yte = tr["is_fraud"].astype(int).values, te["is_fraud"].astype(int).values
else:
    Xtr, Xte = to_matrix(X_train), to_matrix(X_test)
    ytr = np.asarray(y_train).astype(int).ravel()
    yte = np.asarray(y_test).astype(int).ravel()

print("train fraud rate", float(np.mean(ytr)), "test fraud rate", float(np.mean(yte)), flush=True)

# --------------------------------------------------------------- train + eval
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score)
from sklearn.utils.class_weight import compute_sample_weight

clf = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.06, max_depth=4,
    l2_regularization=1.0, random_state=42,
)
sw = compute_sample_weight(class_weight="balanced", y=ytr)
clf.fit(Xtr, ytr, sample_weight=sw)

p_test = clf.predict_proba(Xte)[:, 1]
auc = float(roc_auc_score(yte, p_test)) if len(np.unique(yte)) > 1 else 0.0
ap = float(average_precision_score(yte, p_test)) if len(np.unique(yte)) > 1 else 0.0
acc = float(accuracy_score(yte, (p_test >= 0.5).astype(int)))
f1 = float(f1_score(yte, (p_test >= 0.5).astype(int), zero_division=0))
metrics = {"roc_auc": auc, "average_precision": ap, "accuracy": acc, "f1": f1}
print("METRICS", metrics, flush=True)

# refit on ALL labelled history for best scoring model
X_all = to_matrix(hist_f)
y_all = hist_f["is_fraud"].astype(int).values
clf_full = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.06, max_depth=4,
    l2_regularization=1.0, random_state=42,
)
clf_full.fit(X_all, y_all, sample_weight=compute_sample_weight("balanced", y_all))

# --------------------------------------------------------- register the model
os.makedirs("model_dir", exist_ok=True)
joblib.dump(clf_full, "model_dir/model.pkl")
import json
with open("model_dir/features.json", "w") as fh:
    json.dump(FEATURES, fh)

mr = project.get_model_registry()
input_example = X_all.head(2).to_dict(orient="records")
try:
    model = mr.sklearn.create_model(
        name=MODEL_NAME, metrics=metrics,
        description="Credit-card fraud classifier (HistGradientBoosting).",
        feature_view=fv, input_example=input_example,
    )
    model.save("model_dir")
except Exception as e:
    print("sklearn create_model fallback:", e, flush=True)
    model = mr.python.create_model(name=MODEL_NAME, metrics=metrics,
                                   description="Fraud classifier")
    model.save("model_dir")
print("model registered", MODEL_NAME, flush=True)

# --------------------------------------------------- score + predictions table
Xs = to_matrix(score_f)
proba = clf_full.predict_proba(Xs)[:, 1]
proba = np.clip(proba, 0.0, 1.0).astype("float64")
pred_df = pd.DataFrame({
    "transaction_id": score_f["transaction_id"].astype(str).values,
    "fraud_probability": proba,
})
print("predictions", pred_df.shape, "mean prob", float(pred_df.fraud_probability.mean()), flush=True)

pred_fg = fs.get_or_create_feature_group(
    name=PRED_NAME, version=1,
    primary_key=["transaction_id"],
    online_enabled=True,
    description="Fraud probability predictions for scored transactions.",
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print("predictions inserted into", PRED_NAME, flush=True)
print("PIPELINE_DONE roc_auc=%.4f" % auc, flush=True)
