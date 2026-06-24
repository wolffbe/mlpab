"""
Full FTI fraud-detection pipeline, executed ON the Hopsworks platform as a job.

Feature -> Training -> Inference:
  * engineer fraud features into feature group  cctxn91b9a2
  * build feature view + materialize training dataset cctd91b9a2
  * train + register sklearn classifier            ccmodel91b9a2  (with metrics)
  * score score_transactions.csv into online+offline FG ccpred91b9a2
"""
import os
import numpy as np
import pandas as pd

import hopsworks

PROJ = "ccfraud"
FG_FEAT = "cctxn91b9a2"
FV_NAME = "cctd91b9a2"
MODEL_NAME = "ccmodel91b9a2"
FG_PRED = "ccpred91b9a2"

print(">>> logging in", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
dsapi = project.get_dataset_api()

# ---------------------------------------------------------------- load raw data
def load(remote, local):
    if os.path.exists(local):
        os.remove(local)
    p = dsapi.download("Resources/ccfraud/" + remote, local, overwrite=True)
    return pd.read_csv(p)

train = load("transactions.csv", "transactions.csv")
score = load("score_transactions.csv", "score_transactions.csv")
print(">>> train", train.shape, "score", score.shape, flush=True)

train["__split"] = "train"
score["__split"] = "score"
score["is_fraud"] = np.nan
allrows = pd.concat([train, score], ignore_index=True)
allrows["datetime"] = pd.to_datetime(allrows["datetime"], utc=True)
allrows = allrows.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

# ---------------------------------------------------- card-level stats from TRAIN
tr = allrows[allrows["__split"] == "train"]
home = tr.groupby("cc_num").agg(home_lat=("lat", "mean"),
                                home_long=("long", "mean"),
                                card_mean_amt=("amount", "mean"),
                                card_std_amt=("amount", "std")).reset_index()
global_mean_amt = float(tr["amount"].mean())
home["card_mean_amt"] = home["card_mean_amt"].fillna(global_mean_amt)
home["card_std_amt"] = home["card_std_amt"].fillna(0.0)

# smoothed fraud-rate per category (train only) -> a leakage-free score feature
glob_rate = float(tr["is_fraud"].mean())
catg = tr.groupby("category").agg(n=("is_fraud", "size"),
                                  s=("is_fraud", "sum")).reset_index()
K = 20.0
catg["cat_fraud_rate"] = (catg["s"] + K * glob_rate) / (catg["n"] + K)
cat_map = dict(zip(catg["category"], catg["cat_fraud_rate"]))

allrows = allrows.merge(home, on="cc_num", how="left")
allrows["home_lat"] = allrows["home_lat"].fillna(allrows["lat"])
allrows["home_long"] = allrows["home_long"].fillna(allrows["long"])
allrows["card_mean_amt"] = allrows["card_mean_amt"].fillna(global_mean_amt)
allrows["card_std_amt"] = allrows["card_std_amt"].fillna(0.0)
allrows["cat_fraud_rate"] = allrows["category"].map(cat_map).fillna(glob_rate)


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ----------------------------------------------- per-card sequential signals
g = allrows.groupby("cc_num", sort=False)
prev_dt = g["datetime"].shift(1)
allrows["time_since_prev_s"] = (allrows["datetime"] - prev_dt).dt.total_seconds()
allrows["time_since_prev_s"] = allrows["time_since_prev_s"].fillna(1e6).clip(0, 1e6)
prev_lat = g["lat"].shift(1)
prev_long = g["long"].shift(1)
allrows["dist_from_prev_km"] = haversine(prev_lat, prev_long,
                                         allrows["lat"], allrows["long"]).fillna(0.0)
# velocity: km per hour vs previous txn
allrows["speed_kmph"] = (allrows["dist_from_prev_km"] /
                         (allrows["time_since_prev_s"] / 3600.0 + 1e-3)).clip(0, 1e5)

# trailing 1h transaction count per card
allrows = allrows.sort_values(["cc_num", "datetime"]).reset_index(drop=True)
cnt1h = []
from collections import deque, defaultdict
buf = defaultdict(deque)
for cc, dt in zip(allrows["cc_num"].values, allrows["datetime"].values):
    dq = buf[cc]
    t = pd.Timestamp(dt)
    while dq and (t - dq[0]).total_seconds() > 3600:
        dq.popleft()
    cnt1h.append(len(dq))
    dq.append(t)
allrows["txn_count_1h"] = cnt1h

# distance from card home + amount signals
allrows["dist_from_home_km"] = haversine(allrows["home_lat"], allrows["home_long"],
                                         allrows["lat"], allrows["long"])
allrows["amt_over_cardmean"] = allrows["amount"] / (allrows["card_mean_amt"] + 1.0)
allrows["amt_z"] = (allrows["amount"] - allrows["card_mean_amt"]) / (allrows["card_std_amt"] + 1.0)
allrows["log_amount"] = np.log1p(allrows["amount"])
allrows["hour"] = allrows["datetime"].dt.hour
allrows["dow"] = allrows["datetime"].dt.dayofweek
allrows["is_night"] = ((allrows["hour"] < 6) | (allrows["hour"] >= 22)).astype(int)
allrows["event_time"] = allrows["datetime"].astype("int64") // 10**6  # epoch ms

FEATURES = ["amount", "log_amount", "hour", "dow", "is_night",
            "time_since_prev_s", "dist_from_prev_km", "speed_kmph",
            "txn_count_1h", "dist_from_home_km", "amt_over_cardmean",
            "amt_z", "cat_fraud_rate", "card_mean_amt"]

for c in FEATURES:
    allrows[c] = allrows[c].astype("float64")

allrows["transaction_id"] = allrows["transaction_id"].astype(str)

train_fe = allrows[allrows["__split"] == "train"].copy()
score_fe = allrows[allrows["__split"] == "score"].copy()
train_fe["is_fraud"] = train_fe["is_fraud"].astype(int)
print(">>> features built. train_fe", train_fe.shape, "score_fe", score_fe.shape, flush=True)

# ------------------------------------------------------- FEATURE GROUP cctxn91b9a2
fg_cols = ["transaction_id", "event_time"] + FEATURES + ["is_fraud"]
fg = fs.get_or_create_feature_group(
    name=FG_FEAT, version=1,
    description="Engineered credit-card fraud features (velocity, geo, amount/hour signals)",
    primary_key=["transaction_id"], event_time="event_time",
    online_enabled=False,
)
fg.insert(train_fe[fg_cols], write_options={"wait_for_job": True})
print(">>> inserted feature group", FG_FEAT, flush=True)

# ----------------------------------------------- FEATURE VIEW + TRAINING DATASET
try:
    query = fg.select(FEATURES + ["is_fraud"])
    try:
        fv = fs.get_feature_view(name=FV_NAME, version=1)
    except Exception:
        fv = fs.create_feature_view(name=FV_NAME, version=1, query=query,
                                    labels=["is_fraud"],
                                    description="Fraud training feature view")
    fv.create_train_test_split(test_size=0.2, write_options={"wait_for_job": True})
    print(">>> materialized training dataset", FV_NAME, flush=True)
except Exception as e:
    print("!!! training dataset materialization warning:", repr(e), flush=True)

# ------------------------------------------------------------ TRAIN (in-memory)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

X = train_fe[FEATURES].values
y = train_fe["is_fraud"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = GradientBoostingClassifier(random_state=42)
clf.fit(Xtr, ytr)
proba_te = clf.predict_proba(Xte)[:, 1]
auc = float(roc_auc_score(yte, proba_te))
ap = float(average_precision_score(yte, proba_te))
acc = float(accuracy_score(yte, (proba_te >= 0.5).astype(int)))
print(f">>> holdout AUC={auc:.4f} AP={ap:.4f} ACC={acc:.4f}", flush=True)

# refit on all training data for the production model
clf_full = GradientBoostingClassifier(random_state=42)
clf_full.fit(X, y)

# ------------------------------------------------------- REGISTER MODEL ccmodel91b9a2
import joblib
mdir = "ccmodel_dir"
os.makedirs(mdir, exist_ok=True)
joblib.dump(clf_full, os.path.join(mdir, "model.pkl"))
input_example = train_fe[FEATURES].head(2).to_dict(orient="records")

mr = project.get_model_registry()
metrics = {"roc_auc": auc, "average_precision": ap, "accuracy": acc}
try:
    model = mr.sklearn.create_model(name=MODEL_NAME, metrics=metrics,
                                    input_example=input_example,
                                    description="GradientBoosting fraud classifier")
except Exception:
    model = mr.python.create_model(name=MODEL_NAME, metrics=metrics,
                                   input_example=input_example,
                                   description="GradientBoosting fraud classifier")
model.save(mdir)
print(">>> registered model", MODEL_NAME, "metrics", metrics, flush=True)

# ----------------------------------------------------------------- SCORE + WRITE
score_proba = clf_full.predict_proba(score_fe[FEATURES].values)[:, 1]
preds = pd.DataFrame({
    "transaction_id": score_fe["transaction_id"].astype(str).values,
    "fraud_probability": np.clip(score_proba, 0.0, 1.0).astype("float64"),
})
print(">>> preds", preds.shape, "range",
      float(preds.fraud_probability.min()), float(preds.fraud_probability.max()), flush=True)

fg_pred = fs.get_or_create_feature_group(
    name=FG_PRED, version=1,
    description="Fraud probability predictions for scored transactions",
    primary_key=["transaction_id"],
    online_enabled=True,
)
fg_pred.insert(preds, write_options={"wait_for_job": True})
print(">>> inserted predictions feature group", FG_PRED, preds.shape, flush=True)
print(">>> PIPELINE DONE auc=", auc, flush=True)
