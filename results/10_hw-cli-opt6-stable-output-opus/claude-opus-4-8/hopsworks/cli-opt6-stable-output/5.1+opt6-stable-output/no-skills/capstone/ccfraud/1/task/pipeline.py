"""Full FTI fraud-detection pipeline. Runs as a Hopsworks job (platform-side)."""
import os
import numpy as np
import pandas as pd

import hopsworks

print(">>> logging in", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
dsapi = project.get_dataset_api()

# ----------------------------------------------------------------------------
# Load raw data from HopsFS
# ----------------------------------------------------------------------------
def fetch(remote, local):
    if os.path.exists(local):
        os.remove(local)
    p = dsapi.download(remote, local_path=local, overwrite=True)
    return p if (p and os.path.exists(p)) else local

tp = fetch("Resources/ccfraud/transactions.csv", "transactions.csv")
sp = fetch("Resources/ccfraud/score_transactions.csv", "score_transactions.csv")
train = pd.read_csv(tp)
score = pd.read_csv(sp)
print(">>> loaded", train.shape, score.shape, flush=True)

# ----------------------------------------------------------------------------
# Feature engineering (causal: each row only uses itself + prior rows per card)
# ----------------------------------------------------------------------------
train["__is_score"] = 0
score["__is_score"] = 1
if "is_fraud" not in score.columns:
    score["is_fraud"] = np.nan

df = pd.concat([train, score], ignore_index=True, sort=False)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
df["amount"] = df["amount"].astype(float)
df["lat"] = df["lat"].astype(float)
df["long"] = df["long"].astype(float)
df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)

g = df.groupby("cc_num")

# time since previous txn for this card
df["dt_prev_sec"] = g["datetime"].diff().dt.total_seconds()

# previous location
df["prev_lat"] = g["lat"].shift()
df["prev_long"] = g["long"].shift()


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p = np.pi / 180.0
    a = (0.5 - np.cos((lat2 - lat1) * p) / 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2)
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


df["geo_dist"] = haversine(df["prev_lat"], df["prev_long"], df["lat"], df["long"])
df["geo_speed"] = df["geo_dist"] / ((df["dt_prev_sec"] / 3600.0).clip(lower=1e-3))

# prior mean amount + usual location (expanding, shifted => excludes current row)
df["amt_cummean"] = (g["amount"].apply(lambda s: s.shift().expanding().mean())
                     .reset_index(level=0, drop=True))
df["mean_lat"] = (g["lat"].apply(lambda s: s.shift().expanding().mean())
                  .reset_index(level=0, drop=True))
df["mean_long"] = (g["long"].apply(lambda s: s.shift().expanding().mean())
                   .reset_index(level=0, drop=True))
df["dist_home"] = haversine(df["mean_lat"], df["mean_long"], df["lat"], df["long"])

# rolling time-window velocity / spend (df already sorted by cc_num, datetime)
ser = df.set_index("datetime").groupby("cc_num")["amount"]
df["cnt_1h"] = ser.rolling("1h").count().reset_index(level=0, drop=True).values
df["cnt_24h"] = ser.rolling("24h").count().reset_index(level=0, drop=True).values
df["sum_24h"] = ser.rolling("24h").sum().reset_index(level=0, drop=True).values

# time-of-day / amount signals
df["hour"] = df["datetime"].dt.hour.astype(float)
df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(float)
df["log_amount"] = np.log1p(df["amount"])
df["amt_cummean"] = df["amt_cummean"].fillna(df["amount"])
df["amt_over_mean"] = df["amount"] / (df["amt_cummean"] + 1e-6)

# category fraud rate (learned from TRAIN labels only -> no leakage)
base_rate = float(train["is_fraud"].mean())
cat_rate = train.groupby("category")["is_fraud"].mean()
df["cat_fraud_rate"] = df["category"].map(cat_rate).fillna(base_rate)

FEAT = ["amount", "log_amount", "hour", "is_night", "dt_prev_sec",
        "geo_dist", "geo_speed", "dist_home", "cnt_1h", "cnt_24h",
        "sum_24h", "amt_cummean", "amt_over_mean", "cat_fraud_rate",
        "lat", "long"]

# fill first-txn NaNs
df["dt_prev_sec"] = df["dt_prev_sec"].fillna(1e7)
for c in FEAT:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

train_f = df[df["__is_score"] == 0].copy()
score_f = df[df["__is_score"] == 1].copy()
train_f["is_fraud"] = train_f["is_fraud"].astype(int)
print(">>> features built", train_f.shape, score_f.shape,
      "fraud rate", base_rate, flush=True)

# ----------------------------------------------------------------------------
# 1) Feature group cctxnff6394
# ----------------------------------------------------------------------------
fg = fs.get_or_create_feature_group(
    name="cctxnff6394", version=1,
    primary_key=["transaction_id"], event_time="datetime",
    online_enabled=False,
    description="Engineered credit-card fraud features (velocity, geo, amount/hour).",
)
fg_cols = ["transaction_id", "datetime"] + FEAT + ["is_fraud"]
fg.insert(train_f[fg_cols], write_options={"wait_for_job": True})
print(">>> inserted feature group cctxnff6394", flush=True)

# ----------------------------------------------------------------------------
# 2) Feature view + training dataset cctdff6394
# ----------------------------------------------------------------------------
try:
    query = fg.select(FEAT + ["is_fraud"])
    fv = fs.get_or_create_feature_view(
        name="cctdff6394", version=1, query=query, labels=["is_fraud"],
    )
    fv.create_train_test_split(
        test_size=0.2,
        description="Fraud training dataset cctdff6394",
        write_options={"wait_for_job": True},
    )
    print(">>> created feature view + training dataset cctdff6394", flush=True)
except Exception as e:
    print(">>> WARN training-dataset step:", repr(e), flush=True)

# ----------------------------------------------------------------------------
# 3) Train classifier + register ccmodelff6394 with metrics
# ----------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

X = train_f[FEAT]
y = train_f["is_fraud"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.07, max_depth=None,
    l2_regularization=1.0, max_leaf_nodes=63, random_state=42,
)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:, 1]
auc = float(roc_auc_score(yte, proba))
acc = float(accuracy_score(yte, (proba >= 0.5).astype(int)))
ap = float(average_precision_score(yte, proba))
print(f">>> holdout ROC AUC={auc:.4f} ACC={acc:.4f} AP={ap:.4f}", flush=True)

# refit on ALL labelled data for final scoring model
clf.fit(X, y)

import joblib
mr = project.get_model_registry()
os.makedirs("ccmodel_dir", exist_ok=True)
joblib.dump(clf, "ccmodel_dir/model.pkl")

model_schema = None
try:
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    model_schema = ModelSchema(Schema(X), Schema(pd.DataFrame({"fraud_probability": [0.0]})))
except Exception as e:
    print(">>> WARN schema:", repr(e), flush=True)

metrics = {"roc_auc": auc, "accuracy": acc, "average_precision": ap}
try:
    m = mr.python.create_model(name="ccmodelff6394", metrics=metrics,
                               model_schema=model_schema,
                               description="Credit-card fraud classifier (HistGBDT).")
except TypeError:
    m = mr.python.create_model(name="ccmodelff6394", metrics=metrics,
                               description="Credit-card fraud classifier (HistGBDT).")
m.save("ccmodel_dir")
print(">>> registered model ccmodelff6394 with metrics", metrics, flush=True)

# ----------------------------------------------------------------------------
# 4) Score + write predictions to ccpredff6394 (online enabled)
# ----------------------------------------------------------------------------
score_proba = clf.predict_proba(score_f[FEAT])[:, 1]
pred_df = pd.DataFrame({
    "transaction_id": score_f["transaction_id"].astype(str).values,
    "fraud_probability": np.clip(score_proba.astype(float), 0.0, 1.0),
})
print(">>> scoring", pred_df.shape, "proba range",
      float(pred_df.fraud_probability.min()), float(pred_df.fraud_probability.max()),
      flush=True)

pred_fg = fs.get_or_create_feature_group(
    name="ccpredff6394", version=1,
    primary_key=["transaction_id"],
    online_enabled=True,
    description="Fraud probability predictions for scored transactions.",
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print(">>> wrote predictions feature group ccpredff6394 (online enabled)", flush=True)
print(">>> PIPELINE COMPLETE", flush=True)
