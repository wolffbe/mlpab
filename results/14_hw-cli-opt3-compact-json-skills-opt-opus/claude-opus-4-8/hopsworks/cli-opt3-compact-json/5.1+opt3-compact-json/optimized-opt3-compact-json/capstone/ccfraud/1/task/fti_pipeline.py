"""Full FTI fraud pipeline — runs entirely on the Hopsworks platform as a job."""
import os
import numpy as np
import pandas as pd

import hopsworks

print(">>> logging in", flush=True)
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()
dsapi = project.get_dataset_api()

# ---------------------------------------------------------------- load data
for f in ["transactions.csv", "score_transactions.csv"]:
    if os.path.exists(f):
        os.remove(f)
    dsapi.download(f"Resources/ccfraud/{f}", f, overwrite=True)

train = pd.read_csv("transactions.csv")
score = pd.read_csv("score_transactions.csv")
print(f">>> train={train.shape} score={score.shape}", flush=True)

train["datetime"] = pd.to_datetime(train["datetime"])
score["datetime"] = pd.to_datetime(score["datetime"])
train["is_train"] = 1
score["is_train"] = 0
score["is_fraud"] = np.nan

alld = pd.concat([train, score], ignore_index=True)
alld = alld.sort_values(["cc_num", "datetime"]).reset_index(drop=True)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p = np.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------- feature engineering
gb = alld.groupby("cc_num", group_keys=False)
alld["amt_cummean"] = gb["amount"].apply(lambda s: s.shift(1).expanding().mean())
alld["amt_cumstd"] = gb["amount"].apply(lambda s: s.shift(1).expanding().std())
alld["lat_cummean"] = gb["lat"].apply(lambda s: s.shift(1).expanding().mean())
alld["long_cummean"] = gb["long"].apply(lambda s: s.shift(1).expanding().mean())
alld["prev_lat"] = gb["lat"].shift(1)
alld["prev_long"] = gb["long"].shift(1)
alld["prev_dt"] = gb["datetime"].shift(1)

alld["time_since_prev"] = (alld["datetime"] - alld["prev_dt"]).dt.total_seconds()
alld["time_since_prev"] = alld["time_since_prev"].fillna(1e6)

alld["amount_zscore"] = (alld["amount"] - alld["amt_cummean"]) / alld["amt_cumstd"]
alld["amount_zscore"] = alld["amount_zscore"].replace([np.inf, -np.inf], 0).fillna(0)
alld["amt_ratio"] = (alld["amount"] / alld["amt_cummean"]).replace([np.inf, -np.inf], 1).fillna(1)

alld["dist_from_home"] = haversine(alld["lat"], alld["long"],
                                   alld["lat_cummean"], alld["long_cummean"]).fillna(0)
alld["dist_from_prev"] = haversine(alld["lat"], alld["long"],
                                   alld["prev_lat"], alld["prev_long"]).fillna(0)

alld["hour"] = alld["datetime"].dt.hour
alld["day_of_week"] = alld["datetime"].dt.dayofweek
alld["log_amount"] = np.log1p(alld["amount"])

# transaction velocity: count of same-card txns in prior 24h
alld["ts"] = alld["datetime"].astype("int64") // 10**9
counts = np.zeros(len(alld), dtype=float)
for _, idx in alld.groupby("cc_num").groups.items():
    idx = list(idx)
    times = alld.loc[idx, "ts"].values
    for i, t in enumerate(times):
        lo = np.searchsorted(times, t - 86400, side="left")
        counts[alld.index.get_loc(idx[i])] = i - lo
alld["txn_count_24h"] = counts

# category fraud rate (target encoding from TRAIN only)
tr_mask = alld["is_train"] == 1
global_rate = alld.loc[tr_mask, "is_fraud"].mean()
cat_rate = alld.loc[tr_mask].groupby("category")["is_fraud"].mean().to_dict()
alld["category_fraud_rate"] = alld["category"].map(cat_rate).fillna(global_rate)

FEATURES = ["amount", "log_amount", "hour", "day_of_week", "time_since_prev",
            "amount_zscore", "amt_ratio", "dist_from_home", "dist_from_prev",
            "txn_count_24h", "category_fraud_rate"]

for c in FEATURES:
    alld[c] = alld[c].astype("float64")

# ------------------------------------------------------------- feature group
fg_df = alld[tr_mask].copy()
fg_cols = ["transaction_id", "cc_num", "datetime", "is_fraud"] + FEATURES
fg_df = fg_df[fg_cols].copy()
fg_df["transaction_id"] = fg_df["transaction_id"].astype(str)
fg_df["cc_num"] = fg_df["cc_num"].astype("int64")
fg_df["is_fraud"] = fg_df["is_fraud"].astype("int64")
print(f">>> fg_df={fg_df.shape}", flush=True)

fg = fs.get_or_create_feature_group(
    name="cctxna046f9", version=1,
    description="Engineered credit-card fraud features",
    primary_key=["transaction_id"], event_time="datetime",
    online_enabled=False,
)
fg.insert(fg_df, write_options={"wait_for_job": True})
print(">>> feature group inserted", flush=True)

# ----------------------------------------------------------- feature view + TD
try:
    existing = fs.get_feature_view(name="cctda046f9", version=1)
    existing.delete()
except Exception:
    pass

fv = fs.create_feature_view(
    name="cctda046f9", version=1,
    description="Fraud training dataset feature view",
    query=fg.select(FEATURES + ["is_fraud"]),
    labels=["is_fraud"],
)
print(">>> feature view created", flush=True)

td_version, td_job = fv.create_train_test_split(
    test_size=0.2, description="cctda046f9 training dataset",
    write_options={"wait_for_job": True},
)
print(f">>> training dataset version={td_version}", flush=True)

# ---------------------------------------------------------------- train model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

data = alld[tr_mask].copy()
X = data[FEATURES].values
y = data["is_fraud"].astype(int).values

# time-ordered holdout for honest AUC
order = np.argsort(data["ts"].values)
X, y = X[order], y[order]
cut = int(len(X) * 0.8)
X_tr, X_te = X[:cut], X[cut:]
y_tr, y_te = y[:cut], y[cut:]

clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                                 subsample=0.9, random_state=42)
clf.fit(X_tr, y_tr)
proba_te = clf.predict_proba(X_te)[:, 1]
auc = float(roc_auc_score(y_te, proba_te))
pred_te = (proba_te >= 0.5).astype(int)
metrics = {
    "roc_auc": auc,
    "precision": float(precision_score(y_te, pred_te, zero_division=0)),
    "recall": float(recall_score(y_te, pred_te, zero_division=0)),
    "f1": float(f1_score(y_te, pred_te, zero_division=0)),
}
print(f">>> metrics={metrics}", flush=True)

# refit on all training data for best scoring
clf.fit(X, y)

# ------------------------------------------------------------- register model
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

model_dir = "ccmodela046f9_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(clf, os.path.join(model_dir, "model.pkl"))

input_schema = Schema(data[FEATURES])
output_schema = Schema(pd.DataFrame({"fraud_probability": [0.0]}))
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

model = mr.python.create_model(
    name="ccmodela046f9",
    metrics=metrics,
    description="Credit-card fraud classifier (GradientBoosting)",
    input_example=data[FEATURES].head(2),
    model_schema=model_schema,
)
model.save(model_dir)
print(f">>> model registered version={model.version}", flush=True)

# ------------------------------------------------------------------- score
score_rows = alld[alld["is_train"] == 0].copy()
proba = clf.predict_proba(score_rows[FEATURES].values)[:, 1]
pred_df = pd.DataFrame({
    "transaction_id": score_rows["transaction_id"].astype(str).values,
    "cc_num": score_rows["cc_num"].astype("int64").values,
    "fraud_probability": np.clip(proba, 0.0, 1.0).astype("float64"),
})
print(f">>> pred_df={pred_df.shape} prob[min,max]=[{pred_df.fraud_probability.min():.3f},{pred_df.fraud_probability.max():.3f}]", flush=True)

pred_fg = fs.get_or_create_feature_group(
    name="ccpreda046f9", version=1,
    description="Fraud probability predictions for scoring transactions",
    primary_key=["transaction_id"],
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print(">>> predictions feature group inserted", flush=True)
print(">>> DONE", flush=True)
