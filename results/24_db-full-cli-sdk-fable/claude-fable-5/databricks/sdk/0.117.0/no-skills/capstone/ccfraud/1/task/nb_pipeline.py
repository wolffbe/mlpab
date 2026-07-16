# Databricks notebook source
# Full FTI pipeline: features -> training dataset -> train/register model -> score
import json

import numpy as np
import pandas as pd

CAT = "workspace"
SCH = "mlpab67db84"
VOL = f"/Volumes/{CAT}/{SCH}/raw"

train_raw = spark.read.csv(f"{VOL}/transactions.csv", header=True, inferSchema=True).toPandas()
score_raw = spark.read.csv(f"{VOL}/score_transactions.csv", header=True, inferSchema=True).toPandas()
print("train:", train_raw.shape, "score:", score_raw.shape)

train_raw["datetime"] = pd.to_datetime(train_raw["datetime"], utc=True)
score_raw["datetime"] = pd.to_datetime(score_raw["datetime"], utc=True)
train_raw["is_train"] = 1
score_raw["is_train"] = 0
score_raw["is_fraud"] = np.nan

comb = pd.concat([train_raw, score_raw], ignore_index=True)
comb = comb.sort_values(["cc_num", "datetime"], kind="mergesort").reset_index(drop=True)

# COMMAND ----------
# Feature engineering (labels are only ever taken from the training slice)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


comb["hour"] = comb["datetime"].dt.hour
comb["dayofweek"] = comb["datetime"].dt.dayofweek
comb["is_night"] = ((comb["hour"] >= 0) & (comb["hour"] <= 5)).astype(int)
comb["log_amount"] = np.log1p(comb["amount"])

g = comb.groupby("cc_num")
comb["secs_since_prev"] = (comb["datetime"] - g["datetime"].shift(1)).dt.total_seconds()
comb["dist_prev_km"] = haversine_km(comb["lat"], comb["long"], g["lat"].shift(1), g["long"].shift(1))
comb["speed_kmh"] = (comb["dist_prev_km"] / (comb["secs_since_prev"] / 3600.0)).replace([np.inf, -np.inf], np.nan)

# transaction velocity per card: counts in trailing 1h / 24h windows
epoch = comb["datetime"].astype("int64").to_numpy() // 10**9
cnt_1h = np.zeros(len(comb))
cnt_24h = np.zeros(len(comb))
for idx in comb.groupby("cc_num").indices.values():
    t = epoch[idx]
    pos = np.arange(len(t))
    cnt_1h[idx] = pos - np.searchsorted(t, t - 3600, side="left")
    cnt_24h[idx] = pos - np.searchsorted(t, t - 86400, side="left")
comb["cnt_1h"] = cnt_1h
comb["cnt_24h"] = cnt_24h

# card profiles from labelled history only
prof = train_raw.groupby("cc_num").agg(
    card_amt_mean=("amount", "mean"),
    card_amt_std=("amount", "std"),
    home_lat=("lat", "median"),
    home_long=("long", "median"),
    card_n=("amount", "size"),
).reset_index()
comb = comb.merge(prof, on="cc_num", how="left")
comb["amt_z"] = (comb["amount"] - comb["card_amt_mean"]) / (comb["card_amt_std"].fillna(0.0) + 1.0)
comb["dist_home_km"] = haversine_km(comb["lat"], comb["long"], comb["home_lat"], comb["home_long"])

# smoothed target encoding of category/merchant from labelled history only
global_rate = float(train_raw["is_fraud"].mean())


def target_encode(col, k):
    s = train_raw.groupby(col)["is_fraud"].agg(["sum", "count"])
    return (s["sum"] + k * global_rate) / (s["count"] + k)


comb["cat_te"] = comb["category"].map(target_encode("category", 20)).fillna(global_rate)
comb["mer_te"] = comb["merchant"].map(target_encode("merchant", 10)).fillna(global_rate)

FEATURES = [
    "amount", "log_amount", "hour", "dayofweek", "is_night",
    "secs_since_prev", "dist_prev_km", "speed_kmh", "cnt_1h", "cnt_24h",
    "card_amt_mean", "card_amt_std", "card_n", "amt_z", "dist_home_km",
    "cat_te", "mer_te",
]
for c in FEATURES:
    comb[c] = comb[c].astype(float)

# COMMAND ----------
# Feature group cctxn9d5953: engineered features for every transaction

fg = comb[["transaction_id", "cc_num", "datetime"] + FEATURES + ["is_fraud", "is_train"]]
spark.createDataFrame(fg).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{CAT}.{SCH}.cctxn9d5953"
)
spark.sql(f"ALTER TABLE {CAT}.{SCH}.cctxn9d5953 ALTER COLUMN transaction_id SET NOT NULL")
spark.sql(
    f"ALTER TABLE {CAT}.{SCH}.cctxn9d5953 ADD CONSTRAINT cctxn9d5953_pk PRIMARY KEY(transaction_id)"
)
print("feature group written:", spark.table(f"{CAT}.{SCH}.cctxn9d5953").count())

# COMMAND ----------
# Training dataset cctd9d5953 assembled from the feature group

td_sdf = spark.table(f"{CAT}.{SCH}.cctxn9d5953").where("is_train = 1").select(
    "transaction_id", "datetime", *FEATURES, "is_fraud"
)
td_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CAT}.{SCH}.cctd9d5953")
td = spark.table(f"{CAT}.{SCH}.cctd9d5953").toPandas()
print("training dataset:", td.shape, "fraud rate:", td["is_fraud"].mean())

# COMMAND ----------
# Train, evaluate on a time-based holdout, register with metrics

import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

td = td.sort_values("datetime").reset_index(drop=True)
X = td[FEATURES]
y = td["is_fraud"].astype(int)
split = int(len(td) * 0.8)
X_tr, y_tr = X.iloc[:split], y.iloc[:split]
X_va, y_va = X.iloc[split:], y.iloc[split:]

params = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=30, random_state=42)
clf = HistGradientBoostingClassifier(**params)
clf.fit(X_tr, y_tr)
va_proba = clf.predict_proba(X_va)[:, 1]
val_auc = float(roc_auc_score(y_va, va_proba))
val_ap = float(average_precision_score(y_va, va_proba))
print("holdout ROC AUC:", val_auc, "AP:", val_ap)

final = HistGradientBoostingClassifier(**params)
final.fit(X, y)

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(run_name="ccfraud9d5953") as run:
    mlflow.log_params(params)
    mlflow.log_param("n_features", len(FEATURES))
    mlflow.log_metric("holdout_roc_auc", val_auc)
    mlflow.log_metric("holdout_average_precision", val_ap)
    mlflow.log_metric("train_rows", len(td))
    sig = infer_signature(X.head(50), final.predict_proba(X.head(50))[:, 1])
    mlflow.sklearn.log_model(
        final,
        name="model",
        signature=sig,
        input_example=X.head(5),
        registered_model_name=f"{CAT}.{SCH}.ccmodel9d5953",
    )
    run_id = run.info.run_id
print("model registered:", f"{CAT}.{SCH}.ccmodel9d5953", "run:", run_id)

# COMMAND ----------
# Score every row of score_transactions.csv into ccpred9d5953 (PK + CDF for online sync)

score_feat = comb[comb["is_train"] == 0]
proba = final.predict_proba(score_feat[FEATURES])[:, 1]
pred = pd.DataFrame(
    {"transaction_id": score_feat["transaction_id"].values, "fraud_probability": np.clip(proba, 0.0, 1.0)}
)
assert len(pred) == len(score_raw)

spark.sql(
    f"""CREATE OR REPLACE TABLE {CAT}.{SCH}.ccpred9d5953 (
        transaction_id STRING NOT NULL,
        fraud_probability DOUBLE,
        CONSTRAINT ccpred9d5953_pk PRIMARY KEY(transaction_id)
    ) TBLPROPERTIES (delta.enableChangeDataFeed = true)"""
)
spark.createDataFrame(pred).write.mode("append").saveAsTable(f"{CAT}.{SCH}.ccpred9d5953")
n_pred = spark.table(f"{CAT}.{SCH}.ccpred9d5953").count()
print("predictions written:", n_pred)

dbutils.notebook.exit(
    json.dumps(
        {
            "holdout_roc_auc": val_auc,
            "holdout_average_precision": val_ap,
            "n_pred": n_pred,
            "run_id": run_id,
            "pred_min": float(pred["fraud_probability"].min()),
            "pred_max": float(pred["fraud_probability"].max()),
        }
    )
)
