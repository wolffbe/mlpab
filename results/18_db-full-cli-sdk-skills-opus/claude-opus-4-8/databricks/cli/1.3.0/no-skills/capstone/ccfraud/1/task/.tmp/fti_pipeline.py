# Databricks notebook source
# FTI fraud-detection pipeline — runs entirely on the Databricks platform.
import math
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

CATALOG = "workspace"
SCHEMA = "mlpab943bc4"
PREFIX = "mlpab943bc4"
USER = "benedict@logicalclocks.com"
FQ = f"{CATALOG}.{SCHEMA}"

FG = f"{FQ}.cctxn2dbe0a"        # feature group (engineered features + label)
TD = f"{FQ}.cctd2dbe0a"         # training dataset
PRED = f"{FQ}.ccpred2dbe0a"     # predictions feature table
MODEL = f"{FQ}.ccmodel2dbe0a"   # registered model

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{USER}/{PREFIX}/ccfraud_experiment")

fe = FeatureEngineeringClient()

# COMMAND ----------
# Load raw data from the volume
base = f"/Volumes/{CATALOG}/{SCHEMA}/data"
train_raw = pd.read_csv(f"{base}/transactions.csv")
score_raw = pd.read_csv(f"{base}/score_transactions.csv")
print("train", train_raw.shape, "score", score_raw.shape)

# COMMAND ----------
# ---- Feature engineering ----
def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(a, 0, 1)))

CATEGORIES = sorted(train_raw["category"].dropna().unique().tolist())

# Card profile stats computed from labelled HISTORY only (usual behaviour of a card)
prof = train_raw.groupby("cc_num").agg(
    card_mean_amt=("amount", "mean"),
    card_std_amt=("amount", "std"),
    card_cnt=("amount", "count"),
    card_mean_lat=("lat", "mean"),
    card_mean_long=("long", "mean"),
).reset_index()
glob_mean_amt = train_raw["amount"].mean()
glob_std_amt = train_raw["amount"].std()
glob_lat = train_raw["lat"].mean()
glob_long = train_raw["long"].mean()

def engineer(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values(["cc_num", "datetime"]).reset_index(drop=True)
    df = df.merge(prof, on="cc_num", how="left")
    df["card_mean_amt"] = df["card_mean_amt"].fillna(glob_mean_amt)
    df["card_std_amt"] = df["card_std_amt"].fillna(glob_std_amt).replace(0, glob_std_amt)
    df["card_cnt"] = df["card_cnt"].fillna(0)
    df["card_mean_lat"] = df["card_mean_lat"].fillna(glob_lat)
    df["card_mean_long"] = df["card_mean_long"].fillna(glob_long)

    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_night"] = (df["hour"] < 6).astype(int)
    df["amt_z"] = (df["amount"] - df["card_mean_amt"]) / (df["card_std_amt"] + 1e-6)
    df["amt_ratio"] = df["amount"] / (df["card_mean_amt"] + 1.0)
    df["geo_dist"] = haversine(df["lat"], df["long"], df["card_mean_lat"], df["card_mean_long"])

    # velocity within card (ordered by time inside this dataset)
    g = df.groupby("cc_num")
    df["time_since_prev_min"] = g["datetime"].diff().dt.total_seconds() / 60.0
    df["time_since_prev_min"] = df["time_since_prev_min"].fillna(100000.0)
    # rolling counts in last 1h / 24h per card
    def roll_counts(sub):
        t = sub.set_index("datetime").sort_index()
        c1 = t["amount"].rolling("1h").count() - 1
        c24 = t["amount"].rolling("24h").count() - 1
        return pd.DataFrame({"txns_1h": c1.values, "txns_24h": c24.values}, index=sub.index)
    rc = df.groupby("cc_num", group_keys=False).apply(roll_counts)
    df["txns_1h"] = rc["txns_1h"].fillna(0).values
    df["txns_24h"] = rc["txns_24h"].fillna(0).values

    # category one-hot
    for c in CATEGORIES:
        df[f"cat_{c}"] = (df["category"] == c).astype(int)
    return df

FEATS = ["amount", "log_amount", "hour", "dayofweek", "is_night",
         "card_mean_amt", "card_std_amt", "card_cnt", "amt_z", "amt_ratio",
         "geo_dist", "time_since_prev_min", "txns_1h", "txns_24h"] + [f"cat_{c}" for c in CATEGORIES]

train_fe = engineer(train_raw)
score_fe = engineer(score_raw)
print("engineered feats:", len(FEATS))

# COMMAND ----------
# ---- Write feature group cctxn2dbe0a (features + label, keyed by transaction_id) ----
fg_cols = ["transaction_id"] + FEATS + ["is_fraud"]
fg_pdf = train_fe[fg_cols].copy()
fg_sdf = spark.createDataFrame(fg_pdf)

spark.sql(f"DROP TABLE IF EXISTS {FG}")
fe.create_table(
    name=FG,
    primary_keys=["transaction_id"],
    df=fg_sdf,
    description="Engineered fraud features per transaction (labelled history).",
)
print("feature group created:", FG)

# COMMAND ----------
# ---- Assemble training dataset cctd2dbe0a from the feature group via FeatureLookup ----
spine = train_fe[["transaction_id", "is_fraud"]].copy()
spine_sdf = spark.createDataFrame(spine)
lookups = [FeatureLookup(table_name=FG, lookup_key="transaction_id", feature_names=FEATS)]
training_set = fe.create_training_set(
    df=spine_sdf, feature_lookups=lookups, label="is_fraud", exclude_columns=[]
)
td_sdf = training_set.load_df()
td_sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TD)
print("training dataset created:", TD)

# COMMAND ----------
# ---- Train classifier + log metrics + register ccmodel2dbe0a ----
td = spark.table(TD).toPandas()
X = td[FEATS].astype(float).values
y = td["is_fraud"].astype(int).values

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.08, max_depth=None, max_leaf_nodes=63,
    l2_regularization=1.0, validation_fraction=0.1, random_state=42,
    class_weight="balanced",
)
clf.fit(Xtr, ytr)
va_proba = clf.predict_proba(Xva)[:, 1]
auc = float(roc_auc_score(yva, va_proba))
ap = float(average_precision_score(yva, va_proba))
f1 = float(f1_score(yva, (va_proba >= 0.5).astype(int)))
print(f"held-out ROC AUC={auc:.4f} AP={ap:.4f} F1={f1:.4f}")

# refit on all data for final model
final = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
    l2_regularization=1.0, validation_fraction=0.1, random_state=42,
    class_weight="balanced",
)
final.fit(X, y)

import pandas as _pd
sig_in = _pd.DataFrame(X[:5], columns=FEATS)
from mlflow.models.signature import infer_signature
sig = infer_signature(sig_in, final.predict_proba(X[:5])[:, 1])

with mlflow.start_run(run_name="ccfraud_hgb") as run:
    mlflow.log_params({"model": "HistGradientBoostingClassifier", "max_iter": 400,
                       "learning_rate": 0.08, "max_leaf_nodes": 63, "n_features": len(FEATS)})
    mlflow.log_metrics({"holdout_roc_auc": auc, "holdout_avg_precision": ap, "holdout_f1": f1})
    mlflow.sklearn.log_model(final, artifact_path="model", signature=sig,
                             registered_model_name=MODEL, input_example=sig_in)
    print("run_id", run.info.run_id)
print("model registered:", MODEL)

# COMMAND ----------
# ---- Score every row of score_transactions and write ccpred2dbe0a ----
Xs = score_fe[FEATS].astype(float).values
proba = final.predict_proba(Xs)[:, 1]
pred_pdf = pd.DataFrame({
    "transaction_id": score_fe["transaction_id"].values,
    "fraud_probability": np.clip(proba, 0.0, 1.0).astype(float),
})
pred_sdf = spark.createDataFrame(pred_pdf)

spark.sql(f"DROP TABLE IF EXISTS {PRED}")
fe.create_table(
    name=PRED,
    primary_keys=["transaction_id"],
    df=pred_sdf,
    description="Fraud probability predictions for scored transactions.",
)
print("predictions table created:", PRED, "rows:", pred_pdf.shape[0])
print("prob range:", float(pred_pdf.fraud_probability.min()), float(pred_pdf.fraud_probability.max()))

dbutils.notebook.exit(f"AUC={auc:.4f};rows={pred_pdf.shape[0]}")
