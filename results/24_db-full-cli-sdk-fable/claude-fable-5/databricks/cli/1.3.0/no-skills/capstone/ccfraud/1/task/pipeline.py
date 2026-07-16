# Databricks notebook source
# CC fraud FTI pipeline: features -> training dataset -> model -> predictions
import numpy as np
import pandas as pd

SCHEMA = "workspace.mlpab59b6a9"
VOL = "/Volumes/workspace/mlpab59b6a9/raw"
FG_TABLE = f"{SCHEMA}.cctxn9d5953"
TD_TABLE = f"{SCHEMA}.cctd9d5953"
PRED_TABLE = f"{SCHEMA}.ccpred9d5953"
MODEL_NAME = f"{SCHEMA}.ccmodel9d5953"

train_raw = pd.read_csv(f"{VOL}/transactions.csv")
score_raw = pd.read_csv(f"{VOL}/score_transactions.csv")
score_raw["is_fraud"] = np.nan
print(len(train_raw), len(score_raw))

# COMMAND ----------

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_seq_features(df):
    """Per-card sequential features (velocity, gaps, movement). Past-only."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["ts"] = df["datetime"].astype("int64") // 10 ** 9
    df = df.sort_values(["cc_num", "ts", "transaction_id"]).reset_index(drop=True)
    g = df.groupby("cc_num", sort=False)
    df["gap_sec"] = df["ts"] - g["ts"].shift(1)
    prev_lat = g["lat"].shift(1)
    prev_long = g["long"].shift(1)
    df["dist_prev_km"] = haversine(df["lat"], df["long"], prev_lat, prev_long)
    df["speed_kmh"] = df["dist_prev_km"] / np.maximum(df["gap_sec"] / 3600.0, 1.0 / 60)

    def past_count(s, w):
        v = s.values
        return np.arange(len(v)) - np.searchsorted(v, v - w, side="left")

    df["cnt_1h"] = g["ts"].transform(lambda s: past_count(s, 3600))
    df["cnt_24h"] = g["ts"].transform(lambda s: past_count(s, 86400))
    df["cnt_7d"] = g["ts"].transform(lambda s: past_count(s, 7 * 86400))
    df["amt_mean_prev"] = g["amount"].transform(lambda s: s.expanding().mean().shift(1))
    df["amt_std_prev"] = g["amount"].transform(lambda s: s.expanding().std().shift(1))
    df["amt_z"] = (df["amount"] - df["amt_mean_prev"]) / df["amt_std_prev"].replace(0, np.nan)
    df["amt_ratio"] = df["amount"] / df["amt_mean_prev"].replace(0, np.nan)
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["log_gap"] = np.log1p(df["gap_sec"])
    return df


def add_ref_features(df, ref):
    """Features derived from a labelled reference frame (card profile, target encodings)."""
    df = df.copy()
    home = ref.groupby("cc_num")[["lat", "long"]].median()
    home.columns = ["home_lat", "home_long"]
    df = df.merge(home, on="cc_num", how="left")
    df["home_lat"] = df["home_lat"].fillna(ref["lat"].median())
    df["home_long"] = df["home_long"].fillna(ref["long"].median())
    df["dist_home_km"] = haversine(df["lat"], df["long"], df["home_lat"], df["home_long"])
    prior = ref["is_fraud"].mean()
    for col, alpha in [("category", 30.0), ("merchant", 30.0)]:
        st = ref.groupby(col)["is_fraud"].agg(["sum", "count"])
        enc = (st["sum"] + prior * alpha) / (st["count"] + alpha)
        df[col + "_enc"] = df[col].map(enc).fillna(prior)
    cs = ref.groupby("cc_num")["is_fraud"].agg(["sum", "count"])
    cenc = (cs["sum"] + prior * 50.0) / (cs["count"] + 50.0)
    df["card_enc"] = df["cc_num"].map(cenc).fillna(prior)
    df["card_txn_count"] = df["cc_num"].map(cs["count"]).fillna(0)
    return df


FEATURES = [
    "amount", "log_amount", "hour", "dow", "is_night",
    "log_gap", "dist_prev_km", "speed_kmh",
    "cnt_1h", "cnt_24h", "cnt_7d",
    "amt_mean_prev", "amt_std_prev", "amt_z", "amt_ratio",
    "dist_home_km", "category_enc", "merchant_enc", "card_enc", "card_txn_count",
]

all_raw = pd.concat([train_raw, score_raw], ignore_index=True)
all_seq = build_seq_features(all_raw)
lab_seq = all_seq[all_seq["is_fraud"].notna()].copy()
score_seq = all_seq[all_seq["is_fraud"].isna()].copy()

# COMMAND ----------
# Held-out evaluation on a time-based split of the labelled data

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

cut = lab_seq["ts"].quantile(0.8)
fit_part = lab_seq[lab_seq["ts"] <= cut]
val_part = lab_seq[lab_seq["ts"] > cut]

fit_f = add_ref_features(fit_part, fit_part)
val_f = add_ref_features(val_part, fit_part)

clf_params = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
                  min_samples_leaf=20, l2_regularization=1.0, random_state=42)
clf = HistGradientBoostingClassifier(**clf_params)
clf.fit(fit_f[FEATURES], fit_f["is_fraud"].astype(int))
val_pred = clf.predict_proba(val_f[FEATURES])[:, 1]
auc = roc_auc_score(val_f["is_fraud"].astype(int), val_pred)
ap = average_precision_score(val_f["is_fraud"].astype(int), val_pred)
print(f"holdout ROC AUC = {auc:.4f}  AP = {ap:.4f}  (n_val={len(val_f)})")

# COMMAND ----------
# Final feature frames (reference = full labelled history)

lab_full = add_ref_features(lab_seq, lab_seq)
score_full = add_ref_features(score_seq, lab_seq)

BASE_COLS = ["transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "long"]
fg_pdf = pd.concat([lab_full, score_full], ignore_index=True)[BASE_COLS + FEATURES]
fg_pdf["cc_num"] = fg_pdf["cc_num"].astype("int64")

spark.createDataFrame(fg_pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FG_TABLE)
spark.sql(f"ALTER TABLE {FG_TABLE} ALTER COLUMN transaction_id SET NOT NULL")
spark.sql(f"ALTER TABLE {FG_TABLE} ADD CONSTRAINT cctxn9d5953_pk PRIMARY KEY(transaction_id)")
print("feature group written:", FG_TABLE)

td_pdf = lab_full[BASE_COLS + FEATURES].copy()
td_pdf["is_fraud"] = lab_full["is_fraud"].astype(int)
td_pdf["cc_num"] = td_pdf["cc_num"].astype("int64")
spark.createDataFrame(td_pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TD_TABLE)
print("training dataset written:", TD_TABLE)

# COMMAND ----------
# Train final model on all labelled data, register with metrics

import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/benedict@hopsworks.ai/mlpab59b6a9/ccfraud")

final_clf = HistGradientBoostingClassifier(**clf_params)
X = lab_full[FEATURES]
y = lab_full["is_fraud"].astype(int)
final_clf.fit(X, y)
train_auc = roc_auc_score(y, final_clf.predict_proba(X)[:, 1])

with mlflow.start_run(run_name="ccfraud_hgb") as run:
    mlflow.log_params(clf_params)
    mlflow.log_param("n_features", len(FEATURES))
    mlflow.log_metric("roc_auc_holdout", auc)
    mlflow.log_metric("average_precision_holdout", ap)
    mlflow.log_metric("roc_auc_train", train_auc)
    sig = infer_signature(X.head(5), final_clf.predict_proba(X.head(5))[:, 1])
    mlflow.sklearn.log_model(final_clf, "model", signature=sig,
                             input_example=X.head(5),
                             registered_model_name=MODEL_NAME)
print("model registered:", MODEL_NAME, "run:", run.info.run_id)

# COMMAND ----------
# Score and write predictions table (online-lookup ready: PK + CDF)

probs = final_clf.predict_proba(score_full[FEATURES])[:, 1]
pred_pdf = pd.DataFrame({
    "transaction_id": score_full["transaction_id"].values,
    "fraud_probability": np.clip(probs, 0.0, 1.0),
})
spark.createDataFrame(pred_pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PRED_TABLE)
spark.sql(f"ALTER TABLE {PRED_TABLE} ALTER COLUMN transaction_id SET NOT NULL")
spark.sql(f"ALTER TABLE {PRED_TABLE} ADD CONSTRAINT ccpred9d5953_pk PRIMARY KEY(transaction_id)")
spark.sql(f"ALTER TABLE {PRED_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("predictions written:", PRED_TABLE, len(pred_pdf))
print(f"RESULT holdout_roc_auc={auc:.4f}")
