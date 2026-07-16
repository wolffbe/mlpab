# Databricks notebook source
# MAGIC %md
# MAGIC # Credit-card fraud FTI pipeline

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering mlflow scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "workspace"
SCHEMA = "mlpab08bf79"
FQ = f"{CATALOG}.{SCHEMA}"
DATA = "/Volumes/workspace/mlpab08bf79/data"

FG = f"{FQ}.cctxn2dbe0a"      # feature group
TD = f"{FQ}.cctd2dbe0a"       # training dataset
PRED = f"{FQ}.ccpred2dbe0a"   # predictions feature table
MODEL = f"{FQ}.ccmodel2dbe0a" # registered model

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.sql import types as T

schema_in = T.StructType([
    T.StructField("transaction_id", T.StringType()),
    T.StructField("cc_num", T.StringType()),
    T.StructField("datetime", T.StringType()),
    T.StructField("amount", T.DoubleType()),
    T.StructField("merchant", T.StringType()),
    T.StructField("category", T.StringType()),
    T.StructField("lat", T.DoubleType()),
    T.StructField("long", T.DoubleType()),
    T.StructField("is_fraud", T.IntegerType()),
])
schema_score = T.StructType(schema_in.fields[:-1])

train_raw = spark.read.csv(f"{DATA}/transactions.csv", header=True, schema=schema_in)
score_raw = spark.read.csv(f"{DATA}/score_transactions.csv", header=True, schema=schema_score)

print("train", train_raw.count(), "score", score_raw.count())

# COMMAND ----------

# Per-card home location computed from labelled history (usual location).
home = (train_raw.groupBy("cc_num")
        .agg(F.expr("percentile_approx(lat,0.5)").alias("home_lat"),
             F.expr("percentile_approx(long,0.5)").alias("home_long"),
             F.avg("amount").alias("card_avg_amt"),
             F.stddev("amount").alias("card_std_amt")))

def haversine(lat1, lon1, lat2, lon2):
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = F.sin(dlat / 2) ** 2 + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dlon / 2) ** 2
    return 6371.0 * 2 * F.asin(F.sqrt(a))

def engineer(df):
    df = df.withColumn("ts", F.to_timestamp("datetime"))
    df = df.withColumn("hour", F.hour("ts"))
    df = df.withColumn("dow", F.dayofweek("ts"))
    df = df.withColumn("log_amount", F.log1p(F.col("amount")))
    df = df.join(home, "cc_num", "left")
    df = df.withColumn("dist_from_home_km", haversine(F.col("lat"), F.col("long"), F.col("home_lat"), F.col("home_long")))
    df = df.withColumn("amt_vs_card_avg", (F.col("amount") - F.col("card_avg_amt")) / (F.col("card_std_amt") + F.lit(1.0)))

    w = Window.partitionBy("cc_num").orderBy("ts")
    df = df.withColumn("prev_ts", F.lag("ts").over(w))
    df = df.withColumn("prev_lat", F.lag("lat").over(w))
    df = df.withColumn("prev_long", F.lag("long").over(w))
    df = df.withColumn("secs_since_prev",
                       F.when(F.col("prev_ts").isNotNull(),
                              F.col("ts").cast("long") - F.col("prev_ts").cast("long")).otherwise(F.lit(999999)))
    df = df.withColumn("dist_from_prev_km",
                       F.when(F.col("prev_lat").isNotNull(),
                              haversine(F.col("lat"), F.col("long"), F.col("prev_lat"), F.col("prev_long"))).otherwise(F.lit(0.0)))
    df = df.withColumn("speed_kmh", F.col("dist_from_prev_km") / ((F.col("secs_since_prev") / 3600.0) + F.lit(0.01)))

    wt1 = Window.partitionBy("cc_num").orderBy(F.col("ts").cast("long")).rangeBetween(-3600, 0)
    wt24 = Window.partitionBy("cc_num").orderBy(F.col("ts").cast("long")).rangeBetween(-86400, 0)
    df = df.withColumn("cnt_1h", F.count(F.lit(1)).over(wt1))
    df = df.withColumn("cnt_24h", F.count(F.lit(1)).over(wt24))
    df = df.withColumn("sum_amt_24h", F.sum("amount").over(wt24))

    for c in ["dist_from_home_km", "amt_vs_card_avg", "card_avg_amt", "card_std_amt"]:
        df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))
    return df

FEATURES = ["amount", "log_amount", "hour", "dow", "dist_from_home_km", "amt_vs_card_avg",
            "secs_since_prev", "dist_from_prev_km", "speed_kmh", "cnt_1h", "cnt_24h", "sum_amt_24h"]
CAT = "category"

train_fe = engineer(train_raw)
score_fe = engineer(score_raw)

# COMMAND ----------

# Build feature group: transaction_id + engineered features + label
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

fg_cols = ["transaction_id", "cc_num", CAT] + FEATURES + ["is_fraud"]
fg_df = train_fe.select(*fg_cols)

spark.sql(f"DROP TABLE IF EXISTS {FG}")
fe.create_table(name=FG, primary_keys=["transaction_id"], df=fg_df,
                description="Engineered credit-card fraud features (training history)")
print("feature group created:", FG)

# COMMAND ----------

# Training dataset assembled from the feature group
spark.sql(f"DROP TABLE IF EXISTS {TD}")
spark.table(FG).write.mode("overwrite").saveAsTable(TD)
print("training dataset created:", TD)

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/benedict@logicalclocks.com/mlpab08bf79/ccfraud")

pdf = spark.table(TD).select(*(["transaction_id", CAT] + FEATURES + ["is_fraud"])).toPandas()
X = pdf[[CAT] + FEATURES].copy()
y = pdf["is_fraud"].astype(int).values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), [CAT])],
    remainder="passthrough")
clf = Pipeline([("pre", pre),
                ("gb", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                                      max_depth=6, l2_regularization=1.0,
                                                      random_state=42, class_weight="balanced"))])

with mlflow.start_run(run_name="ccfraud_hgb") as run:
    clf.fit(Xtr, ytr)
    proba_te = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba_te)
    ap = average_precision_score(yte, proba_te)
    mlflow.log_param("model", "HistGradientBoostingClassifier")
    mlflow.log_param("n_features", len(FEATURES) + 1)
    mlflow.log_metric("holdout_roc_auc", float(auc))
    mlflow.log_metric("holdout_avg_precision", float(ap))
    sig = mlflow.models.infer_signature(Xtr, clf.predict_proba(Xtr)[:, 1])
    mlflow.sklearn.log_model(clf, artifact_path="model", signature=sig,
                             registered_model_name=MODEL,
                             input_example=Xtr.head(3))
    run_id = run.info.run_id
    print(f"holdout ROC AUC = {auc:.4f}  AP = {ap:.4f}  run_id={run_id}")

# COMMAND ----------

# Score all rows of score_transactions and write the predictions feature table
score_pdf = score_fe.select(*(["transaction_id", CAT] + FEATURES)).toPandas()
Xs = score_pdf[[CAT] + FEATURES].copy()
score_pdf["fraud_probability"] = clf.predict_proba(Xs)[:, 1].astype(float)
score_pdf["fraud_probability"] = score_pdf["fraud_probability"].clip(0.0, 1.0)

pred_df = spark.createDataFrame(score_pdf[["transaction_id", "fraud_probability"]])

spark.sql(f"DROP TABLE IF EXISTS {PRED}")
fe.create_table(name=PRED, primary_keys=["transaction_id"], df=pred_df,
                description="Per-transaction fraud probability predictions")
# enable CDF so the table can be synced to an online store
spark.sql(f"ALTER TABLE {PRED} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("predictions table created:", PRED, "rows:", pred_df.count())

# COMMAND ----------

# Make predictions available for low-latency lookup (online table)
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

w = WorkspaceClient()
online_name = f"{PRED}_online"
try:
    w.online_tables.delete(name=online_name)
except Exception as e:
    print("no existing online table:", e)

spec = OnlineTableSpec(
    source_table_full_name=PRED,
    primary_key_columns=["transaction_id"],
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=True,
)
try:
    ot = w.online_tables.create_and_wait(name=online_name, spec=spec)
    print("online table created:", online_name, ot.status)
except Exception as e:
    print("online table create issue:", repr(e))

# COMMAND ----------

print("DONE")
print("FG=", FG)
print("TD=", TD)
print("PRED=", PRED)
print("MODEL=", MODEL)
