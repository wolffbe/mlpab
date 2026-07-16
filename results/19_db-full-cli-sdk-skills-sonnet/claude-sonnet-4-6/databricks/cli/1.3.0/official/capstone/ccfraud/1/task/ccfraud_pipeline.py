# Databricks notebook source
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import requests
import os

spark = SparkSession.builder.getOrCreate()

CATALOG = "workspace"
SCHEMA = "mlpab4d3871"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/mlpab4d3871_data"

FG_NAME = "cctxn739ee9"
TD_NAME = "cctd739ee9"
MODEL_NAME = "ccmodel739ee9"
PRED_TABLE = "ccpred739ee9"
EXPERIMENT_PATH = f"/Users/benedict@logicalclocks.com/mlpab4d3871/ccfraud_exp"

print(f"Starting FTI pipeline: {CATALOG}.{SCHEMA}")

# COMMAND ----------
# Load raw data
df_train_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{VOLUME_PATH}/transactions.csv")
df_score_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{VOLUME_PATH}/score_transactions.csv")

print(f"Training rows: {df_train_raw.count()}, Score rows: {df_score_raw.count()}")

# COMMAND ----------
# Tag each source, then union for consistent feature engineering
df_train_tagged = df_train_raw.withColumn("_split", F.lit("train"))
df_score_tagged = df_score_raw.withColumn("is_fraud", F.lit(None).cast("int")).withColumn("_split", F.lit("score"))

df_all = df_train_tagged.unionByName(df_score_tagged)

# COMMAND ----------
# Feature engineering on combined dataset
df = df_all.withColumn("ts", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
df = df.withColumn("hour_of_day", F.hour("ts"))
df = df.withColumn("day_of_week", F.dayofweek("ts"))
df = df.withColumn("unix_ts", F.unix_timestamp("ts"))
df = df.withColumn("amount_log", F.log1p(F.col("amount")))
df = df.withColumn("is_high_amount", (F.col("amount") > 500).cast("int"))
df = df.withColumn("is_night", ((F.col("hour_of_day") >= 23) | (F.col("hour_of_day") <= 6)).cast("int"))
df = df.withColumn("is_weekend", ((F.col("day_of_week") == 1) | (F.col("day_of_week") == 7)).cast("int"))
df = df.withColumn("category_idx", (F.hash("category") % 20).cast("int"))

# Rolling window features by card
w1h = Window.partitionBy("cc_num").orderBy("unix_ts").rangeBetween(-3600, 0)
w24h = Window.partitionBy("cc_num").orderBy("unix_ts").rangeBetween(-86400, 0)
w7d = Window.partitionBy("cc_num").orderBy("unix_ts").rangeBetween(-604800, 0)

df = df.withColumn("txn_count_1h", F.count("transaction_id").over(w1h))
df = df.withColumn("txn_count_24h", F.count("transaction_id").over(w24h))
df = df.withColumn("txn_count_7d", F.count("transaction_id").over(w7d))
df = df.withColumn("amount_sum_1h", F.sum("amount").over(w1h))
df = df.withColumn("amount_sum_24h", F.sum("amount").over(w24h))
df = df.withColumn("avg_amount_7d", F.avg("amount").over(w7d))
df = df.withColumn("amount_ratio_to_avg", F.col("amount") / (F.col("avg_amount_7d") + 0.01))
df = df.withColumn("avg_lat_7d", F.avg("lat").over(w7d))
df = df.withColumn("avg_long_7d", F.avg("long").over(w7d))
df = df.withColumn("geo_dist",
    F.sqrt(F.pow(F.col("lat") - F.col("avg_lat_7d"), 2) + F.pow(F.col("long") - F.col("avg_long_7d"), 2))
)

print("Feature engineering done")

# COMMAND ----------
# Split back into train and score
FEATURE_COLS = [
    "transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "long",
    "hour_of_day", "day_of_week", "amount_log", "is_high_amount",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_1h", "amount_sum_24h",
    "avg_amount_7d", "amount_ratio_to_avg",
    "avg_lat_7d", "avg_long_7d", "geo_dist",
    "category_idx", "is_night", "is_weekend",
    "is_fraud"
]

df_train_feat = df.filter(F.col("_split") == "train").select(FEATURE_COLS)
df_score_feat = df.filter(F.col("_split") == "score").select(FEATURE_COLS)

print(f"Train features: {df_train_feat.count()}, Score features: {df_score_feat.count()}")

# COMMAND ----------
# Write feature group table
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.{FG_NAME}")
df_train_feat.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.{FG_NAME}")
print(f"Feature group written: {CATALOG}.{SCHEMA}.{FG_NAME}")

# COMMAND ----------
# Create training dataset
TD_COLS = [
    "transaction_id", "hour_of_day", "day_of_week", "amount_log", "is_high_amount",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_1h", "amount_sum_24h", "avg_amount_7d", "amount_ratio_to_avg",
    "geo_dist", "category_idx", "is_night", "is_weekend", "is_fraud"
]

df_td = df_train_feat.select(TD_COLS).filter(F.col("is_fraud").isNotNull())
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.{TD_NAME}")
df_td.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.{TD_NAME}")
print(f"Training dataset written: {CATALOG}.{SCHEMA}.{TD_NAME}")

# COMMAND ----------
# Train model
FEATURE_NAMES = [
    "hour_of_day", "day_of_week", "amount_log", "is_high_amount",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_1h", "amount_sum_24h", "avg_amount_7d", "amount_ratio_to_avg",
    "geo_dist", "category_idx", "is_night", "is_weekend"
]

df_pd = df_td.toPandas()
X = df_pd[FEATURE_NAMES].fillna(0)
y = df_pd["is_fraud"].fillna(0)
print(f"Training shape: {X.shape}, Fraud rate: {y.mean():.4f}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

with mlflow.start_run(run_name="ccfraud_gbm") as run:
    model = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_prob)
    avg_precision = average_precision_score(y_val, y_prob)

    print(f"Val ROC AUC: {roc_auc:.4f}, Avg Precision: {avg_precision:.4f}")

    mlflow.log_params({
        "n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
        "min_samples_leaf": 20, "subsample": 0.8, "model_type": "GradientBoostingClassifier"
    })
    mlflow.log_metrics({
        "val_roc_auc": roc_auc, "val_avg_precision": avg_precision,
        "train_size": len(X_train), "val_size": len(X_val), "fraud_rate": float(y.mean())
    })
    signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
    mlflow.sklearn.log_model(model, "model", signature=signature)
    run_id = run.info.run_id

print(f"MLflow run: {run_id}")

# COMMAND ----------
# Register model
full_model_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
client = MlflowClient()

try:
    client.create_registered_model(full_model_name)
except Exception as e:
    print(f"Model registration note: {e}")

mv = mlflow.register_model(f"runs:/{run_id}/model", full_model_name)
client.set_model_version_tag(full_model_name, mv.version, "val_roc_auc", str(round(roc_auc, 4)))
client.set_model_version_tag(full_model_name, mv.version, "val_avg_precision", str(round(avg_precision, 4)))
print(f"Model registered: {full_model_name} v{mv.version}")

# COMMAND ----------
# Score transactions
df_score_pd = df_score_feat.select(["transaction_id"] + FEATURE_NAMES).toPandas()
X_score = df_score_pd[FEATURE_NAMES].fillna(0)
fraud_probs = model.predict_proba(X_score)[:, 1]

df_preds_pd = pd.DataFrame({
    "transaction_id": df_score_pd["transaction_id"].values,
    "fraud_probability": fraud_probs
})
print(f"Scored {len(df_preds_pd)} transactions. Prob range: {fraud_probs.min():.4f} - {fraud_probs.max():.4f}")

# COMMAND ----------
# Write predictions table
df_preds_spark = spark.createDataFrame(df_preds_pd)
spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{SCHEMA}.{PRED_TABLE}")
df_preds_spark.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.{PRED_TABLE}")

try:
    spark.sql(f"ALTER TABLE {CATALOG}.{SCHEMA}.{PRED_TABLE} ADD CONSTRAINT {PRED_TABLE}_pk PRIMARY KEY (transaction_id)")
except Exception as e:
    print(f"PK constraint note: {e}")

print(f"Predictions written: {CATALOG}.{SCHEMA}.{PRED_TABLE}")
spark.sql(f"SELECT COUNT(*), AVG(fraud_probability) FROM {CATALOG}.{SCHEMA}.{PRED_TABLE}").show()

# COMMAND ----------
# Create online table for low-latency lookup
try:
    # Get host from Spark conf (available in serverless)
    host = spark.conf.get("spark.databricks.workspaceUrl", "")
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    if not host:
        host = os.environ.get("DATABRICKS_HOST", "")
    if host and not host.startswith("https://"):
        host = f"https://{host}"

    if host and token:
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        # Delete existing if any
        requests.delete(f"{host}/api/2.0/online-tables/{CATALOG}.{SCHEMA}.{PRED_TABLE}_online", headers=headers)
        online_table_spec = {
            "name": f"{CATALOG}.{SCHEMA}.{PRED_TABLE}_online",
            "spec": {
                "source_table_full_name": f"{CATALOG}.{SCHEMA}.{PRED_TABLE}",
                "primary_key_columns": ["transaction_id"],
                "run_triggered": {}
            }
        }
        resp = requests.post(f"{host}/api/2.0/online-tables", headers=headers, json=online_table_spec)
        print(f"Online table response: {resp.status_code} - {resp.text[:300]}")
    else:
        print(f"Skipping online table: host={host!r}")
except Exception as e:
    print(f"Online table creation skipped: {e}")

# COMMAND ----------
print("=" * 60)
print("PIPELINE COMPLETE")
print(f"Feature Group:    {CATALOG}.{SCHEMA}.{FG_NAME}")
print(f"Training Dataset: {CATALOG}.{SCHEMA}.{TD_NAME}")
print(f"Model:            {full_model_name} v{mv.version}")
print(f"Predictions:      {CATALOG}.{SCHEMA}.{PRED_TABLE}")
print(f"Val ROC AUC:      {roc_auc:.4f}")
print("=" * 60)
