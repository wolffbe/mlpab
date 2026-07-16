# Databricks notebook source
# MAGIC %md # CC Fraud Detection — Full FTI Pipeline

# COMMAND ----------
import os
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.getOrCreate()

SCHEMA = "workspace.mlpab88e583"
VOL_PATH = "/Volumes/workspace/mlpab88e583/data"
PREFIX = "mlpab88e583"

print(f"Schema: {SCHEMA}")
print(f"Volume: {VOL_PATH}")

# COMMAND ----------
# MAGIC %md ## 1. Load raw transactions

txn_df = spark.read.csv(f"{VOL_PATH}/transactions.csv", header=True, inferSchema=True)
txn_df = txn_df.withColumn("datetime", F.to_timestamp("datetime"))
txn_df.printSchema()
print(f"Rows: {txn_df.count()}")
txn_df.show(5)

# COMMAND ----------
# MAGIC %md ## 2. Feature Engineering

# Card-level window for velocity/stats
card_window = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long"))
card_window_1h = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long")).rangeBetween(-3600, 0)
card_window_24h = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long")).rangeBetween(-86400, 0)
card_window_7d = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long")).rangeBetween(-604800, 0)

feat_df = txn_df \
    .withColumn("hour_of_day", F.hour("datetime")) \
    .withColumn("day_of_week", F.dayofweek("datetime")) \
    .withColumn("amount_log", F.log1p("amount")) \
    .withColumn("txn_count_1h", F.count("transaction_id").over(card_window_1h)) \
    .withColumn("txn_count_24h", F.count("transaction_id").over(card_window_24h)) \
    .withColumn("txn_count_7d", F.count("transaction_id").over(card_window_7d)) \
    .withColumn("amount_sum_1h", F.sum("amount").over(card_window_1h)) \
    .withColumn("amount_sum_24h", F.sum("amount").over(card_window_24h)) \
    .withColumn("amount_mean_7d", F.avg("amount").over(card_window_7d)) \
    .withColumn("amount_std_7d", F.stddev("amount").over(card_window_7d)) \
    .withColumn("prev_lat", F.lag("lat", 1).over(card_window)) \
    .withColumn("prev_long", F.lag("long", 1).over(card_window)) \
    .withColumn("geo_dist", F.when(
        F.col("prev_lat").isNotNull(),
        F.sqrt(
            F.pow(F.col("lat") - F.col("prev_lat"), 2) +
            F.pow(F.col("long") - F.col("prev_long"), 2)
        ) * 111.0
    ).otherwise(0.0)) \
    .withColumn("prev_datetime", F.lag("datetime", 1).over(card_window)) \
    .withColumn("time_since_last_txn", F.when(
        F.col("prev_datetime").isNotNull(),
        (F.col("datetime").cast("long") - F.col("prev_datetime").cast("long")) / 3600.0
    ).otherwise(24.0)) \
    .withColumn("amount_std_7d", F.when(F.col("amount_std_7d").isNull(), 0.0).otherwise(F.col("amount_std_7d"))) \
    .withColumn("amount_z_score", F.when(
        F.col("amount_std_7d") > 0,
        (F.col("amount") - F.col("amount_mean_7d")) / F.col("amount_std_7d")
    ).otherwise(0.0))

# Category encoding
cat_indexer = StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep")
cat_model = cat_indexer.fit(feat_df)
feat_df = cat_model.transform(feat_df)

feat_df = feat_df.select(
    "transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "long",
    "is_fraud",
    "hour_of_day", "day_of_week", "amount_log",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_1h", "amount_sum_24h", "amount_mean_7d", "amount_std_7d",
    "geo_dist", "time_since_last_txn", "amount_z_score",
    "category_idx"
)

feat_df.printSchema()
feat_df.show(5)

# COMMAND ----------
# MAGIC %md ## 3. Write Feature Group cctxn739ee9

feat_table = f"{SCHEMA}.cctxn739ee9"
feat_df.write.format("delta").mode("overwrite").saveAsTable(feat_table)
print(f"Feature group written to: {feat_table}")
spark.sql(f"SELECT COUNT(*) FROM {feat_table}").show()

# COMMAND ----------
# MAGIC %md ## 4. Assemble Training Dataset cctd739ee9

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "amount_log",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_1h", "amount_sum_24h", "amount_mean_7d", "amount_std_7d",
    "geo_dist", "time_since_last_txn", "amount_z_score",
    "category_idx"
]

train_df = spark.table(feat_table).select(["transaction_id"] + FEATURE_COLS + ["is_fraud"])
train_df = train_df.fillna(0.0)

train_table = f"{SCHEMA}.cctd739ee9"
train_df.write.format("delta").mode("overwrite").saveAsTable(train_table)
print(f"Training dataset written to: {train_table}")
spark.sql(f"SELECT COUNT(*), SUM(is_fraud) as fraud_count FROM {train_table}").show()

# COMMAND ----------
# MAGIC %md ## 5. Train Fraud Classifier

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="is_fraud",
    maxIter=100,
    maxDepth=6,
    stepSize=0.1,
    subsamplingRate=0.8,
    seed=42
)

ml_pipeline = Pipeline(stages=[assembler, gbt])

# Train/test split
train_data = spark.table(train_table)
train_split, test_split = train_data.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_split.count()}, Test: {test_split.count()}")

mlflow.set_experiment(f"/Users/benedict@logicalclocks.com/{PREFIX}/ccfraud_experiment")

with mlflow.start_run(run_name="ccmodel739ee9_gbt") as run:
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("maxDepth", 6)
    mlflow.log_param("stepSize", 0.1)
    mlflow.log_param("subsamplingRate", 0.8)

    model = ml_pipeline.fit(train_split)

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="is_fraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="is_fraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    test_preds = model.transform(test_split)
    auc_roc = evaluator_auc.evaluate(test_preds)
    auc_pr = evaluator_pr.evaluate(test_preds)

    mlflow.log_metric("roc_auc", auc_roc)
    mlflow.log_metric("pr_auc", auc_pr)

    print(f"ROC AUC: {auc_roc:.4f}")
    print(f"PR AUC: {auc_pr:.4f}")

    mlflow.spark.log_model(model, artifact_path="model")
    run_id = run.info.run_id

print(f"MLflow run_id: {run_id}")

# COMMAND ----------
# MAGIC %md ## 6. Register Model ccmodel739ee9

import mlflow.tracking
client = mlflow.tracking.MlflowClient()
model_uri = f"runs:/{run_id}/model"

# Register model in Unity Catalog
full_model_name = f"{SCHEMA}.ccmodel739ee9"
try:
    client.create_registered_model(full_model_name)
    print(f"Created model: {full_model_name}")
except Exception:
    print(f"Model already exists: {full_model_name}")

mv = client.create_model_version(
    name=full_model_name,
    source=model_uri,
    run_id=run_id
)
print(f"Model version: {mv.version}")

# Alias as champion
client.set_registered_model_alias(full_model_name, "champion", mv.version)
print(f"Set alias 'champion' -> version {mv.version}")

# COMMAND ----------
# MAGIC %md ## 7. Score score_transactions.csv

score_raw = spark.read.csv(f"{VOL_PATH}/score_transactions.csv", header=True, inferSchema=True)
score_raw = score_raw.withColumn("datetime", F.to_timestamp("datetime"))
score_raw = score_raw.withColumn("is_fraud", F.lit(0))
print(f"Score rows: {score_raw.count()}")
score_raw.show(5)

# Need historical context for window features — union with train data, mark scoring rows
hist_df = spark.table(feat_table).select(
    "transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "long",
    F.lit(0).alias("is_score")
)
score_base = score_raw.select(
    "transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "long",
    F.lit(1).alias("is_score")
)

combined = hist_df.unionByName(score_base)

# Recompute features on combined dataset
card_window_c = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long"))
card_window_1h_c = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long")).rangeBetween(-3600, 0)
card_window_24h_c = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long")).rangeBetween(-86400, 0)
card_window_7d_c = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long")).rangeBetween(-604800, 0)

combined_feat = combined \
    .withColumn("hour_of_day", F.hour("datetime")) \
    .withColumn("day_of_week", F.dayofweek("datetime")) \
    .withColumn("amount_log", F.log1p("amount")) \
    .withColumn("txn_count_1h", F.count("transaction_id").over(card_window_1h_c)) \
    .withColumn("txn_count_24h", F.count("transaction_id").over(card_window_24h_c)) \
    .withColumn("txn_count_7d", F.count("transaction_id").over(card_window_7d_c)) \
    .withColumn("amount_sum_1h", F.sum("amount").over(card_window_1h_c)) \
    .withColumn("amount_sum_24h", F.sum("amount").over(card_window_24h_c)) \
    .withColumn("amount_mean_7d", F.avg("amount").over(card_window_7d_c)) \
    .withColumn("amount_std_7d", F.stddev("amount").over(card_window_7d_c)) \
    .withColumn("prev_lat", F.lag("lat", 1).over(card_window_c)) \
    .withColumn("prev_long", F.lag("long", 1).over(card_window_c)) \
    .withColumn("geo_dist", F.when(
        F.col("prev_lat").isNotNull(),
        F.sqrt(
            F.pow(F.col("lat") - F.col("prev_lat"), 2) +
            F.pow(F.col("long") - F.col("prev_long"), 2)
        ) * 111.0
    ).otherwise(0.0)) \
    .withColumn("prev_datetime", F.lag("datetime", 1).over(card_window_c)) \
    .withColumn("time_since_last_txn", F.when(
        F.col("prev_datetime").isNotNull(),
        (F.col("datetime").cast("long") - F.col("prev_datetime").cast("long")) / 3600.0
    ).otherwise(24.0)) \
    .withColumn("amount_std_7d", F.when(F.col("amount_std_7d").isNull(), 0.0).otherwise(F.col("amount_std_7d"))) \
    .withColumn("amount_z_score", F.when(
        F.col("amount_std_7d") > 0,
        (F.col("amount") - F.col("amount_mean_7d")) / F.col("amount_std_7d")
    ).otherwise(0.0))

cat_model_c = cat_model
combined_feat = cat_model_c.transform(combined_feat)

# Filter to scoring rows only
score_feat = combined_feat.filter(F.col("is_score") == 1).select(
    "transaction_id",
    *FEATURE_COLS
).fillna(0.0)

print(f"Score features: {score_feat.count()}")
score_feat.show(5)

# COMMAND ----------
# MAGIC %md ## 8. Generate Predictions

score_preds = model.transform(score_feat)

# Extract probability of fraud (class 1)
extract_prob = F.udf(lambda v: float(v[1]), FloatType())
pred_df = score_preds.select(
    "transaction_id",
    extract_prob("probability").alias("fraud_probability")
)

pred_df.show(20)
print(f"Predictions count: {pred_df.count()}")

# COMMAND ----------
# MAGIC %md ## 9. Write Predictions Feature Table ccpred739ee9

pred_table = f"{SCHEMA}.ccpred739ee9"
pred_df.write.format("delta").mode("overwrite") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(pred_table)

spark.sql(f"ALTER TABLE {pred_table} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

print(f"Predictions written to: {pred_table}")
spark.sql(f"SELECT COUNT(*), MIN(fraud_probability), MAX(fraud_probability), AVG(fraud_probability) FROM {pred_table}").show()
spark.sql(f"SELECT * FROM {pred_table} LIMIT 5").show()

# COMMAND ----------
# MAGIC %md ## Done — Summary

print("="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"Feature Group:     {feat_table}")
print(f"Training Dataset:  {train_table}")
print(f"Registered Model:  {full_model_name}")
print(f"Predictions Table: {pred_table}")
print(f"ROC AUC (test):    {auc_roc:.4f}")
