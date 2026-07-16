from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
import base64, time

w = WorkspaceClient()
user = 'benedict@logicalclocks.com'
prefix = 'mlpab88e583'

nb_content = r"""# Databricks notebook source
# Pipeline v4 - Train model with proper MLflow UC volume tmpdir
import mlflow
import mlflow.spark
import os
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

SCHEMA = "workspace.mlpab88e583"
VOL_PATH = "/Volumes/workspace/mlpab88e583/data"
PREFIX = "mlpab88e583"
# Required for mlflow.spark in serverless/shared compute
MLFLOW_TMP = f"{VOL_PATH}/mlflow_tmp"
os.environ['MLFLOW_DFS_TMP'] = MLFLOW_TMP

print(f"Schema: {SCHEMA}")
print(f"MLflow tmp: {MLFLOW_TMP}")

FEATURE_COLS = [
    "hour_of_day", "day_of_week", "amount_log",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "amount_sum_1h", "amount_sum_24h", "amount_mean_7d", "amount_std_7d",
    "geo_dist", "time_since_last_txn", "amount_z_score",
    "category_idx"
]

# COMMAND ----------
# Load training dataset (already created)
train_data = spark.table(f"{SCHEMA}.cctd739ee9")
print(f"Training rows: {train_data.count()}")
train_data.show(3)

train_split, test_split = train_data.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_split.count()}, Test: {test_split.count()}")

# COMMAND ----------
# Train GBT classifier
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features", handleInvalid="keep")
gbt = GBTClassifier(
    featuresCol="features", labelCol="is_fraud",
    maxIter=100, maxDepth=6, stepSize=0.1,
    subsamplingRate=0.8, seed=42
)
ml_pipeline = Pipeline(stages=[assembler, gbt])

mlflow.set_registry_uri("databricks-uc")
exp_path = f"/Users/benedict@logicalclocks.com/{PREFIX}/ccfraud_exp"
mlflow.set_experiment(exp_path)

with mlflow.start_run(run_name="gbt_v1") as run:
    mlflow.log_params({"maxIter": 100, "maxDepth": 6, "stepSize": 0.1, "subsamplingRate": 0.8})

    model = ml_pipeline.fit(train_split)

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

    test_preds = model.transform(test_split)
    auc_roc = evaluator_auc.evaluate(test_preds)
    auc_pr = evaluator_pr.evaluate(test_preds)

    mlflow.log_metrics({"roc_auc": auc_roc, "pr_auc": auc_pr})

    # Log model with UC volume tmpdir
    mlflow.spark.log_model(
        model,
        artifact_path="model",
        dfs_tmpdir=MLFLOW_TMP,
        registered_model_name=f"{SCHEMA}.ccmodel739ee9"
    )
    run_id = run.info.run_id
    print(f"ROC AUC: {auc_roc:.4f}, PR AUC: {auc_pr:.4f}")
    print(f"Run ID: {run_id}")

print(f"Model registered: {SCHEMA}.ccmodel739ee9")

# COMMAND ----------
# Score transactions using the trained model
# Load the feature group table to get category encoding info
feat_df = spark.table(f"{SCHEMA}.cctxn739ee9")

# Rebuild StringIndexer from feature group data
cat_indexer = StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep")
cat_model = cat_indexer.fit(feat_df)

# Load scoring data
score_raw = spark.read.csv(f"{VOL_PATH}/score_transactions.csv", header=True, inferSchema=True)
score_raw = score_raw.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
score_raw = score_raw.withColumnRenamed("long", "lon")
score_raw = score_raw.withColumn("is_fraud", F.lit(0).cast("int"))
print(f"Score rows: {score_raw.count()}")

# Combine with training history for proper window features
hist_base = feat_df.select("transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "lon", F.lit(0).alias("is_score"), "is_fraud")
score_base = score_raw.select("transaction_id", "cc_num", "datetime", "amount", "merchant", "category", "lat", "lon", F.lit(1).alias("is_score"), "is_fraud")
combined = hist_base.unionByName(score_base)

# Compute window features on combined dataset
c_window = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long"))
c_window_1h = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long")).rangeBetween(-3600, 0)
c_window_24h = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long")).rangeBetween(-86400, 0)
c_window_7d = Window.partitionBy("cc_num").orderBy(F.col("datetime").cast("long")).rangeBetween(-604800, 0)

combined_feat = combined \
    .withColumn("hour_of_day", F.hour("datetime").cast("double")) \
    .withColumn("day_of_week", F.dayofweek("datetime").cast("double")) \
    .withColumn("amount_log", F.log1p(F.col("amount"))) \
    .withColumn("txn_count_1h", F.count("transaction_id").over(c_window_1h).cast("double")) \
    .withColumn("txn_count_24h", F.count("transaction_id").over(c_window_24h).cast("double")) \
    .withColumn("txn_count_7d", F.count("transaction_id").over(c_window_7d).cast("double")) \
    .withColumn("amount_sum_1h", F.sum("amount").over(c_window_1h)) \
    .withColumn("amount_sum_24h", F.sum("amount").over(c_window_24h)) \
    .withColumn("amount_mean_7d", F.avg("amount").over(c_window_7d)) \
    .withColumn("amount_std_7d", F.coalesce(F.stddev("amount").over(c_window_7d), F.lit(0.0))) \
    .withColumn("prev_lat", F.lag("lat", 1).over(c_window)) \
    .withColumn("prev_lon", F.lag("lon", 1).over(c_window)) \
    .withColumn("geo_dist", F.when(
        F.col("prev_lat").isNotNull(),
        F.sqrt(F.pow(F.col("lat") - F.col("prev_lat"), 2) +
               F.pow(F.col("lon") - F.col("prev_lon"), 2)) * 111.0
    ).otherwise(0.0)) \
    .withColumn("prev_datetime", F.lag("datetime", 1).over(c_window)) \
    .withColumn("time_since_last_txn", F.when(
        F.col("prev_datetime").isNotNull(),
        (F.col("datetime").cast("long") - F.col("prev_datetime").cast("long")) / 3600.0
    ).otherwise(24.0)) \
    .withColumn("amount_z_score", F.when(
        F.col("amount_std_7d") > 0,
        (F.col("amount") - F.col("amount_mean_7d")) / F.col("amount_std_7d")
    ).otherwise(0.0))

combined_feat = cat_model.transform(combined_feat)
score_feat = combined_feat.filter(F.col("is_score") == 1).select("transaction_id", *FEATURE_COLS).fillna(0.0)
print(f"Score features count: {score_feat.count()}")

# COMMAND ----------
# Generate predictions
score_preds = model.transform(score_feat)

@F.udf(returnType=DoubleType())
def extract_prob(v):
    return float(v[1])

pred_df = score_preds.select(
    "transaction_id",
    extract_prob("probability").alias("fraud_probability")
)
cnt = pred_df.count()
print(f"Predictions: {cnt}")
pred_df.show(10)
pred_df.describe("fraud_probability").show()

# COMMAND ----------
# Write predictions feature table with CDF enabled
pred_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.ccpred739ee9")
spark.sql(f"ALTER TABLE {SCHEMA}.ccpred739ee9 SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
print(f"Predictions written: {SCHEMA}.ccpred739ee9")
spark.sql(f"SELECT COUNT(*), MIN(fraud_probability), MAX(fraud_probability), AVG(fraud_probability) FROM {SCHEMA}.ccpred739ee9").show()
spark.sql(f"SELECT * FROM {SCHEMA}.ccpred739ee9 LIMIT 5").show()

# COMMAND ----------
print("="*60)
print("PIPELINE COMPLETE")
print(f"Feature Group:     {SCHEMA}.cctxn739ee9")
print(f"Training Dataset:  {SCHEMA}.cctd739ee9")
print(f"Registered Model:  {SCHEMA}.ccmodel739ee9")
print(f"Predictions Table: {SCHEMA}.ccpred739ee9")
print(f"ROC AUC (test):    {auc_roc:.4f}")
print("="*60)
dbutils.notebook.exit(f"SUCCESS|roc_auc={auc_roc:.4f}|pr_auc={auc_pr:.4f}|preds={cnt}")
"""

nb_path = f'/Users/{user}/{prefix}/ccfraud_pipeline_v4'
encoded = base64.b64encode(nb_content.encode()).decode()
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded,
    overwrite=True
)
print(f'Notebook uploaded: {nb_path}')

from databricks.sdk.service.jobs import NotebookTask, Task, Source
task = Task(
    task_key='pipeline_v4',
    notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE),
    timeout_seconds=3000
)
job = w.jobs.create(name=f'{prefix}_ccfraud_v4', tasks=[task])
run = w.jobs.run_now(job_id=job.job_id)
print(f'Job: {job.job_id}, Run: {run.run_id}')

start = time.time()
while True:
    r = w.jobs.get_run(run_id=run.run_id)
    lc = r.state.life_cycle_state.value if r.state.life_cycle_state else 'UNKNOWN'
    rs = r.state.result_state.value if r.state.result_state else ''
    elapsed = int(time.time() - start)
    print(f'[{elapsed}s] {lc} {rs}')
    if lc in ('TERMINATED', 'SKIPPED', 'INTERNAL_ERROR'):
        for t in (r.tasks or []):
            if t.run_id:
                out = w.jobs.get_run_output(run_id=t.run_id)
                if out.notebook_output:
                    print('Output:', out.notebook_output.result)
                if out.error:
                    print('Error:', out.error[:3000])
                if out.error_trace:
                    print('Trace:', out.error_trace[:3000])
        print(f'FINAL: {lc} {rs}')
        break
    time.sleep(30)
