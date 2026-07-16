"""
Full FTI pipeline for credit card fraud detection on Databricks.
Runs via Databricks SDK: uploads data, creates notebook, runs job, creates online table.
"""
import os
import base64
import json
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs as jobs_svc, compute, workspace
from databricks.sdk.service.catalog import VolumeType

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]       # workspace.mlpabdbf4d5
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]       # mlpabdbf4d5
HOST   = os.environ["DATABRICKS_HOST"]
TOKEN  = os.environ["DATABRICKS_TOKEN"]

CATALOG, SCHEMA_NAME = SCHEMA.split(".")
USER_EMAIL = "benedict@logicalclocks.com"
NOTEBOOK_PATH = f"/Users/{USER_EMAIL}/{PREFIX}/ccfraud_pipeline"
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/{PREFIX}/ccfraud_experiment"
VOLUME_NAME = "data_upload"

FG_NAME   = "cctxn739ee9"
TD_NAME   = "cctd739ee9"
MDL_NAME  = "ccmodel739ee9"
PRED_NAME = "ccpred739ee9"

w = WorkspaceClient()

# ── 1. Ensure volume exists for data upload ──────────────────────────────────
print(f"[1] Creating volume {CATALOG}.{SCHEMA_NAME}.{VOLUME_NAME} ...")
try:
    w.volumes.create(
        catalog_name=CATALOG,
        schema_name=SCHEMA_NAME,
        name=VOLUME_NAME,
        volume_type=VolumeType.MANAGED
    )
    print("    Volume created.")
except Exception as e:
    if "already exists" in str(e).lower():
        print("    Volume already exists, continuing.")
    else:
        raise

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}"

# ── 2. Upload CSV files ───────────────────────────────────────────────────────
print("[2] Uploading CSV files ...")
for fname in ["transactions.csv", "score_transactions.csv"]:
    local_path = f"data/{fname}"
    remote_path = f"{VOLUME_PATH}/{fname}"
    with open(local_path, "rb") as f:
        w.files.upload(remote_path, f, overwrite=True)
    print(f"    Uploaded {fname} -> {remote_path}")

# ── 3. Create notebook ────────────────────────────────────────────────────────
print("[3] Creating pipeline notebook ...")

NOTEBOOK_CONTENT = f'''# Databricks notebook source
# MAGIC %md # Credit Card Fraud Detection Pipeline

# COMMAND ----------
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
import math

CATALOG      = "{CATALOG}"
SCHEMA_NAME  = "{SCHEMA_NAME}"
SCHEMA       = f"{{CATALOG}}.{{SCHEMA_NAME}}"
VOLUME_PATH  = "{VOLUME_PATH}"
FG_TABLE     = f"{{SCHEMA}}.{FG_NAME}"
TD_TABLE     = f"{{SCHEMA}}.{TD_NAME}"
PRED_TABLE   = f"{{SCHEMA}}.{PRED_NAME}"
MDL_NAME     = "{MDL_NAME}"
MDL_UC       = f"{{SCHEMA}}.{{MDL_NAME}}"
EXPERIMENT   = "{EXPERIMENT_PATH}"

spark.sql(f"USE CATALOG {{CATALOG}}")
spark.sql(f"USE SCHEMA {{SCHEMA_NAME}}")

# COMMAND ----------
# MAGIC %md ## Load raw data
txn = spark.read.csv(f"{{VOLUME_PATH}}/transactions.csv", header=True, inferSchema=True)
score = spark.read.csv(f"{{VOLUME_PATH}}/score_transactions.csv", header=True, inferSchema=True)

# parse datetime
txn   = txn.withColumn("ts",   F.to_timestamp("datetime"))
score = score.withColumn("ts", F.to_timestamp("datetime"))

# COMMAND ----------
# MAGIC %md ## Feature Engineering
def engineer_features(df, ref_df=None):
    """
    Engineer fraud features:
    - transaction velocity (1h, 24h, 7d)
    - amount stats (mean/std in 24h)
    - hour of day, day of week
    - geo distance from card home location
    - amount ratios
    """
    df = df.withColumn("ts_unix", F.unix_timestamp("ts"))
    df = df.withColumn("hour_of_day", F.hour("ts").cast("double"))
    df = df.withColumn("day_of_week", F.dayofweek("ts").cast("double"))
    df = df.withColumn("is_weekend", (F.dayofweek("ts").isin([1, 7])).cast("double"))
    df = df.withColumn("amount_log", F.log1p("amount"))

    # High-risk category flag
    high_risk = ["online", "travel", "cash_advance"]
    df = df.withColumn("is_high_risk_cat",
        F.when(F.col("category").isin(high_risk), 1.0).otherwise(0.0))

    # Velocity windows (using Window with row_number approach on 1h/24h)
    # We'll compute per-card aggregates over preceding rows
    w1h  = Window.partitionBy("cc_num").orderBy("ts_unix").rangeBetween(-3600, 0)
    w24h = Window.partitionBy("cc_num").orderBy("ts_unix").rangeBetween(-86400, 0)
    w7d  = Window.partitionBy("cc_num").orderBy("ts_unix").rangeBetween(-604800, 0)

    df = df.withColumn("tx_count_1h",    F.count("*").over(w1h).cast("double"))
    df = df.withColumn("tx_count_24h",   F.count("*").over(w24h).cast("double"))
    df = df.withColumn("tx_count_7d",    F.count("*").over(w7d).cast("double"))
    df = df.withColumn("tx_amount_sum_24h",  F.sum("amount").over(w24h))
    df = df.withColumn("tx_amount_mean_24h", F.mean("amount").over(w24h))
    df = df.withColumn("tx_amount_max_24h",  F.max("amount").over(w24h))
    df = df.withColumn("amount_ratio_24h",
        F.when(F.col("tx_amount_mean_24h") > 0,
               F.col("amount") / F.col("tx_amount_mean_24h"))
         .otherwise(1.0))

    # Geo distance: distance from card median location (computed from training set)
    if ref_df is None:
        ref_df = df
    card_home = ref_df.groupBy("cc_num").agg(
        F.mean("lat").alias("home_lat"),
        F.mean("long").alias("home_long")
    )
    df = df.join(card_home, on="cc_num", how="left")
    # Haversine approx (scaled by 111 km/degree)
    df = df.withColumn("geo_distance",
        F.sqrt(
            F.pow((F.col("lat") - F.col("home_lat")) * 111.0, 2) +
            F.pow((F.col("long") - F.col("home_long")) * 111.0 *
                  F.cos(F.col("home_lat") * 3.14159265 / 180.0), 2)
        ))
    df = df.withColumn("geo_distance",
        F.when(F.col("geo_distance").isNull(), 0.0).otherwise(F.col("geo_distance")))

    # Amount z-score over 7d window
    w7d_std = Window.partitionBy("cc_num").orderBy("ts_unix").rangeBetween(-604800, 0)
    df = df.withColumn("tx_amount_std_7d", F.stddev("amount").over(w7d_std))
    df = df.withColumn("amount_zscore_7d",
        F.when((F.col("tx_amount_std_7d") > 0) & F.col("tx_amount_std_7d").isNotNull(),
               (F.col("amount") - F.col("tx_amount_mean_24h")) / F.col("tx_amount_std_7d"))
         .otherwise(0.0))

    # Night transaction flag (22-6)
    df = df.withColumn("is_night",
        F.when((F.col("hour_of_day") >= 22) | (F.col("hour_of_day") < 6), 1.0)
         .otherwise(0.0))

    # Fill nulls in feature columns
    feat_cols = ["tx_count_1h","tx_count_24h","tx_count_7d","tx_amount_sum_24h",
                 "tx_amount_mean_24h","tx_amount_max_24h","amount_ratio_24h",
                 "geo_distance","tx_amount_std_7d","amount_zscore_7d"]
    df = df.fillna(0.0, subset=feat_cols)

    return df, card_home

print("Engineering features on training data...")
txn_feat, card_home_ref = engineer_features(txn)

# COMMAND ----------
# MAGIC %md ## Save Feature Group
print(f"Writing feature group to {{FG_TABLE}}...")
txn_feat.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(FG_TABLE)
print(f"Feature group written: {{FG_TABLE}}")

# COMMAND ----------
# MAGIC %md ## Create Training Dataset
print(f"Creating training dataset {{TD_TABLE}}...")
FEATURE_COLS = [
    "hour_of_day","day_of_week","is_weekend","amount_log",
    "is_high_risk_cat","tx_count_1h","tx_count_24h","tx_count_7d",
    "tx_amount_mean_24h","tx_amount_max_24h","amount_ratio_24h",
    "geo_distance","amount_zscore_7d","is_night","amount"
]
td = txn_feat.select(
    ["transaction_id","cc_num","ts","is_fraud"] + FEATURE_COLS
)
td.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(TD_TABLE)
print(f"Training dataset written: {{TD_TABLE}}")

# COMMAND ----------
# MAGIC %md ## Train Classifier
td_df = spark.table(TD_TABLE)
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip")
td_df = assembler.transform(td_df).withColumnRenamed("is_fraud", "label")
td_df = td_df.withColumn("label", F.col("label").cast("double"))

train_df, val_df = td_df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
val_df.cache()

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run() as run:
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=200, maxDepth=8, seed=42,
        featureSubsetStrategy="sqrt"
    )
    model = rf.fit(train_df)

    # Evaluate
    val_preds = model.transform(val_df)
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(val_preds)
    print(f"Validation ROC AUC: {{auc:.4f}}")

    mlflow.log_metric("roc_auc", auc)
    mlflow.log_param("numTrees", 200)
    mlflow.log_param("maxDepth", 8)

    # Log model
    mlflow.spark.log_model(model, "model", registered_model_name=MDL_UC)

    run_id = run.info.run_id

print(f"MLflow run_id: {{run_id}}")

# COMMAND ----------
# MAGIC %md ## Score transactions
print("Engineering features on score data (using training card home locations)...")
score_feat, _ = engineer_features(score, ref_df=txn)

# Re-apply assembler on score data
assembler2 = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip")
score_feat_vec = assembler2.transform(score_feat)

# Load the registered model
client = MlflowClient()
mv = client.get_registered_model(MDL_UC)
latest_version = sorted(mv.latest_versions, key=lambda x: int(x.version))[-1].version
model_uri = f"models:/{{MDL_UC}}/{{latest_version}}"
print(f"Loading model from: {{model_uri}}")
loaded_model = mlflow.spark.load_model(model_uri)

preds = loaded_model.transform(score_feat_vec)

# Extract probability of fraud (class 1)
extract_prob = F.udf(lambda v: float(v[1]), DoubleType())
preds_out = preds.withColumn("fraud_probability", extract_prob("probability")) \\
                  .select("transaction_id", "fraud_probability")

print(f"Writing predictions to {{PRED_TABLE}}...")
preds_out.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(PRED_TABLE)
print(f"Predictions written: {{PRED_TABLE}}")

# COMMAND ----------
# MAGIC %md ## Verify outputs
fg_count = spark.table(FG_TABLE).count()
td_count = spark.table(TD_TABLE).count()
pred_count = spark.table(PRED_TABLE).count()
print(f"Feature group rows: {{fg_count}}")
print(f"Training dataset rows: {{td_count}}")
print(f"Predictions rows: {{pred_count}}")

sample_preds = spark.table(PRED_TABLE).show(5)
print(f"ROC AUC: {{auc:.4f}}")
print("Pipeline complete!")
'''

# Encode notebook content to base64
nb_bytes = NOTEBOOK_CONTENT.encode("utf-8")
nb_b64 = base64.b64encode(nb_bytes).decode("utf-8")

# Create directory in workspace
dir_path = f"/Users/{USER_EMAIL}/{PREFIX}"
try:
    w.workspace.mkdirs(dir_path)
    print(f"    Created directory {dir_path}")
except Exception as e:
    print(f"    Directory exists or error: {e}")

# Import notebook
w.workspace.import_(
    path=NOTEBOOK_PATH,
    format=workspace.ImportFormat.SOURCE,
    language=workspace.Language.PYTHON,
    content=nb_b64,
    overwrite=True
)
print(f"    Notebook created at {NOTEBOOK_PATH}")

# ── 4. Find or create a cluster / use serverless ─────────────────────────────
print("[4] Creating job to run pipeline ...")

JOB_NAME = f"{PREFIX}_ccfraud_pipeline"

# Use the newest LTS cluster - check existing clusters
clusters = list(w.clusters.list())
print(f"    Found {len(clusters)} clusters")

# Find a running cluster
running_cluster = None
for c in clusters:
    if c.state and c.state.value in ("RUNNING", "RESIZING"):
        running_cluster = c
        print(f"    Using running cluster: {c.cluster_id} ({c.cluster_name})")
        break

# Create task with proper SDK objects - serverless (no cluster specified)
nb_task = jobs_svc.NotebookTask(
    notebook_path=NOTEBOOK_PATH,
    source=jobs_svc.Source.WORKSPACE
)

task = jobs_svc.Task(
    task_key="pipeline",
    notebook_task=nb_task
)

job = w.jobs.create(name=JOB_NAME, tasks=[task])
job_id = job.job_id
print(f"    Job created: {job_id} ({JOB_NAME})")

# ── 5. Run the job ────────────────────────────────────────────────────────────
print("[5] Triggering job run ...")
run = w.jobs.run_now(job_id=job_id)
run_id = run.run_id
print(f"    Run triggered: run_id={run_id}")

# ── 6. Wait for completion ────────────────────────────────────────────────────
print("[6] Waiting for job to complete ...")
start = time.time()
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = run_state.state.life_cycle_state.value if run_state.state else "UNKNOWN"
    result_state = run_state.state.result_state.value if (run_state.state and run_state.state.result_state) else "N/A"
    elapsed = int(time.time() - start)
    print(f"    [{elapsed}s] State: {life_cycle} / {result_state}")

    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        if result_state == "SUCCESS":
            print("[6] Job completed successfully!")
        else:
            print(f"[6] Job FAILED: {result_state}")
            # Get error
            for task in (run_state.tasks or []):
                print(f"    Task {task.task_key}: {task.state}")
            # Get run output
            try:
                output = w.jobs.get_run_output(run_id=run_state.tasks[0].run_id if run_state.tasks else run_id)
                if output.error:
                    print(f"    Error: {output.error}")
                if output.notebook_output:
                    print(f"    Output: {output.notebook_output.result[:2000]}")
            except Exception as e:
                print(f"    Could not get output: {e}")
        break

    time.sleep(30)

# ── 7. Create online table for predictions ────────────────────────────────────
print("[7] Creating online table for predictions ...")
try:
    from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

    ot = w.online_tables.create(
        name=f"{CATALOG}.{SCHEMA_NAME}.{PRED_NAME}_online",
        table=OnlineTable(
            name=f"{CATALOG}.{SCHEMA_NAME}.{PRED_NAME}_online",
            spec=OnlineTableSpec(
                source_table_full_name=f"{CATALOG}.{SCHEMA_NAME}.{PRED_NAME}",
                primary_key_columns=["transaction_id"],
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
            )
        )
    )
    print(f"    Online table creation initiated: {ot}")
except Exception as e:
    print(f"    Online table creation failed (may need to enable): {e}")
    # Try alternate approach
    try:
        from databricks.sdk.service import catalog as cat_svc
        result = w.online_tables.create(
            name=f"{CATALOG}.{SCHEMA_NAME}.{PRED_NAME}",
            spec={
                "source_table_full_name": f"{CATALOG}.{SCHEMA_NAME}.{PRED_NAME}",
                "primary_key_columns": ["transaction_id"],
                "run_triggered": {}
            }
        )
        print(f"    Online table: {result}")
    except Exception as e2:
        print(f"    Second attempt failed: {e2}")

print("[8] Pipeline complete!")
print(f"  Feature group:    {CATALOG}.{SCHEMA_NAME}.{FG_NAME}")
print(f"  Training dataset: {CATALOG}.{SCHEMA_NAME}.{TD_NAME}")
print(f"  Model:            {CATALOG}.{SCHEMA_NAME}.{MDL_NAME}")
print(f"  Predictions:      {CATALOG}.{SCHEMA_NAME}.{PRED_NAME}")
