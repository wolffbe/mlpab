"""
Full FTI pipeline for credit card fraud detection on Databricks.
Creates and runs a notebook via the Databricks SDK.
"""
import os
import base64
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs as jobs_svc, workspace
from databricks.sdk.service.catalog import VolumeType

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
USER_EMAIL = "benedict@logicalclocks.com"
NOTEBOOK_PATH = f"/Users/{USER_EMAIL}/{PREFIX}/ccfraud_pipeline2"
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/{PREFIX}/ccfraud_experiment"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/data_upload"

FG_TABLE   = f"{CATALOG}.{SCHEMA_NAME}.cctxn739ee9"
TD_TABLE   = f"{CATALOG}.{SCHEMA_NAME}.cctd739ee9"
PRED_TABLE = f"{CATALOG}.{SCHEMA_NAME}.ccpred739ee9"
MDL_UC     = f"{CATALOG}.{SCHEMA_NAME}.ccmodel739ee9"

w = WorkspaceClient()

# ── Notebook content (pure Python, no MAGIC cells) ──────────────────────────
NOTEBOOK = f"""# Databricks notebook source
# COMMAND ----------
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

CATALOG     = "{CATALOG}"
SCHEMA_NAME = "{SCHEMA_NAME}"
VOLUME_PATH = "{VOLUME_PATH}"
FG_TABLE    = "{FG_TABLE}"
TD_TABLE    = "{TD_TABLE}"
PRED_TABLE  = "{PRED_TABLE}"
MDL_UC      = "{MDL_UC}"
EXPERIMENT  = "{EXPERIMENT_PATH}"

print("Imports OK")

# COMMAND ----------
# Load raw data
txn = spark.read.csv(VOLUME_PATH + "/transactions.csv", header=True, inferSchema=True)
txn = txn.withColumn("ts", F.to_timestamp("datetime"))
score_raw = spark.read.csv(VOLUME_PATH + "/score_transactions.csv", header=True, inferSchema=True)
score_raw = score_raw.withColumn("ts", F.to_timestamp("datetime"))
print("Loaded txn:", txn.count(), "score:", score_raw.count())

# COMMAND ----------
# Feature engineering
def make_features(df, home_df=None):
    df = df.withColumn("ts_unix", F.unix_timestamp("ts"))
    df = df.withColumn("hour_of_day",  F.hour("ts").cast("double"))
    df = df.withColumn("day_of_week",  F.dayofweek("ts").cast("double"))
    df = df.withColumn("is_weekend",   (F.dayofweek("ts").isin([1, 7])).cast("double"))
    df = df.withColumn("amount_log",   F.log1p("amount"))
    df = df.withColumn("is_high_risk",
        F.when(F.col("category").isin(["online","travel","cash_advance"]), 1.0).otherwise(0.0))
    df = df.withColumn("is_night",
        F.when((F.col("hour_of_day") >= 22) | (F.col("hour_of_day") < 6), 1.0).otherwise(0.0))

    w1h  = Window.partitionBy("cc_num").orderBy("ts_unix").rangeBetween(-3600, 0)
    w24h = Window.partitionBy("cc_num").orderBy("ts_unix").rangeBetween(-86400, 0)
    w7d  = Window.partitionBy("cc_num").orderBy("ts_unix").rangeBetween(-604800, 0)

    df = df.withColumn("cnt_1h",   F.count("transaction_id").over(w1h).cast("double"))
    df = df.withColumn("cnt_24h",  F.count("transaction_id").over(w24h).cast("double"))
    df = df.withColumn("cnt_7d",   F.count("transaction_id").over(w7d).cast("double"))
    df = df.withColumn("sum_24h",  F.sum("amount").over(w24h))
    df = df.withColumn("mean_24h", F.mean("amount").over(w24h))
    df = df.withColumn("max_24h",  F.max("amount").over(w24h))
    df = df.withColumn("std_7d",   F.stddev("amount").over(w7d))
    df = df.withColumn("ratio_24h",
        F.when(F.col("mean_24h") > 0, F.col("amount") / F.col("mean_24h")).otherwise(1.0))
    df = df.withColumn("zscore_7d",
        F.when((F.col("std_7d") > 0) & F.col("std_7d").isNotNull(),
               (F.col("amount") - F.col("mean_24h")) / F.col("std_7d")).otherwise(0.0))

    ref = home_df if home_df is not None else df
    card_home = ref.groupBy("cc_num").agg(
        F.mean("lat").alias("home_lat"), F.mean("long").alias("home_long"))
    df = df.join(card_home, on="cc_num", how="left")
    df = df.withColumn("geo_dist",
        F.sqrt(F.pow((F.col("lat")-F.col("home_lat"))*111.0, 2) +
               F.pow((F.col("long")-F.col("home_long"))*111.0*
                     F.cos(F.col("home_lat")*3.14159265/180.0), 2)))
    df = df.fillna(0.0, subset=["cnt_1h","cnt_24h","cnt_7d","sum_24h","mean_24h",
                                 "max_24h","std_7d","ratio_24h","zscore_7d","geo_dist"])
    return df

txn_feat = make_features(txn)
print("Feature engineering done, cols:", len(txn_feat.columns))

# COMMAND ----------
# Save feature group
txn_feat.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(FG_TABLE)
print("Feature group saved:", FG_TABLE)

# COMMAND ----------
# Create training dataset
FEAT_COLS = ["hour_of_day","day_of_week","is_weekend","amount_log","is_high_risk","is_night",
             "cnt_1h","cnt_24h","cnt_7d","sum_24h","mean_24h","max_24h","std_7d","ratio_24h","zscore_7d","geo_dist","amount"]
td = txn_feat.select(["transaction_id","cc_num","ts","is_fraud"] + FEAT_COLS)
td.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(TD_TABLE)
print("Training dataset saved:", TD_TABLE)

# COMMAND ----------
# Train classifier
td_df = spark.table(TD_TABLE)
asm = VectorAssembler(inputCols=FEAT_COLS, outputCol="features", handleInvalid="skip")
td_df = asm.transform(td_df).withColumn("label", F.col("is_fraud").cast("double"))

train_df, val_df = td_df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
val_df.cache()
print("Train:", train_df.count(), "Val:", val_df.count())

mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run() as run:
    gbt = GBTClassifier(featuresCol="features", labelCol="label",
                        maxIter=50, maxDepth=5, seed=42)
    model = gbt.fit(train_df)

    val_preds = model.transform(val_df)
    ev = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = ev.evaluate(val_preds)
    print("Val ROC AUC:", round(auc, 4))

    mlflow.log_metric("roc_auc", auc)
    mlflow.log_param("maxIter", 50)
    mlflow.log_param("maxDepth", 5)
    mlflow.spark.log_model(model, "model", registered_model_name=MDL_UC)
    run_id_val = run.info.run_id

print("MLflow run_id:", run_id_val)
dbutils.notebook.exit("AUC=" + str(round(auc, 4)))

# COMMAND ----------
# Score transactions
score_feat = make_features(score_raw, home_df=txn)
asm2 = VectorAssembler(inputCols=FEAT_COLS, outputCol="features", handleInvalid="skip")
score_vec = asm2.transform(score_feat)

get_prob = F.udf(lambda v: float(v[1]), DoubleType())
preds_out = model.transform(score_vec).withColumn("fraud_probability", get_prob("probability")).select("transaction_id", "fraud_probability")
preds_out.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(PRED_TABLE)
cnt = spark.table(PRED_TABLE).count()
print("Predictions saved:", PRED_TABLE, "rows:", cnt)
print("Sample preds:")
spark.table(PRED_TABLE).show(5)
print("Pipeline complete!")
dbutils.notebook.exit("AUC=" + str(round(auc, 4)) + " PREDS=" + str(cnt))
"""

# Upload notebook
print(f"Creating notebook at {NOTEBOOK_PATH} ...")
try:
    w.workspace.mkdirs(f"/Users/{USER_EMAIL}/{PREFIX}")
except:
    pass

w.workspace.import_(
    path=NOTEBOOK_PATH,
    format=workspace.ImportFormat.SOURCE,
    language=workspace.Language.PYTHON,
    content=base64.b64encode(NOTEBOOK.encode()).decode(),
    overwrite=True
)
print("  Notebook created.")

# Create and run job
JOB_NAME = f"{PREFIX}_ccfraud_v2"
task = jobs_svc.Task(
    task_key="pipeline",
    notebook_task=jobs_svc.NotebookTask(
        notebook_path=NOTEBOOK_PATH,
        source=jobs_svc.Source.WORKSPACE
    )
)
job = w.jobs.create(name=JOB_NAME, tasks=[task])
print(f"  Job created: {job.job_id}")

run = w.jobs.run_now(job_id=job.job_id)
run_id = run.run_id
print(f"  Run triggered: run_id={run_id}")

# Wait
start = time.time()
while True:
    r = w.jobs.get_run(run_id=run_id)
    lc = r.state.life_cycle_state.value
    rs = r.state.result_state.value if r.state.result_state else "N/A"
    elapsed = int(time.time() - start)
    print(f"  [{elapsed}s] {lc}/{rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        print(f"  Final state: {lc}/{rs}")
        if r.tasks:
            out = w.jobs.get_run_output(run_id=r.tasks[0].run_id)
            print("  error:", out.error)
            print("  error_trace:", out.error_trace[:3000] if out.error_trace else None)
            print("  NB exit:", out.notebook_output.result if out.notebook_output else "none")
        break
    time.sleep(20)

# Create online table
print("Creating online table for predictions ...")
try:
    from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
    ot = w.online_tables.create(
        table=OnlineTable(
            name=f"{CATALOG}.{SCHEMA_NAME}.ccpred739ee9_online",
            spec=OnlineTableSpec(
                source_table_full_name=PRED_TABLE,
                primary_key_columns=["transaction_id"],
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
            )
        )
    )
    print("  Online table:", ot.name if hasattr(ot, 'name') else ot)
except Exception as e:
    print(f"  Online table error: {e}")

print("Done!")
