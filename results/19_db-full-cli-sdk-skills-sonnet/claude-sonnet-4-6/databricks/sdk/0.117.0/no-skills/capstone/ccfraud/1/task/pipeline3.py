"""
Phase 2: Train model on existing feature data and score.
"""
import os
import base64
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs as jobs_svc, workspace

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
USER_EMAIL = "benedict@logicalclocks.com"
NOTEBOOK_PATH = f"/Users/{USER_EMAIL}/{PREFIX}/ccfraud_train_score"
EXPERIMENT_PATH = f"/Users/{USER_EMAIL}/{PREFIX}/ccfraud_exp2"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/data_upload"

FG_TABLE   = f"{CATALOG}.{SCHEMA_NAME}.cctxn739ee9"
TD_TABLE   = f"{CATALOG}.{SCHEMA_NAME}.cctd739ee9"
PRED_TABLE = f"{CATALOG}.{SCHEMA_NAME}.ccpred739ee9"
MDL_UC     = f"{CATALOG}.{SCHEMA_NAME}.ccmodel739ee9"

w = WorkspaceClient()

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

FEAT_COLS = ["hour_of_day","day_of_week","is_weekend","amount_log","is_high_risk","is_night",
             "cnt_1h","cnt_24h","cnt_7d","sum_24h","mean_24h","max_24h","std_7d","ratio_24h","zscore_7d","geo_dist","amount"]
print("Setup done")

# COMMAND ----------
# Train on existing training dataset
td_df = spark.table(TD_TABLE)
asm = VectorAssembler(inputCols=FEAT_COLS, outputCol="features", handleInvalid="skip")
td_df = asm.transform(td_df).withColumn("label", F.col("is_fraud").cast("double"))

train_df, val_df = td_df.randomSplit([0.8, 0.2], seed=42)
print("Train count:", train_df.count())
print("Val count:", val_df.count())

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

    # Create signature for UC model registry compliance
    from mlflow.models.signature import infer_signature
    sample = train_df.select(FEAT_COLS).limit(5).toPandas()
    sample_preds = val_preds.select("probability").limit(5).toPandas()
    try:
        sig = infer_signature(sample, sample_preds)
    except Exception:
        sig = None

    mlflow.spark.log_model(model, "model", registered_model_name=MDL_UC,
                           dfs_tmpdir=VOLUME_PATH, signature=sig)
    mlflow_run_id = run.info.run_id

print("Model trained and registered:", MDL_UC)

# COMMAND ----------
# Engineer features for score_transactions.csv using reference from training set
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

# Load raw score data and training reference
score_raw = spark.read.csv(VOLUME_PATH + "/score_transactions.csv", header=True, inferSchema=True)
score_raw = score_raw.withColumn("ts", F.to_timestamp("datetime"))
txn_ref = spark.table(FG_TABLE).select("cc_num","lat","long")

print("Score rows:", score_raw.count())

score_feat = make_features(score_raw, home_df=txn_ref)
asm2 = VectorAssembler(inputCols=FEAT_COLS, outputCol="features", handleInvalid="skip")
score_vec = asm2.transform(score_feat)

get_prob = F.udf(lambda v: float(v[1]), DoubleType())
preds_out = model.transform(score_vec).withColumn("fraud_probability", get_prob("probability")).select("transaction_id", "fraud_probability")
preds_out.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(PRED_TABLE)
pred_cnt = spark.table(PRED_TABLE).count()
print("Predictions saved:", PRED_TABLE, "rows:", pred_cnt)
spark.table(PRED_TABLE).show(5)
dbutils.notebook.exit("SUCCESS AUC=" + str(round(auc, 4)) + " PREDS=" + str(pred_cnt))
"""

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

JOB_NAME = f"{PREFIX}_ccfraud_v3"
task = jobs_svc.Task(
    task_key="train_score",
    notebook_task=jobs_svc.NotebookTask(
        notebook_path=NOTEBOOK_PATH,
        source=jobs_svc.Source.WORKSPACE
    )
)
job = w.jobs.create(name=JOB_NAME, tasks=[task])
print(f"  Job created: {job.job_id}")

run = w.jobs.run_now(job_id=job.job_id)
run_id = run.run_id
print(f"  Run: {run_id}")

start = time.time()
while True:
    r = w.jobs.get_run(run_id=run_id)
    lc = r.state.life_cycle_state.value
    rs = r.state.result_state.value if r.state.result_state else "N/A"
    elapsed = int(time.time() - start)
    print(f"  [{elapsed}s] {lc}/{rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        if r.tasks:
            out = w.jobs.get_run_output(run_id=r.tasks[0].run_id)
            print("  error:", out.error)
            if out.error_trace:
                print("  trace:", out.error_trace[:2000])
            print("  exit:", out.notebook_output.result if out.notebook_output else "none")
        break
    time.sleep(20)

# Check tables
print("\nVerifying tables ...")
wh_id = "4dfab06c923fe3cc"
for tbl in [f"{CATALOG}.{SCHEMA_NAME}.cctxn739ee9",
            f"{CATALOG}.{SCHEMA_NAME}.cctd739ee9",
            f"{CATALOG}.{SCHEMA_NAME}.ccpred739ee9"]:
    try:
        t = w.tables.get(tbl)
        print(f"  {tbl}: OK ({t.table_type})")
    except Exception as e:
        print(f"  {tbl}: MISSING - {e}")

# Check model
try:
    mdl = w.registered_models.get(full_name=MDL_UC)
    print(f"  Model {MDL_UC}: OK ({mdl.name})")
except Exception as e:
    print(f"  Model: MISSING - {e}")

# Try to create online/synced table for predictions
print("\nCreating serving for predictions ...")
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
    print("  Online table created:", ot)
except Exception as e:
    print(f"  Online table error: {e}")

print("\nPipeline complete!")
