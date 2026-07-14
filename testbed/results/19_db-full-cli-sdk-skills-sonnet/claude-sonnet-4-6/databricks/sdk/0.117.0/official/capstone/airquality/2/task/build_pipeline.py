"""
Build and submit the full FTI pipeline as a Databricks notebook job.
"""
import os
import base64
import time
import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import SubmitTask, NotebookTask, Source
from databricks.sdk.service.compute import ClusterSpec, DataSecurityMode

w = WorkspaceClient()

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpab94ed10
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]    # mlpab94ed10
CATALOG = SCHEMA.split(".")[0]                     # workspace
DB = SCHEMA.split(".")[1]                          # mlpab94ed10

me = w.current_user.me()
user = me.user_name
print(f"User: {user}")
print(f"Schema: {SCHEMA}")
print(f"Prefix: {PREFIX}")

NOTEBOOK_DIR = f"/Users/{user}/{PREFIX}"

# Read CSV data
with open("data/airquality_history.csv") as f:
    history_csv = f.read()

with open("data/forecast_days.csv") as f:
    forecast_csv = f.read()

# Escape for embedding in Python string
history_escaped = history_csv.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
forecast_escaped = forecast_csv.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

# Create directory
try:
    w.workspace.mkdirs(NOTEBOOK_DIR)
    print(f"Created: {NOTEBOOK_DIR}")
except Exception as e:
    print(f"Dir exists: {e}")

# The notebook code - uses triple-quoted strings for CSV data
# Note: double-braces {{}} become single braces {} in the resulting notebook
notebook_source = '''# Databricks notebook source
# COMMAND ----------
# Setup

SCHEMA = "{SCHEMA}"
CATALOG = "{CATALOG}"
DB = "{DB}"
FG_NAME = "airqf4aae3"
TD_NAME = "airqtdf4aae3"
MODEL_NAME = "airqmodelf4aae3"
PRED_NAME = "airqpredf4aae3"
USER = "{USER}"
PREFIX = "{PREFIX}"

print(f"Catalog: {{CATALOG}}, DB: {{DB}}")

# COMMAND ----------
# Load data

HISTORY_CSV = """{HISTORY_CSV}"""

FORECAST_CSV = """{FORECAST_CSV}"""

# COMMAND ----------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, to_date, lit
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import (StructType, StructField, StringType,
                                DoubleType, DateType)

spark = SparkSession.builder.getOrCreate()

def parse_csv_to_df(csv_str, schema):
    lines = [l.strip() for l in csv_str.strip().split("\\n") if l.strip()]
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        vals = line.split(",")
        row = []
        for i, field in enumerate(schema.fields):
            v = vals[i]
            if field.dataType == DoubleType():
                row.append(float(v))
            else:
                row.append(v)
        rows.append(tuple(row))
    return spark.createDataFrame(rows, schema=schema)

hist_schema = StructType([
    StructField("date", StringType()),
    StructField("pm25_lag1", DoubleType()),
    StructField("temperature", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("wind_speed", DoubleType()),
    StructField("pressure", DoubleType()),
    StructField("precipitation", DoubleType()),
    StructField("pm25", DoubleType()),
])

fore_schema = StructType([
    StructField("date", StringType()),
    StructField("pm25_lag1", DoubleType()),
    StructField("temperature", DoubleType()),
    StructField("humidity", DoubleType()),
    StructField("wind_speed", DoubleType()),
    StructField("pressure", DoubleType()),
    StructField("precipitation", DoubleType()),
])

df_hist = parse_csv_to_df(HISTORY_CSV, hist_schema).withColumn("date", to_date("date"))
df_fore = parse_csv_to_df(FORECAST_CSV, fore_schema).withColumn("date", to_date("date"))

print(f"History: {{df_hist.count()}} rows")
print(f"Forecast: {{df_fore.count()}} rows")

# COMMAND ----------
# Feature engineering
# Combine history+forecast (with pm25=null for forecast) to compute rolling windows

df_fore_nullpm25 = df_fore.withColumn("pm25", lit(None).cast(DoubleType()))
df_all = df_hist.union(df_fore_nullpm25).orderBy("date")

w7 = Window.orderBy(F.unix_timestamp("date")).rowsBetween(-6, 0)
w3 = Window.orderBy(F.unix_timestamp("date")).rowsBetween(-2, 0)

df_feat = df_all \\
    .withColumn("pm25_roll7", F.avg("pm25_lag1").over(w7)) \\
    .withColumn("pm25_roll3", F.avg("pm25_lag1").over(w3)) \\
    .withColumn("temp_roll3", F.avg("temperature").over(w3)) \\
    .withColumn("humidity_roll3", F.avg("humidity").over(w3))

print("Feature columns:")
df_feat.printSchema()

# COMMAND ----------
# Write feature group - history only (with pm25)
df_fg = df_feat.filter(col("pm25").isNotNull())
print(f"Feature group rows: {{df_fg.count()}}")
df_fg.show(3)

spark.sql(f"USE CATALOG {{CATALOG}}")

df_fg.writeTo(f"{{CATALOG}}.{{DB}}.{{FG_NAME}}") \\
    .using("delta") \\
    .tableProperty("delta.enableChangeDataFeed", "true") \\
    .createOrReplace()

print(f"Feature group written: {{CATALOG}}.{{DB}}.{{FG_NAME}}")

# COMMAND ----------
# Training dataset

df_td = spark.table(f"{{CATALOG}}.{{DB}}.{{FG_NAME}}")

df_td.writeTo(f"{{CATALOG}}.{{DB}}.{{TD_NAME}}") \\
    .using("delta") \\
    .createOrReplace()

print(f"Training dataset: {{CATALOG}}.{{DB}}.{{TD_NAME}}")

# COMMAND ----------
# Train model with MLflow

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")
exp_path = f"/Users/{{USER}}/{{PREFIX}}/airquality_exp"
try:
    mlflow.create_experiment(exp_path)
except:
    pass
mlflow.set_experiment(exp_path)

pdf_td = df_td.toPandas()
pdf_td = pdf_td.sort_values("date").reset_index(drop=True)
pdf_td = pdf_td.ffill().bfill()

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed",
            "pressure", "precipitation", "pm25_roll7", "pm25_roll3",
            "temp_roll3", "humidity_roll3"]

X = pdf_td[FEATURES].values
y = pdf_td["pm25"].values

split_idx = int(len(pdf_td) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
print(f"Train: {{len(X_train)}}, Val: {{len(X_val)}}")

with mlflow.start_run() as run:
    params = dict(n_estimators=300, learning_rate=0.08, max_depth=4,
                  subsample=0.8, min_samples_leaf=3, random_state=42)
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    mae = float(np.mean(np.abs(y_val - y_pred_val)))
    print(f"Val RMSE: {{rmse:.4f}}, MAE: {{mae:.4f}}")

    for k, v in params.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("val_rmse", rmse)
    mlflow.log_param("features", ",".join(FEATURES))

    sig = infer_signature(
        pd.DataFrame(X_train, columns=FEATURES),
        model.predict(X_train)
    )

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=sig,
        input_example=pd.DataFrame(X_val[:3], columns=FEATURES),
        registered_model_name=f"{{CATALOG}}.{{DB}}.{{MODEL_NAME}}"
    )
    run_id = run.info.run_id

print(f"Run ID: {{run_id}}")
print(f"Model: {{CATALOG}}.{{DB}}.{{MODEL_NAME}}")

# COMMAND ----------
# Predict on forecast days

df_fore_feat = df_feat.filter(col("pm25").isNull()).drop("pm25")
pdf_fore = df_fore_feat.toPandas()
pdf_fore = pdf_fore.sort_values("date").reset_index(drop=True)
pdf_fore = pdf_fore.ffill().bfill()

print(f"Forecast rows: {{len(pdf_fore)}}")

loaded = mlflow.sklearn.load_model(f"models:/{{CATALOG}}.{{DB}}.{{MODEL_NAME}}/1")
preds = loaded.predict(pdf_fore[FEATURES].values)
pdf_fore["pm25_pred"] = preds
print(pdf_fore[["date", "pm25_pred"]])

# COMMAND ----------
# Write predictions table

from pyspark.sql.types import DateType

pred_rows = [(str(row["date"]), float(row["pm25_pred"])) for _, row in pdf_fore.iterrows()]
pred_schema = StructType([
    StructField("date", StringType()),
    StructField("pm25_pred", DoubleType()),
])
df_pred = spark.createDataFrame(pred_rows, schema=pred_schema) \\
               .withColumn("date", to_date("date"))

df_pred.writeTo(f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}") \\
    .using("delta") \\
    .tableProperty("delta.enableChangeDataFeed", "true") \\
    .createOrReplace()

print(f"Predictions: {{CATALOG}}.{{DB}}.{{PRED_NAME}}")
df_pred.show()

# COMMAND ----------
print("=== PIPELINE COMPLETE ===")
spark.sql(f"SELECT COUNT(*) FROM {{CATALOG}}.{{DB}}.{{FG_NAME}}").show()
spark.sql(f"SELECT COUNT(*) FROM {{CATALOG}}.{{DB}}.{{TD_NAME}}").show()
spark.sql(f"SELECT COUNT(*) FROM {{CATALOG}}.{{DB}}.{{PRED_NAME}}").show()
'''

# Substitute template values
notebook_source = notebook_source.replace("{SCHEMA}", SCHEMA)
notebook_source = notebook_source.replace("{CATALOG}", CATALOG)
notebook_source = notebook_source.replace("{DB}", DB)
notebook_source = notebook_source.replace("{USER}", user)
notebook_source = notebook_source.replace("{PREFIX}", PREFIX)
notebook_source = notebook_source.replace("{HISTORY_CSV}", history_escaped)
notebook_source = notebook_source.replace("{FORECAST_CSV}", forecast_escaped)

# Upload notebook
notebook_path = f"{NOTEBOOK_DIR}/airquality_pipeline"
content_b64 = base64.b64encode(notebook_source.encode()).decode()

w.workspace.import_(
    path=notebook_path,
    format="SOURCE",
    language="PYTHON",
    content=content_b64,
    overwrite=True,
)
print(f"Notebook uploaded: {notebook_path}")

# Submit job
job_name = f"{PREFIX}_airq_pipeline"

from databricks.sdk.service.compute import ClusterSpec, DataSecurityMode
from databricks.sdk.service.jobs import SubmitTask, NotebookTask, Source

print(f"Submitting job: {job_name}")
run = w.jobs.submit(
    run_name=job_name,
    tasks=[
        SubmitTask(
            task_key="pipeline",
            notebook_task=NotebookTask(
                notebook_path=notebook_path,
                source=Source.WORKSPACE,
            ),
            new_cluster=ClusterSpec(
                spark_version="15.4.x-cpu-ml-scala2.12",
                node_type_id="r3.xlarge",
                num_workers=1,
                data_security_mode=DataSecurityMode.SINGLE_USER,
                single_user_name=user,
            ),
        )
    ]
)
run_id = run.run_id
print(f"Job run_id: {run_id}")

# Poll for completion
print("Waiting for job to complete...")
while True:
    run_info = w.jobs.get_run(run_id=run_id)
    state = run_info.state
    lc = state.life_cycle_state.value if state and state.life_cycle_state else "UNKNOWN"
    rs = state.result_state.value if state and state.result_state else ""
    msg = state.state_message or ""
    print(f"  {lc} / {rs} - {msg[:80]}")

    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        if rs != "SUCCESS":
            # Try to get output
            try:
                tasks = run_info.tasks or []
                for t in tasks:
                    output = w.jobs.get_run_output(run_id=t.run_id)
                    if output and output.error:
                        print(f"Error: {output.error}")
                    if output and output.notebook_output:
                        result = output.notebook_output.result
                        print(f"Notebook output: {result[:500] if result else 'none'}")
            except Exception as e:
                print(f"Could not get output: {e}")
        break

    time.sleep(20)

print(f"Job finished: {lc} / {rs}")
