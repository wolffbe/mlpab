"""
Full FTI pipeline for air quality PM2.5 forecasting on Databricks.
Runs everything on the platform via notebooks.
"""
import os
import base64
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import NotebookTask, Task, Source

w = WorkspaceClient()

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab94ed10
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]   # mlpab94ed10

# Get current user
me = w.current_user.me()
user = me.user_name
print(f"User: {user}")
print(f"Schema: {SCHEMA}")
print(f"Prefix: {PREFIX}")

NOTEBOOK_DIR = f"/Users/{user}/{PREFIX}"

# Read data files
with open("data/airquality_history.csv") as f:
    history_csv = f.read()

with open("data/forecast_days.csv") as f:
    forecast_csv = f.read()

# Create notebook directory
try:
    w.workspace.mkdirs(NOTEBOOK_DIR)
    print(f"Created directory: {NOTEBOOK_DIR}")
except Exception as e:
    print(f"Directory might exist: {e}")

# Build the notebook content as a Python notebook
notebook_code = f'''
# Databricks notebook source
# MAGIC %md # Air Quality PM2.5 FTI Pipeline

# COMMAND ----------
import os

SCHEMA = "{SCHEMA}"
CATALOG_SCHEMA = SCHEMA  # e.g., workspace.mlpab94ed10
CATALOG = CATALOG_SCHEMA.split(".")[0]
DB = CATALOG_SCHEMA.split(".")[1]
FG_NAME = "airqf4aae3"
TD_NAME = "airqtdf4aae3"
MODEL_NAME = "airqmodelf4aae3"
PRED_NAME = "airqpredf4aae3"

print(f"CATALOG: {{CATALOG}}, DB: {{DB}}")
print(f"Feature Group: {{CATALOG}}.{{DB}}.{{FG_NAME}}")

# COMMAND ----------
# MAGIC %md ## Step 1: Load history data and engineer features

# COMMAND ----------
import io

history_csv = """{history_csv}"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, lag, lit
from pyspark.sql.window import Window
import pyspark.sql.functions as F

spark = SparkSession.builder.getOrCreate()

# Parse CSV
lines = [l.strip() for l in history_csv.strip().split("\\n")]
header = lines[0].split(",")
rows = [dict(zip(header, l.split(","))) for l in lines[1:]]

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType

schema_def = StructType([
    StructField("date", StringType(), True),
    StructField("pm25_lag1", DoubleType(), True),
    StructField("temperature", DoubleType(), True),
    StructField("humidity", DoubleType(), True),
    StructField("wind_speed", DoubleType(), True),
    StructField("pressure", DoubleType(), True),
    StructField("precipitation", DoubleType(), True),
    StructField("pm25", DoubleType(), True),
])

rdd_rows = []
for r in rows:
    rdd_rows.append((
        r["date"],
        float(r["pm25_lag1"]),
        float(r["temperature"]),
        float(r["humidity"]),
        float(r["wind_speed"]),
        float(r["pressure"]),
        float(r["precipitation"]),
        float(r["pm25"]),
    ))

df_raw = spark.createDataFrame(rdd_rows, schema=schema_def)
df_raw = df_raw.withColumn("date", F.to_date(col("date")))
print(f"History rows: {{df_raw.count()}}")

# COMMAND ----------
# Feature engineering: rolling averages
w7 = Window.orderBy("date").rowsBetween(-6, 0)
w3 = Window.orderBy("date").rowsBetween(-2, 0)

df_feat = df_raw.withColumn("pm25_roll7", F.avg("pm25_lag1").over(w7)) \\
                .withColumn("pm25_roll3", F.avg("pm25_lag1").over(w3)) \\
                .withColumn("temp_roll3", F.avg("temperature").over(w3)) \\
                .withColumn("humidity_roll3", F.avg("humidity").over(w3))

print("Feature schema:")
df_feat.printSchema()
df_feat.show(5)

# COMMAND ----------
# MAGIC %md ## Step 2: Write feature group

# COMMAND ----------
spark.sql(f"USE CATALOG {{CATALOG}}")
spark.sql(f"USE SCHEMA {{DB}}")

# Write feature group table
df_feat.writeTo(f"{{CATALOG}}.{{DB}}.{{FG_NAME}}") \\
       .using("delta") \\
       .tableProperty("delta.enableChangeDataFeed", "true") \\
       .createOrReplace()

print(f"Feature group written: {{CATALOG}}.{{DB}}.{{FG_NAME}}")
spark.sql(f"SELECT COUNT(*) FROM {{CATALOG}}.{{DB}}.{{FG_NAME}}").show()

# COMMAND ----------
# MAGIC %md ## Step 3: Assemble training dataset

# COMMAND ----------
# Training dataset = feature group (all columns used for training)
df_train = spark.table(f"{{CATALOG}}.{{DB}}.{{FG_NAME}}")

# Save as training dataset table
df_train.writeTo(f"{{CATALOG}}.{{DB}}.{{TD_NAME}}") \\
        .using("delta") \\
        .createOrReplace()

print(f"Training dataset written: {{CATALOG}}.{{DB}}.{{TD_NAME}}")
print(f"Training dataset rows: {{df_train.count()}}")

# COMMAND ----------
# MAGIC %md ## Step 4: Train model and register

# COMMAND ----------
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user}/{PREFIX}/airquality_experiment")

# Load training data
pdf = df_train.toPandas()
pdf["date"] = pd.to_datetime(pdf["date"])
pdf = pdf.sort_values("date")

# Fill any NaN from rolling windows
pdf = pdf.fillna(method="bfill").fillna(method="ffill")

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed",
            "pressure", "precipitation", "pm25_roll7", "pm25_roll3",
            "temp_roll3", "humidity_roll3"]

X = pdf[FEATURES].values
y = pdf["pm25"].values

# Time-based split (last 20% for validation)
split_idx = int(len(pdf) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Train size: {{len(X_train)}}, Val size: {{len(X_val)}}")

with mlflow.start_run() as run:
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae = np.mean(np.abs(y_val - y_pred_val))

    print(f"Validation RMSE: {{rmse:.4f}}")
    print(f"Validation MAE: {{mae:.4f}}")

    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 4)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("val_rmse", rmse)

    # Log feature names as param
    mlflow.log_param("features", ",".join(FEATURES))

    # Create input example
    input_example = pd.DataFrame(X_val[:5], columns=FEATURES)
    signature = infer_signature(
        pd.DataFrame(X_train, columns=FEATURES),
        model.predict(X_train)
    )

    # Register model in Unity Catalog
    model_uri = f"models:/{{CATALOG}}.{{DB}}.{{MODEL_NAME}}"

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=f"{{CATALOG}}.{{DB}}.{{MODEL_NAME}}"
    )

    run_id = run.info.run_id
    print(f"Run ID: {{run_id}}")
    print(f"Model registered: {{CATALOG}}.{{DB}}.{{MODEL_NAME}}")

# COMMAND ----------
# MAGIC %md ## Step 5: Generate predictions for forecast days

# COMMAND ----------
forecast_csv = """{forecast_csv}"""

flines = [l.strip() for l in forecast_csv.strip().split("\\n")]
fheader = flines[0].split(",")
frows = [dict(zip(fheader, l.split(","))) for l in flines[1:]]

forecast_schema = StructType([
    StructField("date", StringType(), True),
    StructField("pm25_lag1", DoubleType(), True),
    StructField("temperature", DoubleType(), True),
    StructField("humidity", DoubleType(), True),
    StructField("wind_speed", DoubleType(), True),
    StructField("pressure", DoubleType(), True),
    StructField("precipitation", DoubleType(), True),
])

frdd_rows = []
for r in frows:
    frdd_rows.append((
        r["date"],
        float(r["pm25_lag1"]),
        float(r["temperature"]),
        float(r["humidity"]),
        float(r["wind_speed"]),
        float(r["pressure"]),
        float(r["precipitation"]),
    ))

df_forecast_raw = spark.createDataFrame(frdd_rows, schema=forecast_schema)
df_forecast_raw = df_forecast_raw.withColumn("date", F.to_date(col("date")))

# Need to engineer same features for forecast days
# Get the last rows from history to compute rolling windows
df_hist_for_roll = df_feat.select("date", "pm25_lag1", "temperature", "humidity",
                                   "wind_speed", "pressure", "precipitation")

# For forecast rows, we approximate rolling features using pm25_lag1 as proxy
# and recent historical values
df_forecast_feat = df_forecast_raw \\
    .withColumn("pm25_roll7", col("pm25_lag1")) \\
    .withColumn("pm25_roll3", col("pm25_lag1")) \\
    .withColumn("temp_roll3", col("temperature")) \\
    .withColumn("humidity_roll3", col("humidity"))

print("Forecast features:")
df_forecast_feat.show()

# Generate predictions using the model
# Load the registered model
import mlflow.pyfunc

model_version_uri = f"models:/{{CATALOG}}.{{DB}}.{{MODEL_NAME}}/1"
loaded_model = mlflow.pyfunc.load_model(model_version_uri)

pdf_forecast = df_forecast_feat.toPandas()
pdf_forecast["date"] = pd.to_datetime(pdf_forecast["date"])

# Fill any missing features
pdf_forecast = pdf_forecast.fillna(method="bfill").fillna(method="ffill")

X_forecast = pdf_forecast[FEATURES].values
preds = loaded_model.predict(pd.DataFrame(X_forecast, columns=FEATURES))

pdf_forecast["pm25_pred"] = preds
print("Predictions:")
print(pdf_forecast[["date", "pm25_pred"]])

# COMMAND ----------
# Write predictions table
from pyspark.sql.types import TimestampType

pred_pdf = pdf_forecast[["date", "pm25_pred"]].copy()
pred_pdf["date"] = pred_pdf["date"].dt.strftime("%Y-%m-%d")

pred_schema = StructType([
    StructField("date", StringType(), True),
    StructField("pm25_pred", DoubleType(), True),
])

pred_rows = [(str(row["date"]), float(row["pm25_pred"])) for _, row in pred_pdf.iterrows()]
df_pred = spark.createDataFrame(pred_rows, schema=pred_schema)
df_pred = df_pred.withColumn("date", F.to_date(col("date")))

df_pred.writeTo(f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}") \\
       .using("delta") \\
       .tableProperty("delta.enableChangeDataFeed", "true") \\
       .createOrReplace()

print(f"Predictions written to: {{CATALOG}}.{{DB}}.{{PRED_NAME}}")
df_pred.show()

# COMMAND ----------
# MAGIC %md ## Step 6: Enable online table for low-latency lookup

# COMMAND ----------
# Online table via Databricks REST (for low-latency lookup)
# The predictions table is a Delta table with change data feed enabled

from databricks.sdk import WorkspaceClient
wc = WorkspaceClient()

online_table_name = f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}_online"

try:
    ot_spec = {{
        "name": online_table_name,
        "spec": {{
            "source_table_full_name": f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
            "primary_key_columns": ["date"],
            "run_triggered": {{
                "triggered_update_continuous": True
            }}
        }}
    }}

    result = wc.api_client.do("POST", "/api/2.0/online-tables", body=ot_spec)
    print(f"Online table created: {{online_table_name}}")
    print(result)
except Exception as e:
    print(f"Online table creation: {{e}}")
    # Try with run_continuously
    try:
        ot_spec2 = {{
            "name": online_table_name,
            "spec": {{
                "source_table_full_name": f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
                "primary_key_columns": ["date"],
                "run_continuously": {{}}
            }}
        }}
        result2 = wc.api_client.do("POST", "/api/2.0/online-tables", body=ot_spec2)
        print(f"Online table created (v2): {{online_table_name}}")
        print(result2)
    except Exception as e2:
        print(f"Online table creation v2: {{e2}}")

# COMMAND ----------
print("=== PIPELINE COMPLETE ===")
print(f"Feature group: {{CATALOG}}.{{DB}}.{{FG_NAME}}")
print(f"Training dataset: {{CATALOG}}.{{DB}}.{{TD_NAME}}")
print(f"Model: {{CATALOG}}.{{DB}}.{{MODEL_NAME}}")
print(f"Predictions: {{CATALOG}}.{{DB}}.{{PRED_NAME}}")
'''

# Write the notebook
notebook_path = f"{NOTEBOOK_DIR}/airquality_pipeline"
notebook_content = notebook_code

# Encode as base64
content_b64 = base64.b64encode(notebook_content.encode()).decode()

print(f"Uploading notebook to: {notebook_path}")

w.workspace.import_(
    path=notebook_path,
    format="SOURCE",
    language="PYTHON",
    content=content_b64,
    overwrite=True,
)

print(f"Notebook uploaded: {notebook_path}")

# Run the notebook as a one-time job
job_name = f"{PREFIX}_airquality_pipeline"

print(f"Creating and running job: {job_name}")

run = w.jobs.submit(
    run_name=job_name,
    tasks=[
        Task(
            task_key="pipeline",
            notebook_task=NotebookTask(
                notebook_path=notebook_path,
                base_parameters={},
                source=Source.WORKSPACE,
            ),
            new_cluster={
                "spark_version": "15.4.x-cpu-ml-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 2,
                "spark_conf": {
                    "spark.databricks.cluster.profile": "serverless"
                },
            },
        )
    ]
)

run_id = run.bind_name if hasattr(run, 'bind_name') else run.run_id
print(f"Job submitted, run_id: {run.run_id}")

# Wait for completion
import time
while True:
    run_info = w.jobs.get_run(run_id=run.run_id)
    state = run_info.state
    life_cycle = state.life_cycle_state.value if state and state.life_cycle_state else "UNKNOWN"
    result_state = state.result_state.value if state and state.result_state else ""
    print(f"  Status: {life_cycle} / {result_state}")

    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        print(f"Job finished: {life_cycle} / {result_state}")
        if result_state == "FAILED":
            # Get error
            print(f"Error message: {state.state_message}")
        break

    time.sleep(30)

print("Done!")
